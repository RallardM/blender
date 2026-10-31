/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edsculpt
 *
 * The Draw Paint brush sculpts and paints in a single pass over the affected nodes, like the
 * combined sculpt and polypaint brushes found in other sculpting applications.
 *
 * The brush shares one set of falloff, mask, automasking and texture factors between the
 * displacement and the color blend, so the two always agree and the nodes are only visited once.
 *
 * Only regular mesh sculpting with an active color attribute can store the color part. In every
 * other case (multires, dynamic topology, no color attribute) the brush behaves as the Draw brush.
 */

#include "editors/sculpt_paint/mesh/brushes/brushes.hh"

#include "DNA_brush_types.h"
#include "DNA_mesh_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "BKE_attribute.hh"
#include "BKE_brush.hh"
#include "BKE_ccg.hh"
#include "BKE_colortools.hh"
#include "BKE_mesh.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_paint_types.hh"
#include "BKE_subdiv_ccg.hh"

#include "BLI_enumerable_thread_specific.hh"
#include "BLI_math_color.h"
#include "BLI_math_vector.hh"
#include "BLI_task.hh"

#include "IMB_colormanagement.hh"
#include "IMB_imbuf.hh"

#include "editors/sculpt_paint/mesh/mesh_brush_common.hh"
#include "editors/sculpt_paint/mesh/sculpt_color.hh"
#include "editors/sculpt_paint/mesh/sculpt_intern.hh"

namespace blender::ed::sculpt_paint::brushes {

inline namespace draw_paint_cc {

struct LocalData {
  Vector<float> factors;
  Vector<float> distances;
  Vector<float3> translations;
  Vector<float4> colors;
  Vector<float3> positions;
};

/**
 * Opacity of the color part for one brush step. Linear in the brush strength setting (unlike the
 * displacement, which squares it) so that the color opacity matches the Paint brush.
 */
static float color_opacity_get(const Sculpt &sd, const Brush &brush, const StrokeCache &cache)
{
  const bke::PaintRuntime &paint_runtime = *sd.paint.runtime;
  const float alpha = BKE_brush_alpha_get(&sd.paint, &brush);
  const float pressure = BKE_brush_use_alpha_pressure(&brush) ?
                             BKE_curvemapping_evaluateF(brush.curve_strength, 0, cache.pressure) :
                             1.0f;
  return alpha * pressure * paint_runtime.overlap_factor;
}

static void calc_faces(const Depsgraph &depsgraph,
                       const Sculpt &sd,
                       const Brush &brush,
                       const float3 &offset,
                       const float3 &brush_color,
                       const float color_opacity,
                       const MeshAttributeData &attribute_data,
                       const Span<float3> vert_normals,
                       const OffsetIndices<int> faces,
                       const Span<int> corner_verts,
                       const GroupedSpan<int> vert_to_face_map,
                       const bke::pbvh::MeshNode &node,
                       Object &object,
                       LocalData &tls,
                       const PositionDeformData &position_data,
                       bke::GSpanAttributeWriter &color_attribute)
{
  const SculptSession &ss = *object.runtime->sculpt_session;

  const Span<int> verts = node.verts();

  calc_factors_common_mesh_indexed(depsgraph,
                                   brush,
                                   object,
                                   attribute_data,
                                   position_data.eval,
                                   vert_normals,
                                   node,
                                   tls.factors,
                                   tls.distances);
  const Span<float> factors = tls.factors;

  /* Sculpt part: identical to the Draw brush. */
  tls.translations.resize(verts.size());
  const MutableSpan<float3> translations = tls.translations;
  translations_from_offset_and_factors(offset, factors, translations);

  clip_and_lock_translations(sd, ss, position_data.eval, verts, translations);
  position_data.deform(translations, verts);

  /* Paint part: blend the brush color using the same factors. */
  tls.colors.resize(verts.size());
  const MutableSpan<float4> colors = tls.colors;
  for (const int i : verts.index_range()) {
    colors[i] = color::color_vert_get(faces,
                                      corner_verts,
                                      vert_to_face_map,
                                      color_attribute.span,
                                      color_attribute.domain,
                                      verts[i]);
  }

  for (const int i : verts.index_range()) {
    const float opacity = math::clamp(factors[i] * color_opacity, 0.0f, 1.0f);
    if (opacity <= 0.0f) {
      continue;
    }
    /* The blend functions expect the source color to be premultiplied. */
    const float4 source(brush_color * opacity, opacity);
    float4 blended;
    IMB_blend_color_float(blended, colors[i], source, IMB_BlendMode(brush.blend));
    color::color_vert_set(faces,
                          corner_verts,
                          vert_to_face_map,
                          color_attribute.domain,
                          verts[i],
                          math::clamp(blended, 0.0f, 1.0f),
                          color_attribute.span);
  }
}

/** Grid colors are 8-bit sRGB encoded like byte color attributes, blending happens in linear. */
static float4 grid_color_to_float(const uchar4 &stored)
{
  float4 color;
  rgba_uchar_to_float(color, reinterpret_cast<const uchar *>(&stored));
  IMB_colormanagement_srgb_to_scene_linear_v3(color, color);
  return color;
}

static uchar4 grid_color_from_float(const float4 &color)
{
  float4 encoded;
  IMB_colormanagement_scene_linear_to_srgb_v3(encoded, color);
  encoded[3] = color[3];
  uchar4 stored;
  rgba_float_to_uchar(reinterpret_cast<uchar *>(&stored), encoded);
  return stored;
}

/** Sculpt and paint one multires node, the colors live in #SubdivCCG::colors. */
static void calc_grids(const Depsgraph &depsgraph,
                       const Sculpt &sd,
                       Object &object,
                       const Brush &brush,
                       const float3 &offset,
                       const float3 &brush_color,
                       const float color_opacity,
                       const bke::pbvh::GridsNode &node,
                       LocalData &tls)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  SubdivCCG &subdiv_ccg = *ss.subdiv_ccg;
  const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);

  const Span<int> grids = node.grids();
  const MutableSpan positions = gather_grids_positions(subdiv_ccg, grids, tls.positions);

  calc_factors_common_grids(depsgraph, brush, object, positions, node, tls.factors, tls.distances);
  const Span<float> factors = tls.factors;

  /* Sculpt part: identical to the Draw brush. */
  tls.translations.resize(positions.size());
  const MutableSpan<float3> translations = tls.translations;
  translations_from_offset_and_factors(offset, factors, translations);
  clip_and_lock_translations(sd, ss, positions, translations);
  apply_translations(translations, grids, subdiv_ccg);

  /* Paint part: blend the brush color into the 8-bit grid colors with the same factors. */
  MutableSpan<uchar4> colors = subdiv_ccg.colors;
  for (const int i : grids.index_range()) {
    const IndexRange node_range = bke::ccg::grid_range(key, i);
    const IndexRange grid_range = bke::ccg::grid_range(key, grids[i]);
    for (const int offset_in_grid : IndexRange(key.grid_area)) {
      const float opacity = math::clamp(factors[node_range[offset_in_grid]] * color_opacity,
                                        0.0f,
                                        1.0f);
      if (opacity <= 0.0f) {
        continue;
      }
      uchar4 &stored = colors[grid_range[offset_in_grid]];
      const float4 current = grid_color_to_float(stored);
      /* The blend functions expect the source color to be premultiplied. */
      const float4 source(brush_color * opacity, opacity);
      float4 blended;
      IMB_blend_color_float(blended, current, source, IMB_BlendMode(brush.blend));
      stored = grid_color_from_float(math::clamp(blended, 0.0f, 1.0f));
    }
  }
}

static void do_draw_paint_grids(const Depsgraph &depsgraph,
                                const Sculpt &sd,
                                Object &object,
                                const IndexMask &node_mask)
{
  const SculptSession &ss = *object.runtime->sculpt_session;
  const StrokeCache &cache = *ss.cache;
  const Brush &brush = *BKE_paint_brush_for_read(&sd.paint);

  const float3 effective_normal = tilt_effective_normal_get(ss, brush);
  const float3 offset = effective_normal * cache.radius * cache.scale * cache.bstrength;
  const float3 brush_color = cache.toggle_settings.invert ?
                                 BKE_brush_secondary_color_get(&sd.paint, &brush) :
                                 BKE_brush_color_get(&sd.paint, &brush);
  const float color_opacity = color_opacity_get(sd, brush, cache);

  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  SubdivCCG &subdiv_ccg = *object.runtime->sculpt_session->subdiv_ccg;
  MutableSpan<float3> positions = subdiv_ccg.positions;
  MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
  threading::EnumerableThreadSpecific<LocalData> all_tls;
  node_mask.foreach_index(
      [&](const int i) {
        LocalData &tls = all_tls.local();
        calc_grids(depsgraph, sd, object, brush, offset, brush_color, color_opacity, nodes[i], tls);
        bke::pbvh::update_node_bounds_grids(subdiv_ccg.grid_area, positions, nodes[i]);
      },
      exec_mode::grain_size(1));

  subdiv_ccg.dirty.colors = true;
  pbvh.tag_positions_changed(node_mask);
  pbvh.tag_attribute_changed(node_mask, SUBDIV_CCG_COLOR_NAME);
  pbvh.flush_bounds_to_parents();
}

}  // namespace draw_paint_cc

bool draw_paint_has_grid_color_target(const Object &object)
{
  const bke::pbvh::Tree *pbvh = bke::object::pbvh_get(object);
  if (!pbvh || pbvh->type() != bke::pbvh::Type::Grids) {
    return false;
  }
  const SculptSession &ss = *object.runtime->sculpt_session;
  return ss.subdiv_ccg && !ss.subdiv_ccg->colors.is_empty();
}

bool draw_paint_has_color_target(const Object &object)
{
  const bke::pbvh::Tree *pbvh = bke::object::pbvh_get(object);
  if (!pbvh || pbvh->type() != bke::pbvh::Type::Mesh) {
    return false;
  }
  const Mesh &mesh = *id_cast<const Mesh *>(object.data);
  return bool(color::active_color_attribute(mesh));
}

void do_draw_paint_brush(const Depsgraph &depsgraph,
                         const Sculpt &sd,
                         Object &object,
                         const IndexMask &node_mask)
{
  PRF_scope(ProfileCategory::Editor);

  if (draw_paint_has_grid_color_target(object)) {
    do_draw_paint_grids(depsgraph, sd, object, node_mask);
    return;
  }

  if (!draw_paint_has_color_target(object)) {
    do_draw_brush(depsgraph, sd, object, node_mask);
    return;
  }

  const SculptSession &ss = *object.runtime->sculpt_session;
  const StrokeCache &cache = *ss.cache;
  const Brush &brush = *BKE_paint_brush_for_read(&sd.paint);

  const float3 effective_normal = tilt_effective_normal_get(ss, brush);
  const float3 offset = effective_normal * cache.radius * cache.scale * cache.bstrength;

  const float3 brush_color = cache.toggle_settings.invert ?
                                 BKE_brush_secondary_color_get(&sd.paint, &brush) :
                                 BKE_brush_color_get(&sd.paint, &brush);
  const float color_opacity = color_opacity_get(sd, brush, cache);

  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  const MeshAttributeData attribute_data(mesh);
  const PositionDeformData position_data(depsgraph, object);
  const Span<float3> vert_normals = bke::pbvh::vert_normals_eval(depsgraph, object);
  const OffsetIndices<int> faces = mesh.faces();
  const Span<int> corner_verts = mesh.corner_verts();
  const GroupedSpan<int> vert_to_face_map = mesh.vert_to_face_map();
  bke::GSpanAttributeWriter color_attribute = color::active_color_attribute_for_write(mesh);
  if (!color_attribute) {
    do_draw_brush(depsgraph, sd, object, node_mask);
    return;
  }

  MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
  threading::EnumerableThreadSpecific<LocalData> all_tls;
  node_mask.foreach_index(
      [&](const int i) {
        LocalData &tls = all_tls.local();
        calc_faces(depsgraph,
                   sd,
                   brush,
                   offset,
                   brush_color,
                   color_opacity,
                   attribute_data,
                   vert_normals,
                   faces,
                   corner_verts,
                   vert_to_face_map,
                   nodes[i],
                   object,
                   tls,
                   position_data,
                   color_attribute);
        bke::pbvh::update_node_bounds_mesh(position_data.eval, nodes[i]);
      },
      exec_mode::grain_size(1));

  pbvh.tag_positions_changed(node_mask);
  pbvh.tag_attribute_changed(node_mask, mesh.active_color_attribute);
  pbvh.flush_bounds_to_parents();
  color_attribute.finish();
}

}  // namespace blender::ed::sculpt_paint::brushes
