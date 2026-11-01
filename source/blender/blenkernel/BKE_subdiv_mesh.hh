/* SPDX-FileCopyrightText: 2018 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */

#pragma once

#include <memory>

#include "BLI_math_vector_types.hh"
#include "BLI_offset_indices.hh"

namespace blender {

struct Mesh;

namespace bke::subdiv {

struct Subdiv;

/** GBrush: evaluates an 8-bit RGBA color at a point of the limit surface, thread safe. */
struct ColorSampler {
  virtual ~ColorSampler() = default;
  /** \param ptex_face_index, u, v: the same coordinates the displacement evaluator receives. */
  virtual uchar4 sample(int ptex_face_index, float u, float v) const = 0;
};

/**
 * GBrush: sampler reading the multires grid color layer of \a mesh, null when the mesh has no
 * such layer. Grids without data read as opaque white. The mesh has to outlive the sampler.
 */
std::unique_ptr<ColorSampler> grid_color_sampler_create(const Mesh &mesh);

struct ToMeshSettings {
  /**
   * Resolution at which regular PTEX (created for quad face) are being
   * evaluated. This defines how many vertices final mesh will have: every
   * regular PTEX has resolution^2 vertices. Special (irregular, or PTEX
   * created for a corner of non-quad face) will have resolution of
   * `resolution - 1`.
   */
  int resolution = -1;
  /** When true, only edges emitted from coarse ones will be displayed. */
  bool use_optimal_display = false;
  /** GBrush: when set, the result gets a point color attribute with the sampled colors. */
  const ColorSampler *color_sampler = nullptr;
};

/** Create real hi-res mesh from subdivision, all geometry is "real". */
Mesh *subdiv_to_mesh(Subdiv *subdiv, const ToMeshSettings *settings, const Mesh *coarse_mesh);

/**
 * Interpolate a position along the `coarse_edge` at the relative `u` coordinate.
 * If `is_simple` is false, this will perform a B-Spline interpolation using the edge neighbors,
 * otherwise a linear interpolation will be done base on the edge vertices.
 */
float3 mesh_interpolate_position_on_edge(Span<float3> coarse_positions,
                                         Span<int2> coarse_edges,
                                         GroupedSpan<int> vert_to_edge_map,
                                         int coarse_edge_index,
                                         bool is_simple,
                                         float u);

/**
 * Calculate positions position of the given mesh vertices at the limit surface of the mesh.
 *
 * The limit_positions is to be sized at exactly the number of the base mesh vertices.
 */
void calculate_limit_positions(Mesh *mesh, MutableSpan<float3> limit_positions);

}  // namespace bke::subdiv
}  // namespace blender
