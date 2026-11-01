/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 *
 * GBrush: evaluate the multires grid color layer (#CD_GRID_PAINT_COLOR) at limit surface points.
 */

#include <algorithm>

#include "BLI_array.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_ccg.hh"
#include "BKE_customdata.hh"
#include "BKE_subdiv.hh"
#include "BKE_subdiv_mesh.hh"

namespace blender::bke::subdiv {

namespace {

struct PtexFaceCorner {
  int face_index;
  int corner;
};

class GridColorSampler final : public ColorSampler {
 private:
  OffsetIndices<int> faces_;
  const GridPaintColor *layer_;
  Array<PtexFaceCorner> ptex_face_corner_;

 public:
  GridColorSampler(const Mesh &mesh, const GridPaintColor *layer)
      : faces_(mesh.faces()), layer_(layer)
  {
    /* Same ptex numbering as the mask evaluator: one ptex for a quad, one per corner otherwise. */
    int ptex_faces_num = 0;
    for (const int face_index : faces_.index_range()) {
      ptex_faces_num += (faces_[face_index].size() == 4) ? 1 : int(faces_[face_index].size());
    }
    ptex_face_corner_.reinitialize(ptex_faces_num);
    int ptex_face_index = 0;
    for (const int face_index : faces_.index_range()) {
      const IndexRange face = faces_[face_index];
      if (face.size() == 4) {
        ptex_face_corner_[ptex_face_index++] = {face_index, 0};
      }
      else {
        for (const int corner : IndexRange(face.size())) {
          ptex_face_corner_[ptex_face_index++] = {face_index, corner};
        }
      }
    }
  }

  uchar4 sample(const int ptex_face_index, const float u, const float v) const override
  {
    const PtexFaceCorner ptex = ptex_face_corner_[ptex_face_index];
    const IndexRange face = faces_[ptex.face_index];
    const int start_grid_index = face.start() + ptex.corner;
    const GridPaintColor *grid;
    float grid_u, grid_v;
    if (face.size() == 4) {
      float corner_u, corner_v;
      const int corner = rotate_quad_to_corner(u, v, &corner_u, &corner_v);
      grid = &layer_[start_grid_index + corner];
      ptex_face_uv_to_grid_uv(corner_u, corner_v, &grid_u, &grid_v);
    }
    else {
      grid = &layer_[start_grid_index];
      ptex_face_uv_to_grid_uv(u, v, &grid_u, &grid_v);
    }
    if (grid->data == nullptr || grid->level < 1) {
      return uchar4(255, 255, 255, 255);
    }
    const int size = CCG_grid_size(int(grid->level));
    if (size < 2) {
      return uchar4(255, 255, 255, 255);
    }
    const float fx = std::clamp(grid_u, 0.0f, 1.0f) * (size - 1);
    const float fy = std::clamp(grid_v, 0.0f, 1.0f) * (size - 1);
    const int x0 = std::min(int(fx), size - 2);
    const int y0 = std::min(int(fy), size - 2);
    const float tx = fx - x0;
    const float ty = fy - y0;
    const unsigned char *t00 = grid->data + 4 * (y0 * size + x0);
    const unsigned char *t10 = t00 + 4;
    const unsigned char *t01 = t00 + 4 * size;
    const unsigned char *t11 = t01 + 4;
    unsigned char result[4];
    for (int channel = 0; channel < 4; channel++) {
      const float bottom = t00[channel] + (t10[channel] - t00[channel]) * tx;
      const float top = t01[channel] + (t11[channel] - t01[channel]) * tx;
      const float value = bottom + (top - bottom) * ty;
      result[channel] = (unsigned char)(std::clamp(int(value + 0.5f), 0, 255));
    }
    return uchar4(result[0], result[1], result[2], result[3]);
  }
};

}  // namespace

std::unique_ptr<ColorSampler> grid_color_sampler_create(const Mesh &mesh)
{
  const GridPaintColor *layer = static_cast<const GridPaintColor *>(
      CustomData_get_layer(&mesh.corner_data, CD_GRID_PAINT_COLOR));
  if (!layer || mesh.corners_num == 0) {
    return nullptr;
  }
  return std::make_unique<GridColorSampler>(mesh, layer);
}

}  // namespace blender::bke::subdiv
