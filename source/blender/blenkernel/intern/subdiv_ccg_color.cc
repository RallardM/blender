/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 *
 * GBrush: fill the colors of a #SubdivCCG from the grid color layer (#CD_GRID_PAINT_COLOR).
 */

#include <algorithm>
#include <cmath>

#include "BLI_task.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_ccg.hh"
#include "BKE_customdata.hh"
#include "BKE_subdiv_ccg.hh"

namespace blender {

void BKE_subdiv_ccg_colors_init_from_paint(SubdivCCG &subdiv_ccg, const Mesh &mesh)
{
  if (subdiv_ccg.colors.is_empty() || subdiv_ccg.grid_size < 2) {
    return;
  }
  const GridPaintColor *layer = static_cast<const GridPaintColor *>(
      CustomData_get_layer(&mesh.corner_data, CD_GRID_PAINT_COLOR));
  if (!layer) {
    return;
  }
  const int grid_size = subdiv_ccg.grid_size;
  const int64_t grid_area = subdiv_ccg.grid_area;
  const int grids_num = std::min<int>(subdiv_ccg.grids_num, mesh.corners_num);

  /* The grid of a corner and the CCG grid of the same index share their (x, y) parameter space,
   * so a grid can be copied directly when the sizes match and resampled otherwise. */
  threading::parallel_for(IndexRange(grids_num), 256, [&](const IndexRange range) {
    for (const int grid : range) {
      const GridPaintColor &src = layer[grid];
      if (src.data == nullptr || src.level < 1) {
        continue;
      }
      const int src_size = CCG_grid_size(int(src.level));
      if (src_size < 2) {
        continue;
      }
      const unsigned char *src_data = src.data;
      MutableSpan<uchar4> dst = subdiv_ccg.colors.as_mutable_span().slice(grid * grid_area,
                                                                          grid_area);
      for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
          uchar4 &out = dst[y * grid_size + x];
          if (src_size == grid_size) {
            const unsigned char *texel = src_data + 4 * (y * src_size + x);
            out = uchar4(texel[0], texel[1], texel[2], texel[3]);
            continue;
          }
          const float fx = x * float(src_size - 1) / float(grid_size - 1);
          const float fy = y * float(src_size - 1) / float(grid_size - 1);
          const int x0 = std::min(int(fx), src_size - 2);
          const int y0 = std::min(int(fy), src_size - 2);
          const float tx = fx - x0;
          const float ty = fy - y0;
          const unsigned char *t00 = src_data + 4 * (y0 * src_size + x0);
          const unsigned char *t10 = t00 + 4;
          const unsigned char *t01 = t00 + 4 * src_size;
          const unsigned char *t11 = t01 + 4;
          unsigned char result[4];
          for (int channel = 0; channel < 4; channel++) {
            const float bottom = t00[channel] + (t10[channel] - t00[channel]) * tx;
            const float top = t01[channel] + (t11[channel] - t01[channel]) * tx;
            const float value = bottom + (top - bottom) * ty;
            result[channel] = (unsigned char)(std::clamp(int(value + 0.5f), 0, 255));
          }
          out = uchar4(result[0], result[1], result[2], result[3]);
        }
      }
    }
  });
}

}  // namespace blender
