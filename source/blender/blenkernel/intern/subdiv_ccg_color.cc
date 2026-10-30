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

#include "BLI_array.hh"
#include "BLI_task.hh"

#include "MEM_guardedalloc.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_ccg.hh"
#include "BKE_customdata.hh"
#include "BKE_subdiv_ccg.hh"

namespace blender {

/** Copy or bilinearly resample one stored grid to a grid of \a dst_size by \a dst_size. */
static void sample_grid(const GridPaintColor &src, const int dst_size, MutableSpan<uchar4> dst)
{
  const int src_size = CCG_grid_size(int(src.level));
  const unsigned char *src_data = src.data;
  for (int y = 0; y < dst_size; y++) {
    for (int x = 0; x < dst_size; x++) {
      uchar4 &out = dst[y * dst_size + x];
      if (src_size == dst_size) {
        const unsigned char *texel = src_data + 4 * (y * src_size + x);
        out = uchar4(texel[0], texel[1], texel[2], texel[3]);
        continue;
      }
      const float fx = x * float(src_size - 1) / float(dst_size - 1);
      const float fy = y * float(src_size - 1) / float(dst_size - 1);
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

static bool grid_is_valid(const GridPaintColor &gpc)
{
  return gpc.data != nullptr && gpc.level >= 1 && CCG_grid_size(int(gpc.level)) >= 2;
}

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
      if (!grid_is_valid(layer[grid])) {
        continue;
      }
      sample_grid(layer[grid],
                  grid_size,
                  subdiv_ccg.colors.as_mutable_span().slice(grid * grid_area, grid_area));
    }
  });
}

void BKE_subdiv_ccg_colors_write_to_paint(const SubdivCCG &subdiv_ccg, Mesh &mesh)
{
  if (subdiv_ccg.colors.is_empty() || subdiv_ccg.grid_size < 2) {
    return;
  }
  GridPaintColor *layer = static_cast<GridPaintColor *>(
      CustomData_get_layer_for_write(&mesh.corner_data, CD_GRID_PAINT_COLOR, mesh.corners_num));
  if (!layer) {
    return;
  }
  const int grid_size = subdiv_ccg.grid_size;
  const int64_t grid_area = subdiv_ccg.grid_area;
  const int grids_num = std::min<int>(subdiv_ccg.grids_num, mesh.corners_num);

  threading::parallel_for(IndexRange(grids_num), 64, [&](const IndexRange range) {
    Array<uchar4> resampled;
    for (const int grid : range) {
      GridPaintColor &dst = layer[grid];
      const Span<uchar4> src = subdiv_ccg.colors.as_span().slice(grid * grid_area, grid_area);
      if (grid_is_valid(dst)) {
        if (CCG_grid_size(int(dst.level)) == grid_size) {
          const unsigned char *stored = dst.data;
          bool same = true;
          for (int i = 0; i < grid_area && same; i++) {
            same = stored[4 * i] == src[i].x && stored[4 * i + 1] == src[i].y &&
                   stored[4 * i + 2] == src[i].z && stored[4 * i + 3] == src[i].w;
          }
          if (same) {
            continue;
          }
        }
        else {
          resampled.reinitialize(grid_area);
          sample_grid(dst, grid_size, resampled);
          bool same = true;
          for (int i = 0; i < grid_area && same; i++) {
            same = resampled[i] == src[i];
          }
          if (same) {
            continue;
          }
        }
      }
      else {
        /* Never painted and still white, keep the grid empty. */
        bool white = true;
        for (int i = 0; i < grid_area && white; i++) {
          white = src[i] == uchar4(255, 255, 255, 255);
        }
        if (white) {
          continue;
        }
      }
      /* Store the grid at the resolution of the CCG. */
      MEM_SAFE_DELETE(dst.data);
      dst.data = MEM_new_array_uninitialized<unsigned char>(size_t(grid_area) * 4,
                                                            "GridPaintColor.data");
      dst.level = uint(subdiv_ccg.level);
      for (int i = 0; i < grid_area; i++) {
        dst.data[4 * i] = src[i].x;
        dst.data[4 * i + 1] = src[i].y;
        dst.data[4 * i + 2] = src[i].z;
        dst.data[4 * i + 3] = src[i].w;
      }
    }
  });
}

}  // namespace blender
