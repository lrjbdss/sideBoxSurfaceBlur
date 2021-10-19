#include "side_box_surface_blur.h"
#include <cmath>
#include <memory.h>
// #include "omp.h"

void SideSurface::_init_weight(uchar thresh)
{
    for (int i = 0; i < 2.5 * thresh; ++i)
    {
        weight_table[i] = 2500 - i * 1000 / thresh;
    }
}

SideSurface::SideSurface(int h, int w, int r, uchar thresh)
    : height(h),
      width(w),
      radius(r)
{
    weight_table = (int *)malloc(int(2.5 * thresh) * sizeof(int));
    _init_weight(thresh);

    // 将(4r+1)*(4r+1)的数组简化合并为5*5的数组
    cellIdx = (int *)malloc((4 * radius + 1) * sizeof(int));
    cellIdx[2 * r] = 2;
    for (int i = 0; i < 2 * r; ++i)
        cellIdx[i] = i / r;
    for (int i = 2 * r + 1; i < 4 * r + 1; ++i)
        cellIdx[i] = (i - 1) / r + 1;

    pwCell = (int *)malloc(25 * sizeof(int));
    wCell = (int *)malloc(25 * sizeof(int));
    sumColPW = (int *)malloc(15 * sizeof(int));
    sumColW = (int *)malloc(15 * sizeof(int));
}

void SideSurface::run(const uchar *input, uchar *output)
{
    const uchar *x = input;
    uchar *xx = output;
    for (int h = 0; h < height; ++h)
    {
        int R = 2 * radius;
        for (int w = 0; w < width; ++w)
        {
            memset(pwCell, 0, 25 * sizeof(int));
            memset(wCell, 0, 25 * sizeof(int));
            int h_start = std::max(h - R, 0);
            int w_start = std::max(w - R, 0);
            int h_end = std::min(h + R, height - 1);
            int w_end = std::min(w + R, width - 1);
            int delta_w = width - h_end + h_start;
            int *cellIdxH = cellIdx + h_start - h + R;
            const uchar *xi = input + h_start * width + w_start;
            for (int kernel_h = h_start; kernel_h <= h_end; ++kernel_h)
            {
                int *cellIdxW = cellIdx + w_start - w + R;
                for (int kernel_w = w_start; kernel_w <= w_end; ++kernel_w)
                {
                    int offset = 5 * *cellIdxH + *cellIdxW++;
                    float weight_i = weight_table[std::abs(*x - *xi)];
                    wCell[offset] += weight_i;
                    pwCell[offset] += weight_i * *xi;
                    xi++;
                }
                cellIdxH++;
                xi += delta_w;
            }

            int *offsetColPW = sumColPW;
            int *offsetCellPW = pwCell;
            int *offsetColW = sumColW;
            int *offsetCellW = wCell;
            for (int row = 0; row < 5; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    *offsetColPW++ = *offsetCellPW + *(offsetCellPW + 1) + *(offsetCellPW + 2);
                    *offsetColW++ = *offsetCellW + *(offsetCellW + 1) + *(offsetCellW + 2);
                    offsetCellPW++;
                    offsetCellW++;
                }
                offsetCellPW += 2;
                offsetCellW += 2;
            }
            _calPix(x, xx);
            x++;
            xx++;
        }
    }
}

void SideSurface::_calPix(const uchar *x, uchar *xx)
{
    int candidate, diff, min_diff = 255;
    int *offsetColPW = sumColPW;
    int *offsetColW = sumColW;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            if (row != 1 || col != 1)
            {
                int weight = *offsetColW + *(offsetColW + 3) + *(offsetColW + 6);
                if (weight != 0)
                {
                    candidate = (*offsetColPW + *(offsetColPW + 3) + *(offsetColPW + 6)) / weight;
                    diff = std::abs(candidate - *x);
                    if (diff == 0)
                    {
                        *xx = *x;
                        return;
                    }
                    if (diff < min_diff)
                    {
                        *xx = (uchar)candidate;
                        min_diff = diff;
                    }
                }
            }
            offsetColPW++;
            offsetColW++;
        }
    }
    if (min_diff == 255)
        *xx = *x;
    return;
}

SideSurface::~SideSurface()
{
    if (cellIdx != NULL)
    {
        free(cellIdx);
        cellIdx = NULL;
    }
    if (weight_table != NULL)
    {
        free(weight_table);
        weight_table = NULL;
    }
    if (pwCell != NULL)
    {
        free(pwCell);
        pwCell = NULL;
    }
    if (wCell != NULL)
    {
        free(wCell);
        wCell = NULL;
    }
    if (sumColPW != NULL)
    {
        free(sumColPW);
        sumColPW = NULL;
    }
    if (sumColW != NULL)
    {
        free(sumColW);
        sumColW = NULL;
    }
}