#include "side_box_surface_blur.h"
#include <cmath>

void SideSurface::_init_weight(uchar thresh)
{
    weights.resize(256);
    float T = 2.5 * thresh;
    for (int i = 0; i < 256; ++i)
    {
        if (i > T)
            weights[i] = 0;
        else
            weights[i] = 1 - i / T;
    }
}

SideSurface::SideSurface(int h, int w, int r, uchar thresh)
    : height(h),
      width(w),
      radius(r)
{
    _init_weight(thresh);
    sumColPW.resize(5);
    for (auto &line : sumColPW)
    {
        line.resize(3, 0);
    }
    sumColW.resize(5);
    for (auto &line : sumColW)
    {
        line.resize(3, 0);
    }

    sumPW.resize(3);
    for (auto &line : sumPW)
    {
        line.resize(3, 0);
    }
    sumW.resize(3);
    for (auto &line : sumW)
    {
        line.resize(3, 0);
    }
    pwCell.resize(5);
    for (auto &line : pwCell)
    {
        line.resize(5, 0);
    }
    wCell.resize(5);
    for (auto &line : wCell)
    {
        line.resize(5, 0);
    }

    idx.resize(4 * r + 1);
    idx[2 * r] = 2;
    for (int i = 0; i < 2 * r; ++i)
        idx[i] = i / r;
    for (int i = 2 * r + 1; i < 4 * r + 1; ++i)
        idx[i] = (i - 1) / r + 1;
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
            for (auto &line : pwCell)
            {
                std::fill(line.begin(), line.end(), 0.f);
            }
            for (auto &line : wCell)
            {
                std::fill(line.begin(), line.end(), 0.f);
            }
            for (int kernel_h = std::max(h - R, 0); kernel_h <= std::min(h + R, height - 1); ++kernel_h)
            {
                int hk = kernel_h - h + R;
                const uchar *xi = input + kernel_h * width + std::max(w - R, 0);
                for (int kernel_w = std::max(w - R, 0); kernel_w <= std::min(w + R, width - 1); ++kernel_w)
                {
                    int wk = kernel_w - w + R;
                    float weight_i = weights[std::abs(*x - *xi)];
                    wCell[idx[hk]][idx[wk]] += weight_i;
                    pwCell[idx[hk]][idx[wk]] += weight_i * *xi++;
                }
            }
            for (int row = 0; row < 5; ++row)
            {
                sumColPW[row][0] = pwCell[row][0] + pwCell[row][1] + pwCell[row][2];
                sumColW[row][0] = wCell[row][0] + wCell[row][1] + wCell[row][2];
            }
            for (int row = 0; row < 5; ++row)
            {
                for (int col = 1; col < 3; ++col)
                {
                    sumColPW[row][col] = sumColPW[row][col - 1] - pwCell[row][col - 1] + pwCell[row][col + 2];
                    sumColW[row][col] = sumColW[row][col - 1] - pwCell[row][col - 1] + pwCell[row][col + 2];
                }
            }

            for (int col = 0; col < 3; ++col)
            {
                sumPW[0][col] = sumColPW[0][col] + sumColPW[1][col] + sumColPW[2][col];
                sumW[0][col] = sumColW[0][col] + sumColW[1][col] + sumColW[2][col];
            }
            for (int row = 1; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    sumPW[row][col] = sumPW[row - 1][col] - sumColPW[row - 1][col] + sumColPW[row + 2][col];
                    sumW[row][col] = sumW[row - 1][col] - sumColW[row - 1][col] + sumColW[row + 2][col];
                }
            }
            float min_diff = 255.f;
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    if (row == 1 && col == 1)
                        continue;
                    float candidate = sumW[row][col] == 0 ? *x : sumPW[row][col] / sumW[row][col];
                    float diff = abs(*x - candidate);
                    if (diff < min_diff)
                    {
                        if (candidate < 0)
                            candidate = 0;
                        if (candidate > 255)
                            candidate = 255;
                        *xx = (uchar)candidate;
                        min_diff = diff;
                    }
                }
            }

            x++;
            xx++;
        }
    }
}