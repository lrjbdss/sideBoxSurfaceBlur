#pragma once
#include <vector>
typedef unsigned char uchar;

class SideSurface
{
public:
    SideSurface(int height,
                int width,
                int radius,
                uchar threshold);
    ~SideSurface();
    void _init_weight(uchar threshold);
    void run(const uchar *input, uchar *output);

private:
    const int height;
    const int width;
    const int radius;
    int *cellIdx;
    int *weight_table;
    int *pwCell;
    int *wCell;
    int *sumColPW;
    int *sumColW;
};