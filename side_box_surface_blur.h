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
    ~SideSurface(){};
    void _init_weight(uchar threshold);
    void run(const uchar *input, uchar *output);

private:
    const int height;
    const int width;
    const int radius;
    std::vector<float> idx;
    std::vector<float> weights;
    std::vector<std::vector<float>> pwCell;
    std::vector<std::vector<float>> wCell;
    std::vector<std::vector<float>> sumPW;
    std::vector<std::vector<float>> sumW;
    std::vector<std::vector<float>> sumColPW;
    std::vector<std::vector<float>> sumColW;
};