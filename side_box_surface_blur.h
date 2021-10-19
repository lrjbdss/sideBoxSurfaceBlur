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
    void _init_weight(uchar threshold);          // 初始化权重查找表
    void run(const uchar *input, uchar *output); // 处理主函数
    void _calPix(const uchar *x, uchar *xx);     // 根据sumColPW和sumColW计算目标像素值

private:
    const int height;  // 图片的高
    const int width;   // 图片的宽
    const int radius;  // 表面滤波半径
    int *cellIdx;      // 从(4r+1)长度序列到5长度序列的索引对应关系
    int *weight_table; // 按像素差的绝对值排序的像素权重列表
    int *pwCell;       // 5*5数组中每个元素的像素与权重的乘积
    int *wCell;        // 5*5数组中每个元素的权重
    int *sumColPW;     // 水平方向像素与权重的乘积相加，数组大小3*5
    int *sumColW;      // 水平方向权重相加，数组大小3*5
};