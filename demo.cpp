#include "side_box_surface_blur.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

int main()
{
    char *img_path = "../0.jpg";
    cv::Mat img = cv::imread(img_path);
    SideSurface sideSurface(img.rows, img.cols, 10, 20);
    std::vector<cv::Mat> inp_chns(img.channels());
    std::vector<cv::Mat> out_chns;
    cv::split(img, inp_chns);
    for (int i = 0; i < img.channels(); ++i)
    {
        out_chns.emplace_back(img.rows, img.cols, inp_chns[i].type());
        sideSurface.run(inp_chns[i].data, out_chns[i].data);
    }
    cv::Mat out_img;
    cv::merge(out_chns, out_img);
    cv::imwrite("0_r10.jpg", out_img);
}