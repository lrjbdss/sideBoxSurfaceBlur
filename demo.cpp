#include "side_box_surface_blur.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

int main()
{
    char *img_path = "../0.jpg";
    cv::Mat img = cv::imread(img_path);
    SideSurface sideSurface(img.rows, img.cols, 10, 15);
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
    cv::imwrite("0_r3.jpg", out_img);
    // cv::imshow("res", out_img);
    // cv::waitKey();

    // char *img_path = "../0.jpg";
    // cv::Mat img_bgr = cv::imread(img_path);
    // cv::Mat img;
    // cv::cvtColor(img_bgr, img, cv::COLOR_BGR2YCrCb);
    // SideSurface sideSurface(img.rows, img.cols, 1, 10);
    // std::vector<cv::Mat> inp_chns(img.channels());
    // std::vector<cv::Mat> out_chns;
    // cv::split(img, inp_chns);
    // out_chns.emplace_back(img.rows, img.cols, inp_chns[0].type());
    // sideSurface.run(inp_chns[0].data, out_chns[0].data);
    // out_chns.push_back(inp_chns[1]);
    // out_chns.push_back(inp_chns[2]);
    // cv::Mat out_img_yuv, out_bgr;
    // cv::merge(out_chns, out_img_yuv);
    // cv::cvtColor(out_img_yuv, out_bgr, cv::COLOR_YCrCb2BGR);
    // cv::imwrite("0_r3.jpg", out_bgr);
}