#pragma once
#include <opencv2/opencv.hpp>

struct AffineParams {
    float tx = 0.f;  // 平移 x (像素)
    float ty = 0.f;  // 平移 y (像素)
    float scale = 1.f;
    float thetaDeg = 0.f; // 旋转角（度）
};

void warpCpuAffine(cv::Mat& img, const AffineParams& p);

// 生成 3x3 仿射矩阵（用于 GPU uniform）
cv::Matx33f affineMatrix(const AffineParams& p, int width, int height);
