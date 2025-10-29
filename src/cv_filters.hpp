#pragma once
#include <opencv2/opencv.hpp>
#include <string>

enum class FilterType {
    None = 0,
    Pixelate,
    SinCity
};

struct FilterParams {
    int   pixelBlock = 8;         // Pixelate block size
    cv::Vec3b keepBGR = { 20, 20, 200 }; // SinCity: 保留的近似颜色（B,G,R）
    int   thresh = 60;         // 颜色阈值 (0..255)
};

void applyCpuFilter(cv::Mat& img, FilterType type, const FilterParams& params);
std::string filterName(FilterType t);
