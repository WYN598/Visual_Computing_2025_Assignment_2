#include "cv_filters.hpp"

static void pixelateCPU(cv::Mat& img, int block) {
    if (img.empty() || block <= 1) return;

    // Guard: block should not be too large, and the resized target must be at least 1x1
    int b = std::max(2, block);
    int sx = std::max(1, img.cols / b);
    int sy = std::max(1, img.rows / b);

    cv::Mat small;
    cv::resize(img, small, cv::Size(sx, sy), 0, 0, cv::INTER_LINEAR);
    cv::resize(small, img, img.size(), 0, 0, cv::INTER_NEAREST);
}

static void sinCityCPU(cv::Mat& img, cv::Vec3b keepBGR, int thresh) {
    CV_Assert(img.channels() == 3);
    cv::Mat gray; cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat bw; cv::cvtColor(gray, bw, cv::COLOR_GRAY2BGR);

    cv::Mat diff;
    cv::Mat keep(img.size(), CV_8UC1, cv::Scalar(0));

    // Compute color difference from keepBGR
    for (int y = 0; y < img.rows; ++y) {
        const cv::Vec3b* src = img.ptr<cv::Vec3b>(y);
        uchar* kdst = keep.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x) {
            int db = int(src[x][0]) - int(keepBGR[0]);
            int dg = int(src[x][1]) - int(keepBGR[1]);
            int dr = int(src[x][2]) - int(keepBGR[2]);
            int d = int(std::sqrt(double(db * db + dg * dg + dr * dr)));
            kdst[x] = (d <= thresh) ? 255 : 0;
        }
    }
    // Keep color where pixels are close to the target color; otherwise use grayscale
    img.copyTo(bw, keep);
    img = bw;
}

void applyCpuFilter(cv::Mat& img, FilterType type, const FilterParams& params) {
    switch (type) {
    case FilterType::None:     break;
    case FilterType::Pixelate: pixelateCPU(img, params.pixelBlock); break;
    case FilterType::SinCity:  sinCityCPU(img, params.keepBGR, params.thresh); break;
    }
}

std::string filterName(FilterType t) {
    switch (t) {
    case FilterType::None: return "None";
    case FilterType::Pixelate: return "Pixelate";
    case FilterType::SinCity: return "SinCity";
    }
    return "Unknown";
}
