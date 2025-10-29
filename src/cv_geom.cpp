#include "cv_geom.hpp"
#include <cmath>

static cv::Matx23f makeAffine23(const AffineParams& p, int w, int h) {
    // Use the image center as the rotation and scaling origin
    float cx = w * 0.5f;
    float cy = h * 0.5f;
    // Rotation and scaling
    cv::Matx23f R = cv::getRotationMatrix2D(cv::Point2f(cx, cy), p.thetaDeg, p.scale);
    // Translation
    R(0, 2) += p.tx;
    R(1, 2) += p.ty;
    return R;
}

void warpCpuAffine(cv::Mat& img, const AffineParams& p) {
    if (p.scale == 1.f && std::abs(p.thetaDeg) < 1e-4 && std::abs(p.tx) < 1e-4 && std::abs(p.ty) < 1e-4)
        return;
    cv::Mat out;
    cv::Matx23f A = makeAffine23(p, img.cols, img.rows);
    cv::warpAffine(img, out, A, img.size(),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    img = out;
}

cv::Matx33f affineMatrix(const AffineParams& p, int w, int h) {
    float cx = w * 0.5f;
    float cy = h * 0.5f;
    float rad = p.thetaDeg * 3.1415926535f / 180.f;
    float c = std::cos(rad) * p.scale;
    float s = std::sin(rad) * p.scale;

    cv::Matx33f M = {
        c, -s,  cx - c * cx + s * cy + p.tx,
        s,  c,  cy - s * cx - c * cy + p.ty,
        0,  0,  1
    };
    return M;
}
