#pragma once
#include <string>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include "cv_filters.hpp"
#include "cv_geom.hpp"

struct GpuPrograms {
    GLuint passProg = 0;        // 直通着色
    GLuint pixelateProg = 0;    // GPU 像素化
    GLuint sincityProg = 0;     // GPU SinCity
};

class GpuPipeline {
public:
    bool init(const std::string& shaderDir);
    void draw(GLuint vao, GLuint tex, int texW, int texH,
        FilterType filter, const FilterParams& fp,
        const AffineParams& ap);

private:
    GpuPrograms prog_;
    GLint loc_uTex_pass_ = -1;

    // Pixelate uniforms
    GLint loc_uTex_pix_ = -1, loc_uTexSize_pix_ = -1, loc_uBlock_pix_ = -1, loc_uAff_pix_ = -1;

    // SinCity uniforms
    GLint loc_uTex_sc_ = -1, loc_uKeep_sc_ = -1, loc_uThresh_sc_ = -1, loc_uAff_sc_ = -1;
};
