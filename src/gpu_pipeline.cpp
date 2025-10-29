#include "gpu_pipeline.hpp"
#include "gl_utils.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <opencv2/opencv.hpp>

// Convert OpenCV 3x3 affine matrix to GLM mat3 (filled in column-major order)
static glm::mat3 toGLM(const cv::Matx33f& m) {
    return glm::mat3(
        m(0, 0), m(1, 0), m(2, 0),
        m(0, 1), m(1, 1), m(2, 1),
        m(0, 2), m(1, 2), m(2, 2)
    );
}

bool GpuPipeline::init(const std::string& shaderDir) {
    try {
        // Load three shader programs
        prog_.passProg = glutils::loadShaderProgram(shaderDir + "/passthrough.vert",
            shaderDir + "/passthrough.frag");

        prog_.pixelateProg = glutils::loadShaderProgram(shaderDir + "/passthrough.vert",
            shaderDir + "/filter_pixelate.frag");

        prog_.sincityProg = glutils::loadShaderProgram(shaderDir + "/passthrough.vert",
            shaderDir + "/filter_sincity.frag");
    }
    catch (const std::exception& e) {
        fprintf(stderr, "[GpuPipeline] Shader load error: %s\n", e.what());
        return false;
    }

    glUseProgram(prog_.passProg);
    loc_uTex_pass_ = glGetUniformLocation(prog_.passProg, "uTex");

    // ---- pixelate  ----
    glUseProgram(prog_.pixelateProg);
    loc_uTex_pix_ = glGetUniformLocation(prog_.pixelateProg, "uTex");
    loc_uTexSize_pix_ = glGetUniformLocation(prog_.pixelateProg, "uTexSize");
    loc_uBlock_pix_ = glGetUniformLocation(prog_.pixelateProg, "uBlock");
    loc_uAff_pix_ = glGetUniformLocation(prog_.pixelateProg, "uAffine");

    // ---- SinCity   ----
    glUseProgram(prog_.sincityProg);
    loc_uTex_sc_ = glGetUniformLocation(prog_.sincityProg, "uTex");
    loc_uKeep_sc_ = glGetUniformLocation(prog_.sincityProg, "uKeepColor");
    loc_uThresh_sc_ = glGetUniformLocation(prog_.sincityProg, "uThresh");
    loc_uAff_sc_ = glGetUniformLocation(prog_.sincityProg, "uAffine");

    glUseProgram(0);
    return true;
}

void GpuPipeline::draw(GLuint vao, GLuint tex, int texW, int texH,
    FilterType filter, const FilterParams& fp,
    const AffineParams& ap)
{
    // Bind input texture to texture unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    // Build 3x3 affine matrix in pixel coordinates (consistent with CPU version)
    const cv::Matx33f M = affineMatrix(ap, texW, texH);
    const glm::mat3   gM = toGLM(M);

    // Choose program
    GLuint prog = prog_.passProg;
    switch (filter) {
    case FilterType::None:     prog = prog_.passProg;      break;
    case FilterType::Pixelate: prog = prog_.pixelateProg;  break;
    case FilterType::SinCity:  prog = prog_.sincityProg;   break;
    }

    glUseProgram(prog);

    // Set uniforms for the selected path
    if (prog == prog_.passProg) {

        if (loc_uTex_pass_ >= 0) glUniform1i(loc_uTex_pass_, 0);
        GLint loc_uAff_pass = glGetUniformLocation(prog_.passProg, "uAffine");
        if (loc_uAff_pass >= 0)  glUniformMatrix3fv(loc_uAff_pass, 1, GL_FALSE, glm::value_ptr(gM));
    }
    else if (prog == prog_.pixelateProg) {
        if (loc_uTex_pix_ >= 0) glUniform1i(loc_uTex_pix_, 0);
        if (loc_uTexSize_pix_ >= 0) glUniform2f(loc_uTexSize_pix_, (float)texW, (float)texH);
        if (loc_uBlock_pix_ >= 0) glUniform1f(loc_uBlock_pix_, (float)std::max(1, fp.pixelBlock));
        if (loc_uAff_pix_ >= 0) glUniformMatrix3fv(loc_uAff_pix_, 1, GL_FALSE, glm::value_ptr(gM));
    }
    else if (prog == prog_.sincityProg) {
        if (loc_uTex_sc_ >= 0) glUniform1i(loc_uTex_sc_, 0);

        // keep color: BGR(0..255) -> RGB(0..1)
        const glm::vec3 keepRGB(
            fp.keepBGR[2] / 255.f,
            fp.keepBGR[1] / 255.f,
            fp.keepBGR[0] / 255.f
        );
        if (loc_uKeep_sc_ >= 0) glUniform3fv(loc_uKeep_sc_, 1, &keepRGB[0]);
        if (loc_uThresh_sc_ >= 0) glUniform1f(loc_uThresh_sc_, (float)fp.thresh / 255.f);
        if (loc_uAff_sc_ >= 0) glUniformMatrix3fv(loc_uAff_sc_, 1, GL_FALSE, glm::value_ptr(gM));
    }

    // Draw
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    // Cleanup bindings
    glUseProgram(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}
