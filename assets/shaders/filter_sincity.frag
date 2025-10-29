#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uTex;
uniform vec3  uKeepColor; // RGB(0..1)
uniform float uThresh;    // 0..1
uniform mat3  uAffine;

vec2 uv_flip(vec2 uv){ return vec2(uv.x, 1.0 - uv.y); }

vec2 apply_affine(vec2 uv, vec2 size){
    vec2 px = vec2(uv.x * size.x, uv.y * size.y);
    vec3 pxa = uAffine * vec3(px, 1.0);
    return vec2(pxa.x / size.x, pxa.y / size.y);
}

float lum(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }

void main(){
    vec2 size = vec2(textureSize(uTex, 0));
    vec2 uv = uv_flip(vUV);
    uv = apply_affine(uv, size);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 c = texture(uTex, uv).rgb;
    float d = distance(c, uKeepColor);
    vec3 bw = vec3(lum(c));
    vec3 outc = (d <= uThresh) ? c : bw;
    FragColor = vec4(outc, 1.0);
}
