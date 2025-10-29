#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uTex;
uniform vec2  uTexSize; 
uniform float uBlock;   
uniform mat3  uAffine;

vec2 uv_flip(vec2 uv){ return vec2(uv.x, 1.0 - uv.y); }

vec2 uv_to_px(vec2 uv){ return vec2(uv.x * uTexSize.x, uv.y * uTexSize.y); }
vec2 px_to_uv(vec2 px){ return vec2(px.x / uTexSize.x, px.y / uTexSize.y); }

void main(){
    vec2 uv = uv_flip(vUV);

    vec2 px = uv_to_px(uv);
    vec3 pxa = uAffine * vec3(px, 1.0);
    vec2 uvT = px_to_uv(pxa.xy);

    if (uvT.x < 0.0 || uvT.x > 1.0 || uvT.y < 0.0 || uvT.y > 1.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec2 block = vec2(max(uBlock, 1.0));
    vec2 grid_px = floor(uv_to_px(uvT) / block) * block + block * 0.5;
    vec2 uvq = px_to_uv(grid_px);

    FragColor = texture(uTex, uvq);
}
