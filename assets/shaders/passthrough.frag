#version 330 core
in vec2 vUV;                         
out vec4 FragColor;

uniform sampler2D uTex;              
uniform mat3 uAffine;                

void main()
{
    vec3 p  = uAffine * vec3(vUV, 1.0);
    vec2 uv = p.xy;

    uv.y = 1.0 - uv.y;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    FragColor = texture(uTex, uv);
}
