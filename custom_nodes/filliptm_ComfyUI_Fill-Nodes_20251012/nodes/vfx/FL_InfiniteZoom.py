import torch
import numpy as np
from PIL import Image
import sys
import OpenGL.GL as gl
import glfw
import ctypes
from comfy.utils import ProgressBar


VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform vec2 iResolution;

void main()
{
    vec2 scale = vec2(1.0, iResolution.y / iResolution.x);
    gl_Position = vec4(aPos.xy * scale, aPos.z, 1.0);
    TexCoord = aTexCoord;
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D iChannel0;
uniform vec3 iResolution;
uniform float iTime;
uniform float iScale;
uniform float iSwirl;
uniform float iSwirlStrength;
uniform float iIterations;
uniform float iTimeSpeed;
void main()
{
    vec4 O = vec4(0.0);
    vec2 U = TexCoord;
    float s = 0.0, s2 = 0.0, t = iTime * iTimeSpeed;
    U = U - 0.5;
    U.x += 0.03 * sin(1.14 * t);
    float sc = pow(iScale, -mod(t, 2.0) - 0.8);
    U *= sc;
    for (int i = 0; i < int(iIterations); i++) {
        vec2 V = abs(U + U);
        if (max(V.x, V.y) > 1.0) break;
        V = smoothstep(1.0, 0.5, V);
        float m = V.x * V.y;
        O = mix(O, texture(iChannel0, U + 0.5), m);
        s = mix(s, 1.0, m);
        s2 = s2 * (1.0 - m) * (1.0 - m) + m * m;
        U *= iScale;
        if (iSwirl > 0.5) {
            U.x = -U.x * iSwirlStrength;
        }
    }
    vec4 mean = texture(iChannel0, U, 10.0);
    O = mean + (O - s * mean) / sqrt(s2);
    FragColor = O;
}
"""

class FL_InfiniteZoom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 2.00, "min": 1.10, "max": 10.00, "step": 0.05}),
                "mirror": (["on", "off"],),
                "mirror_warp": ("FLOAT", {"default": 1.00, "min": 0.50, "max": 1.50, "step": 0.05}),
                "iterations": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_shader"
    CATEGORY = "🏵️Fill Nodes/VFX"

    def apply_shader(self, images, scale, mirror, mirror_warp, iterations, speed, fps):
        result = []
        total_images = len(images)
        pbar = ProgressBar(total_images)
        frame_time = 1.0 / fps
        for i, image in enumerate(images, start=1):
            img = self.t2p(image)
            result_img = self.process_image(img, scale, mirror, mirror_warp, iterations, speed, i * frame_time)
            result_img = self.p2t(result_img)
            result.append(result_img)
            pbar.update_absolute(i)

        return (torch.cat(result, dim=0),)

    def process_image(self, image, scale, mirror, mirror_warp, iterations, speed, time):
        img_array = np.array(image).astype(np.float32) / 255.0
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(image.width, image.height, "Hidden Window", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)

        # Set the viewport
        gl.glViewport(0, 0, image.width, image.height)

        vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vertex_shader, VERTEX_SHADER)
        gl.glCompileShader(vertex_shader)
        fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragment_shader, FRAGMENT_SHADER)
        gl.glCompileShader(fragment_shader)
        shader_program = gl.glCreateProgram()
        gl.glAttachShader(shader_program, vertex_shader)
        gl.glAttachShader(shader_program, fragment_shader)
        gl.glLinkProgram(shader_program)
        gl.glUseProgram(shader_program)

        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0, 0.0,
            1.0, -1.0, 0.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 1.0, 1.0
        ], dtype=np.float32)

        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 5 * vertices.itemsize, None)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)

        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, image.width, image.height, 0, gl.GL_RGB, gl.GL_FLOAT, img_array)

        gl.glUniform1i(gl.glGetUniformLocation(shader_program, "iChannel0"), 0)
        gl.glUniform2f(gl.glGetUniformLocation(shader_program, "iResolution"), image.width, image.height)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iTime"), time)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iScale"), scale)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iSwirl"), 1.0 if mirror == "on" else 0.0)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iSwirlStrength"), mirror_warp)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iIterations"), iterations)
        gl.glUniform1f(gl.glGetUniformLocation(shader_program, "iTimeSpeed"), speed)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

        img_data = gl.glReadPixels(0, 0, image.width, image.height, gl.GL_RGB, gl.GL_FLOAT)
        img_array = np.frombuffer(img_data, dtype=np.float32).reshape((image.height, image.width, 3))

        gl.glDeleteTextures(1, (texture,))
        gl.glDeleteBuffers(1, [vbo])
        gl.glDeleteVertexArrays(1, [vao])
        gl.glDeleteProgram(shader_program)
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        glfw.destroy_window(window)
        glfw.terminate()

        processed_image = Image.fromarray((img_array * 255).astype(np.uint8))
        return processed_image

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0)
        return t
