
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// Dimensiunea imaginii de intrare
// Output-ul este 640 x 640

__device__ float get_pixel(
    const unsigned char* input,
    int x,
    int y,
    int channel,
    int width
) {
    return static_cast<float>(
        input[(y * width + x) * 3 + channel]
    );
}


// Kernel CUDA pentru preprocessing YOLO26n
__global__ void preprocess_kernel(
    const unsigned char* input,
    float* output,
    int src_width,
    int src_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    int output_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_pixels = output_size * output_size;

    if (idx >= total_pixels) {
        return;
    }

    int x = idx % output_size;
    int y = idx / output_size;

    // Indicii pentru cele 3 canale
    int plane_size = total_pixels;

    // Valoarea de padding folosita de Ultralytics
    const float padding_value = 114.0f / 255.0f;

    float r = padding_value;
    float g = padding_value;
    float b = padding_value;

    // Verificam daca pixelul se afla in imaginea redimensionata
    int resize_x = x - pad_left;
    int resize_y = y - pad_top;

    if (
        resize_x >= 0 &&
        resize_x < resized_width &&
        resize_y >= 0 &&
        resize_y < resized_height
    ) {
        // Mapare coordonate output -> imagine originala
        float scale_x = (float)src_width / resized_width;
        float scale_y = (float)src_height / resized_height;

        float src_x = (resize_x + 0.5f) * scale_x - 0.5f;
        float src_y = (resize_y + 0.5f) * scale_y - 0.5f;

        src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
        src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));

        int x0 = (int)floorf(src_x);
        int y0 = (int)floorf(src_y);

        int x1 = min(x0 + 1, src_width - 1);
        int y1 = min(y0 + 1, src_height - 1);

        float dx = src_x - x0;
        float dy = src_y - y0;

        // Interpolare biliniara pentru fiecare canal BGR
        for (int c = 0; c < 3; c++) {

            float p00 = get_pixel(input, x0, y0, c, src_width);
            float p01 = get_pixel(input, x1, y0, c, src_width);
            float p10 = get_pixel(input, x0, y1, c, src_width);
            float p11 = get_pixel(input, x1, y1, c, src_width);

            float top = p00 * (1.0f - dx) + p01 * dx;
            float bottom = p10 * (1.0f - dx) + p11 * dx;

            float value = top * (1.0f - dy) + bottom * dy;

            value /= 255.0f;

            // BGR -> RGB si HWC -> CHW
            if (c == 2) {
                r = value;
            }
            else if (c == 1) {
                g = value;
            }
            else {
                b = value;
            }
        }
    }

    // Scriem rezultatul in format CHW
    output[idx] = r;
    output[plane_size + idx] = g;
    output[2 * plane_size + idx] = b;
}


// Functie host pentru lansarea kernelului
void launch_preprocess(
    const unsigned char* input,
    float* output,
    int src_width,
    int src_height,
    int output_size,
    cudaStream_t stream
) {
    // Calculam factorul de redimensionare
    float scale = fminf(
        (float)output_size / src_width,
        (float)output_size / src_height
    );

    int resized_width = (int)roundf(src_width * scale);
    int resized_height = (int)roundf(src_height * scale);

    // Calculam padding-ul
    int pad_left = (output_size - resized_width) / 2;
    int pad_top = (output_size - resized_height) / 2;

    int total_pixels = output_size * output_size;

    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;

    preprocess_kernel<<<blocks, threads, 0, stream>>>(
        input,
        output,
        src_width,
        src_height,
        resized_width,
        resized_height,
        pad_left,
        pad_top,
        output_size
    );
}
W