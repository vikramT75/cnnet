#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <raylib.h>

#include "./thirdparty/stb_image.h"       // STB_IMAGE_IMPLEMENTATION
#include "./thirdparty/stb_image_write.h" // AND STB_IMAGE_WRIE_IMPLEMENTATION already in raylib lib.

#define NN_IMPLEMENTATION
#include "nn.h"

typedef struct
{
    size_t *items;
    size_t count;
    size_t capacity;
} Arch;

typedef struct
{
    float *items;
    size_t count;
    size_t capacity;
} Cost_Plot;

#define DA_INIT_CAP 256
#define da_append(da, item)                                                            \
    do                                                                                 \
    {                                                                                  \
        if ((da)->count >= (da)->capacity)                                             \
        {                                                                              \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity * 2;   \
            (da)->items = realloc((da)->items, (da)->capacity * sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Not enough memory");                        \
        }                                                                              \
        (da)->items[(da)->count++] = (item);                                           \
    } while (0)

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh)
{
    Color low_colour = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_colour = {0x00, 0xFF, 0x00, 0x00};
    Color background_colour = {0x18, 0x18, 0x18, 0xFF};

    // ClearBackground(background_colour);

    float neuron_radius = rh * 0.04;
    int layer_border_vpad = 50;
    int layer_border_hpad = 50;
    int nn_width = rw - 2 * layer_border_hpad;
    int nn_height = rh - 2 * layer_border_vpad;
    int nn_x = rx + rw / 2 - nn_width / 2;
    int nn_y = ry + rh / 2 - nn_height / 2;
    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / (arch_count);
    for (size_t l = 0; l < arch_count; l++)
    {

        int layer_vpad1 = nn_height / nn.as[l].cols;

        for (size_t i = 0; i < nn.as[l].cols; i++)
        {
            int cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
            int cy1 = nn_y + i * layer_vpad1 + layer_vpad1 / 2;

            if (l + 1 < arch_count)
            {
                int layer_vpad2 = nn_height / nn.as[l + 1].cols;
                for (size_t j = 0; j < nn.as[l + 1].cols; j++)
                {
                    int cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
                    int cy2 = nn_y + j * layer_vpad2 + layer_vpad2 / 2;

                    // uint32_t alpha = floorf(255.f * sigmoidf(MAT_AT(nn.ws[l], i, j)));
                    float value = sigmoidf(MAT_AT(nn.ws[l], i, j));
                    high_colour.a = floorf(255.f * value);
                    float thickness = rh * 0.004f;
                    Vector2 start = {cx1, cy1};
                    Vector2 end = {cx2, cy2};
                    // DrawLine(cx1, cy1, cx2, cy2, ColorAlphaBlend(low_colour, high_colour, RAYWHITE));
                    DrawLineEx(start, end, thickness, ColorAlphaBlend(low_colour, high_colour, RAYWHITE));
                }
            }

            if (l > 0)
            {
                high_colour.a = floorf(255.f * sigmoidf(MAT_AT(nn.bs[l - 1], 0, i)));
                DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_colour, high_colour, RAYWHITE));
            }
            else
                DrawCircle(cx1, cy1, neuron_radius, GRAY);
        }
    }
}

void cost_plot_minmax(Cost_Plot plot, float *min, float *max)
{
    *max = FLT_MIN;
    *min = FLT_MAX;

    for (size_t i = 0; i < plot.count; i++)
    {
        if (*max < plot.items[i])
            *max = plot.items[i];
        if (*min > plot.items[i])
            *min = plot.items[i];
    }
    return;
}

void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh)
{
    float max, min;
    cost_plot_minmax(plot, &min, &max);
    if (min > 0)
        min = 0;
    float range = max - min;
    size_t n = plot.count;
    if (n < 1000)
        n = 1000;
    // DrawRectangle(rx, ry, rw, rh, RAYWHITE);
    for (size_t i = 0; i + 1 < plot.count; i++)
    {
        float x1 = rx + (float)rw * i / n;
        float y1 = ry + rh * (1 - (plot.items[i] - min) / range);
        float x2 = rx + (float)rw * (i + 1) / n;
        float y2 = ry + rh * (1 - (plot.items[i + 1] - min) / range);
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh * 0.005, RED);
        // DrawCircle(x1, y1, rh * 0.005, RED);
    }
}

int main(int argc, char **argv)
{

    size_t IMG_FACTOR = 80;
    size_t IMG_WIDTH = (16 * IMG_FACTOR);
    size_t IMG_HEIGHT = (9 * IMG_FACTOR);

    const char *program = args_shift(&argc, &argv);

    if (argc <= 0)
    {
        fprintf(stderr, "Usage : %s <input.png>\n", program);
        fprintf(stderr, "ERROR: input file not provided\n");
        return 1;
    }

    const char *img_file_path = args_shift(&argc, &argv);

    int img_width, img_height, img_comp;
    uint8_t *img_pixels = (uint8_t *)stbi_load(img_file_path, &img_width, &img_height, &img_comp, 0);

    if (img_pixels == NULL)
    {
        fprintf(stderr, "ERROR : Could not read image %s\n", img_file_path);
        return 1;
    }

    if (img_comp != 1)
    {
        fprintf(stderr, "ERROR : %s is %d bits long. Only 8 bit greyscale images supported\n", img_file_path, img_comp * 8);
    }

    printf("%ss size %dx%d %d bits\n", img_file_path, img_width, img_height, img_comp * 8);

    Mat t = mat_alloc(img_width * img_height, 3);

    for (size_t y = 0; y < (size_t)img_width; y++)
    {
        for (size_t x = 0; x < (size_t)img_width; x++)
        {

            size_t i = y * img_width + x;
            MAT_AT(t, i, 0) = (float)x / (img_width - 1);
            MAT_AT(t, i, 1) = (float)y / (img_height - 1);
            MAT_AT(t, i, 2) = (float)img_pixels[i] / 255.f;
        }
    }

    Mat ti = {
        .rows = t.rows,
        .cols = 2,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, 0)};

    Mat to = {
        .rows = t.rows,
        .cols = 1,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, ti.cols)};

    // MAT_PRINT(ti);
    // MAT_PRINT(to);

    size_t arch[] = {2, 7, 4, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "Train");
    SetTargetFPS(120);

    Cost_Plot plot = {0};

    Image preview_image = GenImageColor(img_width, img_height, BLACK);
    Image original_image = GenImageColor(img_width, img_height, BLACK);

    for (size_t y = 0; y < (size_t)img_height; y++)
    {
        for (size_t x = 0; x < (size_t)img_width; x++)
        {
            uint8_t pixel = img_pixels[y * img_width + x];
            ImageDrawPixel(&original_image, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
        }
    }
    Texture2D preview_texture = LoadTextureFromImage(preview_image);
    Texture2D original_texture = LoadTextureFromImage(original_image);

    size_t epoch = 0;
    size_t max_epoch = 100 * 1000;
    size_t epochs_per_frame = 100;
    size_t batches_per_frame = 200;
    size_t batch_size = 28;
    size_t batch_begin = 0;
    size_t batch_count = (t.rows + batch_size - 1) / batch_size;
    float avg_cost = 0.f;
    float rate = 0.5f;
    bool paused = true;

    mat_shuffle_rows(t);

    while (!WindowShouldClose())
    {

        if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_P))
            paused = !paused;

        if (IsKeyPressed(KEY_R))
        {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        if (IsKeyPressed(KEY_S))
        {
            size_t out_width = 512;
            size_t out_height = 512;
            uint8_t *out_pixels = malloc(sizeof(*out_pixels) * out_width * out_height);
            assert(out_pixels != NULL);

            for (size_t y = 0; y < out_height; y++)
            {
                for (size_t x = 0; x < out_width; x++)
                {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (out_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (out_height - 1);
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
                    out_pixels[y * out_width + x] = pixel;
                }
            }

            char outbuffer[256];
            snprintf(outbuffer, sizeof(outbuffer), "./mnist/output/upscaled%c.png", img_file_path[strlen(img_file_path) - 5]);

            if (!stbi_write_png(outbuffer, out_width, out_height, 1, out_pixels, out_width * sizeof(*out_pixels)))
                fprintf(stderr, "ERROR : Could not save image as %s\n", outbuffer);
            else
                printf("Saved image %s\n", outbuffer);

            free(out_pixels);
        }

        for (size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; i++)
        {
            size_t size = batch_size;

            if (batch_begin + batch_size >= t.rows)
                size = t.rows - batch_begin;

            Mat batch_ti = {
                .rows = size,
                .cols = 2,
                .stride = t.stride,
                .es = &MAT_AT(t, batch_begin, 0),
            };

            Mat batch_to = {
                .rows = size,
                .cols = 1,
                .stride = t.stride,
                .es = &MAT_AT(t, batch_begin, batch_ti.cols),
            };

            nn_backprop(nn, g, batch_ti, batch_to);
            nn_learn(nn, g, rate);
            avg_cost += nn_cost(nn, batch_ti, batch_to);
            batch_begin += batch_size;

            if (batch_begin >= t.rows)
            {
                da_append(&plot, avg_cost / batch_count);
                avg_cost = 0.0f;
                epoch++;
                batch_begin = 0;
                mat_shuffle_rows(t);
            }
        }

        BeginDrawing();
        ClearBackground((Color){0x18, 0x18, 0x18, 0xFF});
        int w = GetRenderWidth();
        int h = GetRenderHeight();

        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Epoch : %zu/%zu\t\tRate : %f\t\tCost : %f", epoch, max_epoch, rate, (plot.count > 0) ? plot.items[plot.count - 1] : 0);
        DrawText(buffer, 0, 0, h * 0.04, WHITE);
        int rw, rh, rx, ry;

        rw = w / 3;
        rh = h * 2 / 3;
        rx = 0;
        ry = h / 2 - rh / 2;
        plot_cost(plot, rx, ry, rw, rh);

        int zero_y = ry + rh; // Or simply (ry + rh)

        DrawLine(rx, zero_y, rx + rw, zero_y, RAYWHITE);

        int font_size = h * 0.04;
        int text_padding_x = 10;
        int text_padding_y = 5;
        DrawText("0", rx + text_padding_x, zero_y - font_size - text_padding_y, font_size, RAYWHITE);

        rx += rw;
        nn_render_raylib(nn, rx, ry, rw, rh);
        rx += rw;

        float scale = 11.f;

        for (size_t y = 0; y < img_width; y++)
        {
            for (size_t x = 0; x < img_height; x++)
            {
                MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (img_width - 1);
                MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (img_height - 1);
                nn_forward(nn);
                uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
                ImageDrawPixel(&preview_image, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
            }
        }

        UpdateTexture(preview_texture, preview_image.data);
        DrawTextureEx(preview_texture, CLITERAL(Vector2){rx, ry - (img_height * scale) * 0.3}, 0, scale, WHITE);
        DrawTextureEx(original_texture, CLITERAL(Vector2){rx, ry + (img_height * scale) * 0.85}, 0, scale, WHITE);

        EndDrawing();
    }

    for (size_t y = 0; y < (size_t)img_height; y++)
    {
        for (size_t x = 0; x < (size_t)img_width; x++)
        {
            uint8_t pixel = img_pixels[y * img_width + x];
            if (pixel)
                printf("%3u ", pixel);
            else
                printf("    ");
        }
        printf("\n");
    }

    printf("\n\n");

    for (size_t y = 0; y < (size_t)img_height; y++)
    {
        for (size_t x = 0; x < (size_t)img_width; x++)
        {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (img_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (img_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
            if (pixel)
                printf("%3u ", pixel);
            else
                printf("    ");
        }
        printf("\n");
    }

    printf("\n\n");

    size_t out_width = 512;
    size_t out_height = 512;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels) * out_width * out_height);
    assert(out_pixels != NULL);

    for (size_t y = 0; y < out_height; y++)
    {
        for (size_t x = 0; x < out_width; x++)
        {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (out_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (out_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
            out_pixels[y * out_width + x] = pixel;
        }
    }

    char outbuffer[256];
    snprintf(outbuffer, sizeof(outbuffer), "./mnist/output/upscaled%c.png", img_file_path[strlen(img_file_path) - 5]);
    // snprintf(outbuffer, sizeof(outbuffer), "./mnist/output/upscaled.png");

    const char *out_file_path = outbuffer;

    if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width * sizeof(*out_pixels)))
        fprintf(stderr, "ERROR : Could not save image as %s\n", out_file_path);

    printf("Generated image %s from input %s\n", out_file_path, img_file_path);

    return 0;
}