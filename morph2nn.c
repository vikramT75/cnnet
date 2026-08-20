#include <stdio.h>
#include <assert.h>
#include <float.h>

#include "./thirdparty/stb_image.h"       // STB_IMAGE_IMPLEMENTATION
#include "./thirdparty/stb_image_write.h" // AND STB_IMAGE_WRIE_IMPLEMENTATION already in raylib lib.

#define NN_ACT ACT_RELU
// #define NN_BACKPROP_TRADITIONAL
#define NN_IMPLEMENTATION
#include "nn.h"

#define NN_UI_IMPLEMENTATION
#include "nn_ui.h"

size_t arch[] = {3, 10, 10, 4, 1};
size_t max_epoch = 100 * 1000;
size_t batches_per_frame = 200;
size_t batch_size = 28;
float rate = 0.1f;
bool paused = true;

typedef struct
{
    size_t *items;
    size_t count;
    size_t capacity;
} Arch;

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

int main(int argc, char **argv)
{

    Region temp = region_alloc_allocator(256 * 1024 * 1024);

    size_t IMG_FACTOR = 80;
    size_t IMG_WIDTH = (16 * IMG_FACTOR);
    size_t IMG_HEIGHT = (9 * IMG_FACTOR);

    const char *program = args_shift(&argc, &argv);

    if (argc <= 0)
    {
        fprintf(stderr, "Usage : %s <image1> <image2>\n", program);
        fprintf(stderr, "ERROR: image1 file not provided\n");
        return 1;
    }

    const char *img1_file_path = args_shift(&argc, &argv);

    if (argc <= 0)
    {
        fprintf(stderr, "Usage : %s <image1> <image2>\n", program);
        fprintf(stderr, "ERROR: image2 file not provided\n");
        return 1;
    }

    const char *img2_file_path = args_shift(&argc, &argv);

    // img1
    int img1_width, img1_height, img1_comp;
    uint8_t *img1_pixels = (uint8_t *)stbi_load(img1_file_path, &img1_width, &img1_height, &img1_comp, 0);

    if (img1_pixels == NULL)
    {
        fprintf(stderr, "ERROR : Could not read image %s\n", img1_file_path);
        return 1;
    }

    if (img1_comp != 1)
    {
        fprintf(stderr, "ERROR : %s is %d bits long. Only 8 bit greyscale images supported\n", img1_file_path, img1_comp * 8);
        fprintf(stderr, "Reason: %s\n", stbi_failure_reason());
        return 1;
    }

    // img2
    int img2_width, img2_height, img2_comp;
    uint8_t *img2_pixels = (uint8_t *)stbi_load(img2_file_path, &img2_width, &img2_height, &img2_comp, 0);

    if (img2_pixels == NULL)
    {
        fprintf(stderr, "ERROR : Could not read image %s\n", img2_file_path);
        fprintf(stderr, "Reason: %s\n", stbi_failure_reason());
        return 1;
    }

    if (img2_comp != 1)
    {
        fprintf(stderr, "ERROR : %s is %d bits long. Only 8 bit greyscale images supported\n", img2_file_path, img2_comp * 8);
        return 1;
    }

    printf("%s size %dx%d %d bits\n", img1_file_path, img1_width, img1_height, img1_comp * 8);
    printf("%s size %dx%d %d bits\n", img2_file_path, img2_width, img2_height, img2_comp * 8);

    NN nn = nn_alloc(NULL, arch, ARRAY_LEN(arch));

    Mat t = mat_alloc(NULL, img1_width * img1_height + img2_width * img2_height, NN_INPUT(nn).cols + NN_OUTPUT(nn).cols);

    // handle img1
    for (size_t y = 0; y < (size_t)img1_width; y++)
    {
        for (size_t x = 0; x < (size_t)img1_width; x++)
        {
            size_t i = y * img1_width + x;
            MAT_AT(t, i, 0) = (float)x / (img1_width - 1);
            MAT_AT(t, i, 1) = (float)y / (img1_height - 1);
            MAT_AT(t, i, 2) = 0.0f;
            MAT_AT(t, i, 3) = (float)img1_pixels[i] / 255.f;
        }
    }

    // handle img2

    for (size_t y = 0; y < (size_t)img1_width; y++)
    {
        for (size_t x = 0; x < (size_t)img1_width; x++)
        {

            size_t i = img1_width * img1_height + y * img1_width + x;
            MAT_AT(t, i, 0) = (float)x / (img1_width - 1);
            MAT_AT(t, i, 1) = (float)y / (img1_height - 1);
            MAT_AT(t, i, 2) = 1.0f;
            MAT_AT(t, i, 3) = (float)img2_pixels[y * img1_width + x] / 255.f;
        }
    }

    // matrices made during epochs
    /*Mat ti = {
        .rows = t.rows,
        .cols = 2,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, 0)};

    Mat to = {
        .rows = t.rows,
        .cols = 1,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, ti.cols)};*/

    // MAT_PRINT(ti);
    // MAT_PRINT(to);

    nn_rand(nn, -1, 1);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "Train");
    SetTargetFPS(120);

    Cost_Plot plot = {0};

    size_t preview_width = 28;
    size_t preview_height = 28;

    Image preview_image1 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture1 = LoadTextureFromImage(preview_image1);

    Image preview_image2 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture2 = LoadTextureFromImage(preview_image2);

    Image preview_image3 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture3 = LoadTextureFromImage(preview_image3);

    Image original_image1 = GenImageColor(img1_width, img1_height, BLACK);
    for (size_t y = 0; y < (size_t)img1_height; y++)
    {
        for (size_t x = 0; x < (size_t)img1_width; x++)
        {
            uint8_t pixel = img1_pixels[y * img1_width + x];
            ImageDrawPixel(&original_image1, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
        }
    }
    Texture2D original_texture1 = LoadTextureFromImage(original_image1);

    Image original_image2 = GenImageColor(img2_width, img2_height, BLACK);
    for (size_t y = 0; y < (size_t)img2_height; y++)
    {
        for (size_t x = 0; x < (size_t)img2_width; x++)
        {
            uint8_t pixel = img2_pixels[y * img2_width + x];
            ImageDrawPixel(&original_image2, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
        }
    }
    Texture2D original_texture2 = LoadTextureFromImage(original_image2);

    Batch batch = {0};
    size_t epoch = 0;
    bool scroll_dragging = false;
    float scroll = 0.5f;

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
                    MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
                    nn_forward(nn);
                    float a = MAT_AT(NN_OUTPUT(nn), 0, 0);
                    if (a < 0)
                        a = 0;
                    if (a > 1)
                        a = 1;
                    uint8_t pixel = a * 255.f;
                    out_pixels[y * out_width + x] = pixel;
                }
            }

            char outbuffer[256];
            snprintf(outbuffer, sizeof(outbuffer), "./mnist/output/morph.png");
            if (!stbi_write_png(outbuffer, out_width, out_height, 1, out_pixels, out_width * sizeof(*out_pixels)))
                fprintf(stderr, "ERROR : Could not save image as %s\n", outbuffer);
            else
                printf("Saved image %s\n", outbuffer);

            free(out_pixels);
        }

        for (size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; i++)
        {
            batch_process(&temp, &batch, batch_size, nn, t, rate);

            if (batch.finished)
            {
                epoch++;
                da_append(&plot, batch.cost);
                mat_shuffle_rows(t);
            }
        }

        BeginDrawing();
        ClearBackground((Color){0x18, 0x18, 0x18, 0xFF});
        int w = GetRenderWidth();
        int h = GetRenderHeight();

        char buffer[256];
        // snprintf(buffer, sizeof(buffer), "Epoch : %zu/%zu\t\tRate : %f\t\tCost: %f", epoch, max_epoch, rate, cost);
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

        float scale = 8.0f;

        for (size_t y = 0; y < preview_width; y++)
        {
            for (size_t x = 0; x < preview_height; x++)
            {
                MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (img1_width - 1);
                MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (img1_height - 1);
                MAT_AT(NN_INPUT(nn), 0, 2) = 0.0f;
                nn_forward(nn);
                float a = MAT_AT(NN_OUTPUT(nn), 0, 0);
                if (a < 0)
                    a = 0;
                if (a > 1)
                    a = 1;
                uint8_t pixel = a * 255.f;
                ImageDrawPixel(&preview_image1, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
            }
        }

        for (size_t y = 0; y < preview_width; y++)
        {
            for (size_t x = 0; x < preview_height; x++)
            {
                MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (img2_width - 1);
                MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (img2_height - 1);
                MAT_AT(NN_INPUT(nn), 0, 2) = 1.0f;
                nn_forward(nn);
                float a = MAT_AT(NN_OUTPUT(nn), 0, 0);
                if (a < 0)
                    a = 0;
                if (a > 1)
                    a = 1;
                uint8_t pixel = a * 255.f;
                ImageDrawPixel(&preview_image2, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
            }
        }

        for (size_t y = 0; y < preview_width; y++)
        {
            for (size_t x = 0; x < preview_height; x++)
            {
                MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (preview_width - 1);
                MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (preview_height - 1);
                MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
                nn_forward(nn);
                float a = MAT_AT(NN_OUTPUT(nn), 0, 0);
                if (a < 0)
                    a = 0;
                if (a > 1)
                    a = 1;
                uint8_t pixel = a * 255.f;
                ImageDrawPixel(&preview_image3, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
            }
        }

        float padding = scale * 2.0f;
        float start_y = ry - (img1_height * scale) * 0.3f;

        UpdateTexture(preview_texture1, preview_image1.data);
        DrawTextureEx(preview_texture1, CLITERAL(Vector2){rx, start_y}, 0, scale, WHITE);
        DrawTextureEx(original_texture1, CLITERAL(Vector2){rx, start_y + preview_height * scale + padding}, 0, scale, WHITE);

        UpdateTexture(preview_texture2, preview_image2.data);
        DrawTextureEx(preview_texture2, CLITERAL(Vector2){rx + img1_width * scale + padding, start_y}, 0, scale, WHITE);
        DrawTextureEx(original_texture2, CLITERAL(Vector2){rx + img1_width * scale + padding, start_y + preview_height * scale + padding}, 0, scale, WHITE);

        UpdateTexture(preview_texture3, preview_image3.data);
        DrawTextureEx(preview_texture3, CLITERAL(Vector2){rx + img1_width * scale * 0.5f + padding * 0.5f, start_y + preview_height * scale * 2 + padding * 5.2f}, 0, scale, WHITE);

        {
            Vector2 size = {img1_width * scale * 2 + padding, rh * 0.02f};
            Vector2 pos = {rx, start_y + preview_height * scale * 2 + padding * 3.0f};
            float slider_radius = rh * 0.023f;
            Vector2 slider_pos = {rx + size.x * scroll, pos.y + size.y * 0.5f};

            DrawRectangleRounded(CLITERAL(Rectangle){pos.x, pos.y, size.x, size.y}, 0.5f, 10, RAYWHITE);
            DrawCircleV(slider_pos, slider_radius, RED);

            if (scroll_dragging)
            {
                Vector2 mouse_pos = GetMousePosition();
                float x = mouse_pos.x;
                if (x < pos.x)
                    x = pos.x;
                if (x > pos.x + size.x)
                    x = pos.x + size.x;

                scroll = (x - pos.x) / size.x;
            }

            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
            {
                Vector2 mouse_pos = GetMousePosition();
                if (Vector2Distance(mouse_pos, slider_pos) <= slider_radius)
                {
                    scroll_dragging = true;
                }
            }

            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
            {
                scroll_dragging = false;
            }
        }

        EndDrawing();
        region_reset(&temp);
    }

    printf("\n\n");

    for (size_t y = 0; y < (size_t)preview_height; y++)
    {
        for (size_t x = 0; x < (size_t)preview_width; x++)
        {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (img1_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (img1_height - 1);
            MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
            nn_forward(nn);
            float a = MAT_AT(NN_OUTPUT(nn), 0, 0);
            if (a < 0)
                a = 0;
            if (a > 1)
                a = 1;
            uint8_t pixel = a * 255.f;
            if (pixel)
                printf("%3u ", pixel);
            else
                printf("    ");
        }
        printf("\n");
    }

    printf("\n");

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
            MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
            nn_forward(nn);
            float a = MAT_AT(NN_OUTPUT(nn), 0, 0);
            if (a < 0)
                a = 0;
            if (a > 1)
                a = 1;
            uint8_t pixel = a * 255.f;
            out_pixels[y * out_width + x] = pixel;
        }
    }

    char outbuffer[256];
    snprintf(outbuffer, sizeof(outbuffer), "./mnist/output/morph.png");
    // snprintf(outbuffer, sizeof(outbuffer), "./mnist/output/upscaled%c.png", img1_file_path[strlen(img1_file_path) - 5]);
    //  snprintf(outbuffer, sizeof(outbuffer), "./mnist/output/upscaled.png");

    const char *out_file_path = outbuffer;

    if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width * sizeof(*out_pixels)))
        fprintf(stderr, "ERROR : Could not save image as %s\n", out_file_path);

    printf("Generated image %s\n", out_file_path);

    return 0;
}