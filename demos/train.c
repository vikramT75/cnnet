#include <time.h>
#include <stdio.h>
#include <float.h>
#include "raylib.h"
#define NN_IMPLEMENTAION
#include "../nn.h"
#define SV_IMPLEMENTATION
#include "../thirdparty/sv.h"

#define BITS 3

#define IMG_FACTOR 80
#define IMG_WIDTH (16 * IMG_FACTOR)
#define IMG_HEIGHT (9 * IMG_FACTOR)
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
    const char *program = args_shift(&argc, &argv);

    if (argc <= 0)
    {
        fprintf(stderr, "Usage : %s <model.arch> <model.mat>\n", program);
        fprintf(stderr, "ERROR: architecture file not provided\n");
        return 1;
    }
    const char *arch_file_path = args_shift(&argc, &argv);

    if (argc <= 0)
    {
        fprintf(stderr, "Usage : %s <model.arch> <model.mat>\n", program);
        fprintf(stderr, "ERROR: data file not provided\n");
        return 1;
    }
    const char *data_file_path = args_shift(&argc, &argv);

    unsigned int buffer_len = 0;
    unsigned char *buffer = LoadFileData(arch_file_path, &buffer_len);
    if (buffer == NULL)
    {
        return 1;
    }

    String_View content = sv_from_parts((const char *)buffer, buffer_len);

    Arch arch = {0};
    content = sv_trim_left(content);
    while (content.count > 0 && isdigit(content.data[0]))
    {
        size_t x = sv_chop_u64(&content);
        da_append(&arch, x);
        printf("%d\n", x);
        content = sv_trim_left(content);
    }

    FILE *in = fopen(data_file_path, "rb");
    if (in == NULL)
    {
        fprintf(stderr, "Error : Could not read file %s\n", data_file_path);
        return 1;
    }

    Mat t = mat_load(in);
    fclose(in);

    NN_ASSERT(arch.count > 1);
    size_t ins_sz = arch.items[0];
    size_t outs_sz = arch.items[arch.count - 1];
    NN_ASSERT(t.cols == ins_sz + outs_sz);

    Mat ti = {
        .rows = t.rows,
        .cols = ins_sz,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, 0)};

    Mat to = {
        .rows = t.rows,
        .cols = outs_sz,
        .stride = t.stride,
        .es = &MAT_AT(t, 0, ins_sz)};

    MAT_PRINT(ti);
    MAT_PRINT(to);

    NN nn = nn_alloc(arch.items, arch.count);
    NN g = nn_alloc(arch.items, arch.count);
    nn_rand(nn, 0, 1);
    NN_PRINT(nn);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "Train");
    SetTargetFPS(120);

    float rate = 1;
    float cost = 0;
    Cost_Plot plot = {0};

    size_t epoch = 0;
    size_t max_epoch = 20000;
    size_t epochs_per_frame = 10;
    bool paused = true;
    while (!WindowShouldClose())
    {

        if (IsKeyPressed(KEY_SPACE))
            paused = !paused;

        if (IsKeyPressed(KEY_R))
        {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        for (size_t i = 0; i < epochs_per_frame && !paused && epoch < max_epoch; i++)
        {
            if (epoch < max_epoch)
            {
                nn_backprop(nn, g, ti, to);
                nn_learn(nn, g, rate);
                cost = nn_cost(nn, ti, to);
                // printf("cost#%d = %f\n", (epoch + 1), cost);
                //  if (epoch % 100 == 0)
                da_append(&plot, nn_cost(nn, ti, to));
                epoch++;
            }
        }

        BeginDrawing();
        ClearBackground((Color){0x18, 0x18, 0x18, 0xFF});
        //   ClearBackground(RAYWHITE);
        int rw, rh, rx, ry;
        int w = GetRenderWidth();
        int h = GetRenderHeight();

        rw = w / 2;
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

        rw = w / 2;
        rh = h * 2 / 3;
        rx = w - rw;
        ry = h / 2 - rh / 2;
        nn_render_raylib(nn, rx, ry, rw, rh);

        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Epoch : %zu/%zu\t\tRate : %f\t\tCost: %f", epoch, max_epoch, rate, cost);
        DrawText(buffer, 0, 0, h * 0.04, WHITE);

        EndDrawing();
    }

    return 0;
}