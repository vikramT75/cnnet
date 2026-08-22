#include <time.h>
#include <stdio.h>
#include <float.h>

#define NN_ACT ACT_SIG
#define NN_BACKPROP_TRADITIONAL
#define NN_IMPLEMENTATION
#include "../nn.h"

#define NN_UI_IMPLEMENTATION
#include "../nn_ui.h"

#define SV_IMPLEMENTATION
#include "../thirdparty/sv.h"

#define BITS 3

#define IMG_FACTOR 80
#define IMG_WIDTH (16 * IMG_FACTOR)
#define IMG_HEIGHT (9 * IMG_FACTOR)

size_t max_epoch = 20000;
size_t batches_per_frame = 100;
size_t batch_size = 28;
bool paused = true;
float rate = 1.0f;

typedef struct
{
    size_t *items;
    size_t count;
    size_t capacity;
} Arch;

Arch arch = {0};

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

    content = sv_trim_left(content);
    while (content.count > 0 && isdigit(content.data[0]))
    {
        size_t x = sv_chop_u64(&content);
        da_append(&arch, x);
        printf("%zu\n", x);
        content = sv_trim_left(content);
    }

    FILE *in = fopen(data_file_path, "rb");
    if (in == NULL)
    {
        fprintf(stderr, "Error : Could not read file %s\n", data_file_path);
        return 1;
    }

    Mat t = mat_load(NULL, in);
    fclose(in);

    NN_ASSERT(arch.count > 1);
    size_t ins_sz = arch.items[0];
    size_t outs_sz = arch.items[arch.count - 1];
    NN_ASSERT(t.cols == ins_sz + outs_sz);

    Region temp = region_alloc_allocator(256 * 1024 * 1024);
    NN nn = nn_alloc(NULL, arch.items, arch.count);
    nn_rand(nn, -1, 1);
    NN_PRINT(nn);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "Train");
    SetTargetFPS(120);

    float cost = 0.0f;
    Cost_Plot plot = {0};

    size_t epoch = 0;
    Batch batch = {0};

    batch_size = 28;

    while (!WindowShouldClose())
    {

        if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_P))
            paused = !paused;

        if (IsKeyPressed(KEY_R))
        {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
            batch.finished = false;
            batch.begin = 0;
            batch.cost = 0;
        }

        for (size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; i++)
        {
            batch_process(&temp, &batch, batch_size, nn, t, rate);
            if (batch.finished)
            {
                epoch++;
                cost = batch.cost;
                da_append(&plot, cost);
                mat_shuffle_rows(t);
            }
        }

        BeginDrawing();
        ClearBackground((Color){0x18, 0x18, 0x18, 0xFF});

        int rw, rh, rx, ry;
        int w = GetRenderWidth();
        int h = GetRenderHeight();

        rw = w / 2;
        rh = h * 2 / 3;
        rx = 0;
        ry = h / 2 - rh / 2;
        plot_cost(plot, rx, ry, rw, rh);

        int zero_y = ry + rh;

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
        region_reset(&temp);
    }

    return 0;
}
