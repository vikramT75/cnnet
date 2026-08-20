#include <time.h>

#define NN_ACT ACT_SIG
#define NN_BACKPROP_TRADITIONAL
#define NN_IMPLEMENTATION
#include "../nn.h"

#define NN_UI_IMPLEMENTATION
#include "../nn_ui.h"

#define IMG_FACTOR 80
#define IMG_WIDTH (16 * IMG_FACTOR)
#define IMG_HEIGHT (9 * IMG_FACTOR)

size_t max_epoch = 20000;
size_t batches_per_frame = 100;
size_t batch_size = 4;
bool paused = true;
float rate = 1.f;
size_t arch[] = {2, 4, 1};

int main()
{
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "XOR NN");
    SetTargetFPS(120);

    srand(time(0));

    size_t rows = 4;
    Mat t = mat_alloc(NULL, rows, 3);

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            size_t row = i * 2 + j;
            MAT_AT(t, row, 0) = i;
            MAT_AT(t, row, 1) = j;
            MAT_AT(t, row, 2) = i ^ j;
        }
    }

    Region temp = region_alloc_allocator(256 * 1024 * 1024);
    NN nn = nn_alloc(NULL, arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);
    NN_PRINT(nn);

    float cost = 0.0f;
    Cost_Plot plot = {0};

    size_t epoch = 0;
    Batch batch = {0};

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

        int w = GetRenderWidth();
        int h = GetRenderHeight();

        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Epoch : %zu/%zu\t\tRate : %f\t\tCost : %f", epoch, max_epoch, rate, cost);
        DrawText(buffer, 0, 0, h * 0.04, WHITE);

        int rw, rh, rx, ry;

        rw = w / 3;
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

        rx += rw;
        nn_render_raylib(nn, rx, ry, rw, rh);

        rx += rw;

        int text_x = rx + rw * 0.1;
        int text_y = ry + rh * 0.2;
        int row_height = h * 0.1;
        int table_font_size = h * 0.05;

        DrawText("Truth Table", text_x, text_y - row_height, table_font_size * 1.2, GREEN);

        for (size_t x = 0; x < 2; x++)
        {
            for (size_t y = 0; y < 2; y++)
            {
                MAT_AT(NN_INPUT(nn), 0, 0) = x;
                MAT_AT(NN_INPUT(nn), 0, 1) = y;
                nn_forward(nn);
                float out = MAT_AT(NN_OUTPUT(nn), 0, 0);
                char txt[256];
                snprintf(txt, sizeof(txt), "%zu ^ %zu = %.4f", x, y, out);

                Color c = WHITE;
                if (out > 0.9f && (x ^ y) == 1)
                    c = GREEN;
                else if (out < 0.1f && (x ^ y) == 0)
                    c = GREEN;
                else
                    c = RED;

                DrawText(txt, text_x, text_y, table_font_size, c);
                text_y += row_height;
            }
        }

        EndDrawing();
        region_reset(&temp);
    }

    printf("FINAL PREDICTIONS:\n");
    for (size_t x = 0; x < 2; x++)
    {
        for (size_t y = 0; y < 2; y++)
        {
            MAT_AT(NN_INPUT(nn), 0, 0) = x;
            MAT_AT(NN_INPUT(nn), 0, 1) = y;
            nn_forward(nn);
            printf("%zu ^ %zu = %f\n", x, y, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }
    }

    return 0;
}
