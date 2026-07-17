#include <time.h>
#include "raylib.h"
#define NN_IMPLEMENTAION
#include "../nn.h"

#define BITS 4

#define IMG_FACTOR 80
#define IMG_WIDTH (16 * IMG_FACTOR)
#define IMG_HEIGHT (9 * IMG_FACTOR)

void nn_render_raylib(NN nn)
{
    Color low_colour = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_colour = {0x00, 0xFF, 0x00, 0x00};
    Color background_colour = {0x18, 0x18, 0x18, 0xFF};

    ClearBackground(background_colour);

    int neuron_radius = 25;
    int layer_border_vpad = 50;
    int layer_border_hpad = 50;
    int nn_width = IMG_WIDTH - 2 * layer_border_hpad;
    int nn_height = IMG_HEIGHT - 2 * layer_border_vpad;
    int nn_x = IMG_WIDTH / 2 - nn_width / 2;
    int nn_y = IMG_HEIGHT / 2 - nn_height / 2;
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
                    high_colour.a = floorf(255.f * sigmoidf(MAT_AT(nn.ws[l], i, j)));
                    DrawLine(cx1, cy1, cx2, cy2, ColorAlphaBlend(low_colour, high_colour, RAYWHITE));
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

int main()
{
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "Adder NN");
    SetTargetFPS(120);

    srand(time(0));
    size_t n = (1 << BITS);
    size_t rows = n * n;
    Mat ti = mat_alloc(rows, 2 * BITS);
    Mat to = mat_alloc(rows, BITS + 1);

    for (size_t i = 0; i < ti.rows; i++)
    {
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        for (size_t j = 0; j < BITS; j++)
        {
            MAT_AT(ti, i, j) = (x >> j) & 1;
            MAT_AT(ti, i, j + BITS) = (y >> j) & 1;
            MAT_AT(to, i, j) = (z >> j) & 1;
        }
        MAT_AT(to, i, BITS) = z >= n;
    }

    // MAT_PRINT(ti);
    // MAT_PRINT(to);

    size_t arch[] = {2 * BITS, 4 * BITS, BITS + 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);
    NN_PRINT(nn);

    float rate = 1;

    // float pt = nn_cost(nn, ti, to);
    size_t i = 0;
    while (!WindowShouldClose())
    {
        if (i < 10000)
        {
            nn_backprop(nn, g, ti, to);
            nn_learn(nn, g, rate);
            printf("cost#%d = %f\n", (i + 1), nn_cost(nn, ti, to));
            i++;
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);
        nn_render_raylib(nn);
        EndDrawing();
    }

    /*printf("-------------------------------\n");
    printf("PRE TRAIN Cost = %f\n", pt);
    printf("POST TRAIN Cost = %f\n", nn_cost(nn, ti, to));
    printf("-------------------------------\n");*/

    for (size_t x = 0; x < n; x++)
    {
        for (size_t y = 0; y < n; y++)
        {
            printf("%d + %d = ", x, y);
            for (size_t j = 0; j < BITS; j++)
            {
                MAT_AT(NN_INPUT(nn), 0, j) = (x >> j) & 1;
                MAT_AT(NN_INPUT(nn), 0, j + BITS) = (y >> j) & 1;
            }
            nn_forward(nn);
            if (MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f)
                printf("OVERFLOW\n");
            else
            {
                size_t z = 0;
                for (size_t j = 0; j < BITS; j++)
                {
                    size_t bit = MAT_AT(NN_OUTPUT(nn), 0, j) > 0.5f;
                    z |= bit << j;
                }
                printf("%d\n", z);
            }
        }
    }

    return 0;
}