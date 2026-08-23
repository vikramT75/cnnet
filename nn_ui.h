#ifndef NN_UI_H_
#define NN_UI_H_

#include <raylib.h>
#include <raymath.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"

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

void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh);
void cost_plot_minmax(Cost_Plot plot, float *min, float *max);
void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh);

#endif // NN_UI_H_

#ifdef NN_UI_IMPLEMENTATION
#ifndef NN_UI_IMPLEMENTATION_DONE
#define NN_UI_IMPLEMENTATION_DONE

void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh)
{
    Color low_colour = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_colour = {0x00, 0xFF, 0x00, 0x00};
    Color background_colour = {0x18, 0x18, 0x18, 0xFF};
    (void)background_colour;

    // ClearBackground(background_colour);

    float neuron_radius = rh * 0.04;
    int layer_border_vpad = 50;
    int layer_border_hpad = 50;
    int nn_width = rw - 2 * layer_border_hpad;
    int nn_height = rh - 2 * layer_border_vpad;
    int nn_x = rx + rw / 2 - nn_width / 2;
    int nn_y = ry + rh / 2 - nn_height / 2;
    int layer_hpad = nn_width / (nn.arch_count);
    for (size_t l = 0; l < nn.arch_count; l++)
    {

        int layer_vpad1 = nn_height / nn.as[l].cols;

        for (size_t i = 0; i < nn.as[l].cols; i++)
        {
            int cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
            int cy1 = nn_y + i * layer_vpad1 + layer_vpad1 / 2;

            if (l + 1 < nn.arch_count)
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

#endif // NN_UI_IMPLEMENTATION_DONE
#endif // NN_UI_IMPLEMENTATION
