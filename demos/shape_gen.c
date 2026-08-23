#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define NN_IMPLEMENTATION
#include "../nn.h"

#define IMG_SIZE 28
#define NUM_INPUTS (IMG_SIZE * IMG_SIZE) // 784
#define NUM_CLASSES 3
#define NUM_COLS (NUM_INPUTS + NUM_CLASSES) // 787

#define NUM_TRAIN 50000
#define NUM_TEST 7000

#define CLASS_CIRCLE 0
#define CLASS_SQUARE 1
#define CLASS_TRIANGLE 2

#ifndef PI
#define PI 3.14159265359f
#endif

static void save_mat(FILE *out, Mat m)
{
    const char *magic = "nn.h.mat";
    fwrite(magic, strlen(magic), 1, out);
    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);
    for (size_t i = 0; i < m.rows; i++)
    {
        fwrite(&MAT_AT(m, i, 0), sizeof(float), m.cols, out);
    }
}

static float randf_range(float lo, float hi)
{
    return lo + ((float)rand() / (float)RAND_MAX) * (hi - lo);
}

static int point_in_triangle(float px, float py,
                             float x0, float y0,
                             float x1, float y1,
                             float x2, float y2)
{
    float d1 = (px - x1) * (y0 - y1) - (x0 - x1) * (py - y1);
    float d2 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2);
    float d3 = (px - x0) * (y2 - y0) - (x2 - x0) * (py - y0);
    int has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0);
    int has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0);
    return !(has_neg & has_pos);
}

static void generate_sample(float pixels[NUM_INPUTS], int class_id)
{
    float target_size = randf_range(10.0f, 22.0f);
    float r;

    if (class_id == CLASS_TRIANGLE)
    {
        r = target_size / 1.73205081f;
    }
    else
    {
        r = target_size / 2.0f;
    }

    float cx = randf_range(14.0f - (24.0f - target_size) / 2.0f, 14.0f + (24.0f - target_size) / 2.0f);
    float cy = randf_range(14.0f - (24.0f - target_size) / 2.0f, 14.0f + (24.0f - target_size) / 2.0f);

    float theta = randf_range(0.0f, 2.0f * PI);
    float c_theta = cosf(theta);
    float s_theta = sinf(theta);

    int is_hollow = (rand() % 2 == 0);
    float inner_r = r - randf_range(1.5f, 3.0f);
    if (inner_r < 1.0f)
        inner_r = 1.0f;

    for (int y = 0; y < IMG_SIZE; y++)
    {
        for (int x = 0; x < IMG_SIZE; x++)
        {
            float px = (float)x + 0.5f;
            float py = (float)y + 0.5f;
            int inside = 0;

            float dx = px - cx;
            float dy = py - cy;

            float rdx = dx * c_theta - dy * s_theta;
            float rdy = dx * s_theta + dy * c_theta;

            switch (class_id)
            {
            case CLASS_CIRCLE:
            {
                int outer = (dx * dx + dy * dy) <= (r * r);
                int inner = (dx * dx + dy * dy) <= (inner_r * inner_r);
                inside = is_hollow ? (outer && !inner) : outer;
            }
            break;

            case CLASS_SQUARE:
            {
                int outer = (fabsf(rdx) <= r) & (fabsf(rdy) <= r);
                int inner = (fabsf(rdx) <= inner_r) & (fabsf(rdy) <= inner_r);
                inside = is_hollow ? (outer && !inner) : outer;
            }
            break;

            case CLASS_TRIANGLE:
            {
                float sqrt3 = 1.73205081f;
                float bx0 = 0.0f, by0 = -r;
                float bx1 = -r * sqrt3 / 2.0f, by1 = r / 2.0f;
                float bx2 = r * sqrt3 / 2.0f, by2 = r / 2.0f;

                float rx0 = bx0 * c_theta - by0 * s_theta;
                float ry0 = bx0 * s_theta + by0 * c_theta;
                float rx1 = bx1 * c_theta - by1 * s_theta;
                float ry1 = bx1 * s_theta + by1 * c_theta;
                float rx2 = bx2 * c_theta - by2 * s_theta;
                float ry2 = bx2 * s_theta + by2 * c_theta;

                float min_x = fminf(fminf(rx0, rx1), rx2);
                float max_x = fmaxf(fmaxf(rx0, rx1), rx2);
                float min_y = fminf(fminf(ry0, ry1), ry2);
                float max_y = fmaxf(fmaxf(ry0, ry1), ry2);

                float bbc_x = (min_x + max_x) / 2.0f;
                float bbc_y = (min_y + max_y) / 2.0f;

                float tx0 = rx0 - bbc_x, ty0 = ry0 - bbc_y;
                float tx1 = rx1 - bbc_x, ty1 = ry1 - bbc_y;
                float tx2 = rx2 - bbc_x, ty2 = ry2 - bbc_y;

                int outer = point_in_triangle(dx, dy, tx0, ty0, tx1, ty1, tx2, ty2);

                float f = inner_r / r;
                float cx_cc = -bbc_x;
                float cy_cc = -bbc_y;

                float ix0 = cx_cc + f * (tx0 - cx_cc);
                float iy0 = cy_cc + f * (ty0 - cy_cc);
                float ix1 = cx_cc + f * (tx1 - cx_cc);
                float iy1 = cy_cc + f * (ty1 - cy_cc);
                float ix2 = cx_cc + f * (tx2 - cx_cc);
                float iy2 = cy_cc + f * (ty2 - cy_cc);

                int inner = point_in_triangle(dx, dy, ix0, iy0, ix1, iy1, ix2, iy2);
                inside = is_hollow ? (outer && !inner) : outer;
            }
            break;
            }

            float val = inside ? randf_range(0.80f, 1.00f) : randf_range(0.00f, 0.12f);
            pixels[y * IMG_SIZE + x] = val;
        }
    }
}

static void fill_dataset(Mat t, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        int class_id = (int)(i % NUM_CLASSES);

        float pixels[NUM_INPUTS];
        generate_sample(pixels, class_id);

        for (size_t j = 0; j < NUM_INPUTS; j++)
            MAT_AT(t, i, j) = pixels[j];

        for (size_t j = 0; j < NUM_CLASSES; j++)
            MAT_AT(t, i, NUM_INPUTS + j) = (j == (size_t)class_id) ? 1.0f : 0.0f;
    }

    mat_shuffle_rows(t);
}

static int write_dataset(Mat t, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "ERROR: could not open '%s' for writing\n", path);
        return 0;
    }
    save_mat(f, t);
    fclose(f);
    printf("Saved %zu samples → %s\n", t.rows, path);
    return 1;
}

int main(void)
{
    srand((unsigned)time(NULL));

    printf("Generating shapes dataset (28x28, %d classes)...\n", NUM_CLASSES);

    Mat train = mat_alloc(NULL, NUM_TRAIN, NUM_COLS);
    fill_dataset(train, NUM_TRAIN);
    if (!write_dataset(train, "demos/data/shapes_train.mat"))
        return 1;

    Mat test = mat_alloc(NULL, NUM_TEST, NUM_COLS);
    fill_dataset(test, NUM_TEST);
    if (!write_dataset(test, "demos/data/shapes_test.mat"))
        return 1;

    printf("Done.\n");
    return 0;
}
