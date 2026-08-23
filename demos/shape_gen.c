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

#define NUM_TRAIN 8000
#define NUM_TEST 2000

#define CLASS_CIRCLE 0
#define CLASS_SQUARE 1
#define CLASS_TRIANGLE 2

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
    float r = randf_range(5.0f, 10.0f);
    float cx = randf_range(r + 1.0f, IMG_SIZE - r - 1.0f);
    float cy = randf_range(r + 1.0f, IMG_SIZE - r - 1.0f);

    for (int y = 0; y < IMG_SIZE; y++)
    {
        for (int x = 0; x < IMG_SIZE; x++)
        {
            float px = (float)x + 0.5f;
            float py = (float)y + 0.5f;
            int inside = 0;

            switch (class_id)
            {
            case CLASS_CIRCLE:
            {
                float dx = px - cx, dy = py - cy;
                inside = (dx * dx + dy * dy) <= (r * r);
            }
            break;

            case CLASS_SQUARE:
            {
                inside = (fabsf(px - cx) <= r) & (fabsf(py - cy) <= r);
            }
            break;

            case CLASS_TRIANGLE:
            {
                float tx0 = cx, ty0 = cy - r;
                float tx1 = cx - r, ty1 = cy + r;
                float tx2 = cx + r, ty2 = cy + r;
                inside = point_in_triangle(px, py, tx0, ty0, tx1, ty1, tx2, ty2);
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
