// Usage:
//   ./demos/trainer <model.arch> <train.mat> <test.mat> <max_epochs>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define NN_ACT ACT_SIG
#define NN_CROSS_ENTROPY // use cross-entropy gradient at output layer
#define NN_IMPLEMENTATION
#include "../nn.h"

typedef struct
{
    size_t *items;
    size_t count;
    size_t capacity;
} Arch;

#define BATCH_SIZE 32

// Adam hyper-parameters
#define ADAM_LR 0.001f
#define ADAM_B1 0.9f
#define ADAM_B2 0.999f
#define ADAM_EPS 1e-8f

static char *args_shift(int *argc, char ***argv)
{
    NN_ASSERT(*argc > 0);
    char *r = **argv;
    (*argc)--;
    (*argv)++;
    return r;
}

static size_t nn_argmax(NN nn)
{
    size_t best = 0;
    float bval = MAT_AT(NN_OUTPUT(nn), 0, 0);
    for (size_t j = 1; j < NN_OUTPUT(nn).cols; j++)
    {
        if (MAT_AT(NN_OUTPUT(nn), 0, j) > bval)
        {
            bval = MAT_AT(NN_OUTPUT(nn), 0, j);
            best = j;
        }
    }
    return best;
}

static size_t row_argmax(Mat m, size_t row)
{
    size_t best = 0;
    float bval = MAT_AT(m, row, 0);
    for (size_t j = 1; j < m.cols; j++)
    {
        if (MAT_AT(m, row, j) > bval)
        {
            bval = MAT_AT(m, row, j);
            best = j;
        }
    }
    return best;
}

static float eval_test_accuracy(NN nn, Mat test_ti, Mat test_to)
{
    size_t correct = 0;
    for (size_t i = 0; i < test_ti.rows; i++)
    {
        mat_copy(NN_INPUT(nn), mat_row(test_ti, i));
        nn_forward(nn);
        if (nn_argmax(nn) == row_argmax(test_to, i))
            correct++;
    }
    return 100.f * (float)correct / (float)test_ti.rows;
}

static void save_weights(NN nn, const char *arch_path)
{
    char path[512];
    snprintf(path, sizeof(path), "%s.weights.mat", arch_path);
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "ERROR: cannot save to %s\n", path);
        return;
    }

    const char *magic = "nn.h.mat";
    for (size_t l = 0; l < nn.arch_count - 1; l++)
    {
        Mat w = nn.ws[l];
        fwrite(magic, strlen(magic), 1, f);
        fwrite(&w.rows, sizeof(w.rows), 1, f);
        fwrite(&w.cols, sizeof(w.cols), 1, f);
        for (size_t r = 0; r < w.rows; r++)
            fwrite(&MAT_AT(w, r, 0), sizeof(float), w.cols, f);

        Mat b = nn.bs[l];
        fwrite(magic, strlen(magic), 1, f);
        fwrite(&b.rows, sizeof(b.rows), 1, f);
        fwrite(&b.cols, sizeof(b.cols), 1, f);
        for (size_t r = 0; r < b.rows; r++)
            fwrite(&MAT_AT(b, r, 0), sizeof(float), b.cols, f);
    }
    fclose(f);
}

static void adam_learn(NN nn, NN g, NN m_nn, NN v_nn, size_t t)
{
    float bc1 = 1.f - powf(ADAM_B1, (float)t);
    float bc2 = 1.f - powf(ADAM_B2, (float)t);

    for (size_t l = 0; l < nn.arch_count - 1; l++)
    {
        for (size_t r = 0; r < nn.ws[l].rows; r++)
        {
            for (size_t c = 0; c < nn.ws[l].cols; c++)
            {
                float grad = MAT_AT(g.ws[l], r, c);
                float *m = &MAT_AT(m_nn.ws[l], r, c);
                float *v = &MAT_AT(v_nn.ws[l], r, c);

                *m = ADAM_B1 * (*m) + (1.f - ADAM_B1) * grad;
                *v = ADAM_B2 * (*v) + (1.f - ADAM_B2) * grad * grad;

                float m_hat = (*m) / bc1;
                float v_hat = (*v) / bc2;

                MAT_AT(nn.ws[l], r, c) -= ADAM_LR * m_hat / (sqrtf(v_hat) + ADAM_EPS);
            }
        }
        for (size_t r = 0; r < nn.bs[l].rows; r++)
        {
            for (size_t c = 0; c < nn.bs[l].cols; c++)
            {
                float grad = MAT_AT(g.bs[l], r, c);
                float *m = &MAT_AT(m_nn.bs[l], r, c);
                float *v = &MAT_AT(v_nn.bs[l], r, c);

                *m = ADAM_B1 * (*m) + (1.f - ADAM_B1) * grad;
                *v = ADAM_B2 * (*v) + (1.f - ADAM_B2) * grad * grad;

                float m_hat = (*m) / bc1;
                float v_hat = (*v) / bc2;

                MAT_AT(nn.bs[l], r, c) -= ADAM_LR * m_hat / (sqrtf(v_hat) + ADAM_EPS);
            }
        }
    }
}

int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));
    const char *program = args_shift(&argc, &argv);

    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <model.arch> <train.mat> <test.mat> <max_epochs>\n", program);
        return 1;
    }

    const char *arch_path = args_shift(&argc, &argv);
    const char *train_path = args_shift(&argc, &argv);
    const char *test_path = args_shift(&argc, &argv);
    size_t max_epochs = (size_t)atoi(args_shift(&argc, &argv));

    size_t arch_items[256];
    size_t arch_count = 0;
    FILE *af = fopen(arch_path, "r");
    if (!af)
    {
        fprintf(stderr, "ERROR: could not read arch file '%s'\n", arch_path);
        return 1;
    }
    size_t layer_size;
    while (fscanf(af, "%zu", &layer_size) == 1)
        if (arch_count < 256)
            arch_items[arch_count++] = layer_size;
    fclose(af);

    if (arch_count < 2)
    {
        fprintf(stderr, "ERROR: arch must have at least 2 layers\n");
        return 1;
    }
    size_t ins = arch_items[0];
    size_t outs = arch_items[arch_count - 1];

    FILE *f;
    f = fopen(train_path, "rb");
    if (!f)
    {
        fprintf(stderr, "ERROR: cannot open '%s'\n", train_path);
        return 1;
    }
    Mat train_t = mat_load(NULL, f);
    fclose(f);

    f = fopen(test_path, "rb");
    if (!f)
    {
        fprintf(stderr, "ERROR: cannot open '%s'\n", test_path);
        return 1;
    }
    Mat test_t = mat_load(NULL, f);
    fclose(f);

    NN_ASSERT(train_t.cols == ins + outs);
    NN_ASSERT(test_t.cols == ins + outs);
    printf("Train: %zu   Test: %zu   Arch:", train_t.rows, test_t.rows);
    for (size_t i = 0; i < arch_count; i++)
        printf(" %zu", arch_items[i]);
    printf("\n");

    Mat train_ti = {.rows = train_t.rows, .cols = ins, .stride = train_t.stride, .es = &MAT_AT(train_t, 0, 0)};
    Mat train_to = {.rows = train_t.rows, .cols = outs, .stride = train_t.stride, .es = &MAT_AT(train_t, 0, ins)};
    Mat test_ti = {.rows = test_t.rows, .cols = ins, .stride = test_t.stride, .es = &MAT_AT(test_t, 0, 0)};
    Mat test_to = {.rows = test_t.rows, .cols = outs, .stride = test_t.stride, .es = &MAT_AT(test_t, 0, ins)};

    Region temp = region_alloc_allocator(256 * 1024 * 1024);
    NN nn = nn_alloc(NULL, arch_items, arch_count);
    for (size_t l = 0; l < nn.arch_count - 1; l++)
    {
        float scale = sqrtf(6.0f / (float)(nn.ws[l].rows + nn.ws[l].cols));
        mat_rand(nn.ws[l], -scale, scale);
        mat_fill(nn.bs[l], 0.0f);
    }

    NN m_nn = nn_alloc(NULL, arch_items, arch_count);
    NN v_nn = nn_alloc(NULL, arch_items, arch_count);
    nn_zero(m_nn);
    nn_zero(v_nn);

    size_t n_batches = (train_t.rows + BATCH_SIZE - 1) / BATCH_SIZE;
    size_t adam_t = 0;
    float best_acc = 0.f;

    mat_shuffle_rows(train_t);

    printf("Adam optimizer: lr=%.4f  β1=%.3f  β2=%.3f  batch=%d\n",
           ADAM_LR, ADAM_B1, ADAM_B2, BATCH_SIZE);
    printf("Starting training for %zu epochs...\n\n", max_epochs);

    for (size_t epoch = 1; epoch <= max_epochs; epoch++)
    {
        clock_t start = clock();
        float epoch_cost = 0.f;

        for (size_t b = 0; b < n_batches; b++)
        {
            size_t begin = b * BATCH_SIZE;
            size_t size = (begin + BATCH_SIZE <= train_t.rows)
                              ? BATCH_SIZE
                              : (train_t.rows - begin);

            Mat batch_ti = {.rows = size, .cols = ins, .stride = train_t.stride, .es = &MAT_AT(train_t, begin, 0)};
            Mat batch_to = {.rows = size, .cols = outs, .stride = train_t.stride, .es = &MAT_AT(train_t, begin, ins)};

            NN g = nn_backprop(&temp, nn, batch_ti, batch_to);

            adam_t++;
            adam_learn(nn, g, m_nn, v_nn, adam_t);

            epoch_cost += nn_cost(nn, batch_ti, batch_to);

            region_reset(&temp);
        }

        epoch_cost /= (float)n_batches;

        clock_t end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

        float test_acc = eval_test_accuracy(nn, test_ti, test_to);

        const char *marker = "";
        if (test_acc > best_acc)
        {
            best_acc = test_acc;
            save_weights(nn, arch_path);
            marker = "  <- BEST, saved";
        }

        printf("Epoch %4zu/%zu | Loss: %7.6f | Acc: %5.1f%% | %.2fs%s\n",
               epoch, max_epochs, epoch_cost, test_acc, time_taken, marker);

        mat_shuffle_rows(train_t);
    }

    printf("\nBest test accuracy: %.1f%%\n", best_acc);
    return 0;
}
