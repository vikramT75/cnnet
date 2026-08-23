// Usage:
//   ./demos/infer <demos/models/model.arch> <demos/data/weights.mat> <demos/models/labels.label>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define NN_CROSS_ENTROPY
#define NN_ACT ACT_SIG
#define NN_IMPLEMENTATION
#include "../nn.h"

#define NN_UI_IMPLEMENTATION
#include "../nn_ui.h"

#define MAX_LABELS 32
#define LABEL_LEN 64
#define CANVAS_SIZE 28

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

static void softmax_copy(NN nn, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = MAT_AT(NN_OUTPUT(nn), 0, i);
}

static size_t load_labels(const char *path, char labels[MAX_LABELS][LABEL_LEN])
{
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    size_t n = 0;
    while (n < (size_t)MAX_LABELS && fgets(labels[n], LABEL_LEN, f))
    {
        size_t len = strlen(labels[n]);
        while (len > 0 && (labels[n][len - 1] == '\n' || labels[n][len - 1] == '\r'))
            labels[n][--len] = '\0';
        if (len > 0)
            n++;
    }
    fclose(f);
    return n;
}

static void load_weights(NN nn, const char *weights_path)
{
    FILE *f = fopen(weights_path, "rb");
    if (!f)
    {
        fprintf(stderr, "ERROR: cannot open weights file '%s'\n", weights_path);
        exit(1);
    }
    for (size_t l = 0; l < nn.arch_count - 1; l++)
    {
        Mat w = mat_load(NULL, f);
        mat_copy(nn.ws[l], w);
        Mat b = mat_load(NULL, f);
        mat_copy(nn.bs[l], b);
    }
    fclose(f);
}

static void center_and_normalize(Image *canvas_img, float *out_pixels, Image *processed_img)
{
    int min_x = CANVAS_SIZE, max_x = -1;
    int min_y = CANVAS_SIZE, max_y = -1;

    for (int y = 0; y < CANVAS_SIZE; y++)
    {
        for (int x = 0; x < CANVAS_SIZE; x++)
        {
            Color c = GetImageColor(*canvas_img, x, y);
            if (c.r > 16)
            {
                if (x < min_x)
                    min_x = x;
                if (x > max_x)
                    max_x = x;
                if (y < min_y)
                    min_y = y;
                if (y > max_y)
                    max_y = y;
            }
        }
    }

    for (int i = 0; i < CANVAS_SIZE * CANVAS_SIZE; i++)
        out_pixels[i] = 0.f;
    ImageClearBackground(processed_img, BLACK);

    if (max_x < 0)
        return;

    int content_w = max_x - min_x + 1;
    int content_h = max_y - min_y + 1;
    int target = 20;
    float scale = (float)target / (float)(content_w > content_h ? content_w : content_h);

    int scaled_w = (int)(content_w * scale + 0.5f);
    int scaled_h = (int)(content_h * scale + 0.5f);

    int off_x = (CANVAS_SIZE - scaled_w) / 2;
    int off_y = (CANVAS_SIZE - scaled_h) / 2;

    for (int y = 0; y < scaled_h; y++)
    {
        for (int x = 0; x < scaled_w; x++)
        {
            int src_x = min_x + (int)(x / scale);
            int src_y = min_y + (int)(y / scale);
            if (src_x >= CANVAS_SIZE)
                src_x = CANVAS_SIZE - 1;
            if (src_y >= CANVAS_SIZE)
                src_y = CANVAS_SIZE - 1;

            Color c = GetImageColor(*canvas_img, src_x, src_y);
            float val = (float)c.r / 255.f;

            int dst_x = x + off_x;
            int dst_y = y + off_y;
            if (dst_x >= 0 && dst_x < CANVAS_SIZE && dst_y >= 0 && dst_y < CANVAS_SIZE)
            {
                out_pixels[dst_y * CANVAS_SIZE + dst_x] = val;
                uint8_t pix = (uint8_t)(val * 255.f);
                ImageDrawPixel(processed_img, dst_x, dst_y, (Color){pix, pix, pix, 255});
            }
        }
    }
}

int main(int argc, char **argv)
{
    const char *program = args_shift(&argc, &argv);

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <model.arch> <model.arch.weights.mat> [labels.txt]\n", program);
        return 1;
    }

    const char *arch_path = args_shift(&argc, &argv);
    const char *weights_path = args_shift(&argc, &argv);
    const char *labels_path = (argc > 0) ? args_shift(&argc, &argv) : NULL;

    size_t arch_items[256];
    size_t arch_count = 0;
    FILE *af = fopen(arch_path, "r");
    if (!af)
        return 1;
    size_t layer_size;
    while (fscanf(af, "%zu", &layer_size) == 1)
    {
        if (arch_count < 256)
            arch_items[arch_count++] = layer_size;
    }
    fclose(af);

    if (arch_count < 2)
        return 1;
    size_t outs = arch_items[arch_count - 1];

    char labels[MAX_LABELS][LABEL_LEN];
    size_t label_count = 0;
    if (labels_path)
        label_count = load_labels(labels_path, labels);

    NN nn = nn_alloc(NULL, arch_items, arch_count);
    load_weights(nn, weights_path);
    printf("Loaded weights successfully.\n");

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(1200, 700, "Inference Canvas");
    SetTargetFPS(60);

    Image canvas_img = GenImageColor(CANVAS_SIZE, CANVAS_SIZE, BLACK);
    Texture2D canvas_tex = LoadTextureFromImage(canvas_img);

    Image processed_img = GenImageColor(CANVAS_SIZE, CANVAS_SIZE, BLACK);
    Texture2D processed_tex = LoadTextureFromImage(processed_img);

    float nn_inputs[CANVAS_SIZE * CANVAS_SIZE];

    while (!WindowShouldClose())
    {
        int W = GetRenderWidth();
        int H = GetRenderHeight();
        int col_w = W / 3;

        if (IsKeyPressed(KEY_C))
            ImageClearBackground(&canvas_img, BLACK);

        int ry = 48;
        int rh = H - ry;
        int canvas_draw_size = (col_w < rh ? col_w : rh) - 20;
        int canvas_x = (col_w - canvas_draw_size) / 2;
        int canvas_y = ry + (rh - canvas_draw_size) / 2;

        static Vector2 prev_mouse = {-1, -1};

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
        {
            Vector2 mouse = GetMousePosition();
            if (mouse.x >= canvas_x && mouse.x <= canvas_x + canvas_draw_size &&
                mouse.y >= canvas_y && mouse.y <= canvas_y + canvas_draw_size)
            {
                float cx = (mouse.x - canvas_x) / (float)canvas_draw_size * CANVAS_SIZE;
                float cy = (mouse.y - canvas_y) / (float)canvas_draw_size * CANVAS_SIZE;
                Color color = IsMouseButtonDown(MOUSE_BUTTON_LEFT) ? WHITE : BLACK;

                int icx = (int)cx, icy = (int)cy;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        ImageDrawPixel(&canvas_img, icx + dx, icy + dy, color);
            }
            prev_mouse = mouse;
        }
        else
        {
            prev_mouse = (Vector2){-1, -1};
        }

        UpdateTexture(canvas_tex, canvas_img.data);

        center_and_normalize(&canvas_img, nn_inputs, &processed_img);
        UpdateTexture(processed_tex, processed_img.data);

        for (int i = 0; i < CANVAS_SIZE * CANVAS_SIZE; i++)
            MAT_AT(NN_INPUT(nn), 0, i) = nn_inputs[i];

        nn_forward(nn);

        float probs[MAX_LABELS];
        size_t pred = nn_argmax(nn);
        softmax_copy(nn, probs, outs);

        BeginDrawing();
        ClearBackground((Color){0x18, 0x18, 0x18, 0xFF});

        int hdr_fs = (int)(H * 0.028f);
        if (hdr_fs < 10)
            hdr_fs = 10;
        DrawText("Left=Draw  Right=Erase  [C]=Clear  |  Draw large & centered for best results",
                 8, 10, hdr_fs, WHITE);

        DrawText("Your drawing", canvas_x, ry - hdr_fs - 2, hdr_fs, GRAY);
        DrawRectangle(canvas_x - 2, canvas_y - 2, canvas_draw_size + 4, canvas_draw_size + 4, DARKGRAY);
        DrawTexturePro(canvas_tex,
                       (Rectangle){0, 0, CANVAS_SIZE, CANVAS_SIZE},
                       (Rectangle){canvas_x, canvas_y, canvas_draw_size, canvas_draw_size},
                       (Vector2){0, 0}, 0.f, WHITE);

        int px = col_w + (col_w - canvas_draw_size) / 2;
        int py = canvas_y;
        DrawText("NN input (centered)", col_w + (col_w - canvas_draw_size) / 2, ry - hdr_fs - 2, hdr_fs, GRAY);
        DrawRectangle(px - 2, py - 2, canvas_draw_size + 4, canvas_draw_size + 4, DARKGRAY);
        DrawTexturePro(processed_tex,
                       (Rectangle){0, 0, CANVAS_SIZE, CANVAS_SIZE},
                       (Rectangle){px, py, canvas_draw_size, canvas_draw_size},
                       (Vector2){0, 0}, 0.f, WHITE);

        int rx_bars = col_w * 2;
        int bars_top = ry + 20;
        int bars_avail = rh - 40;
        int row_h = (outs > 0) ? (bars_avail / (int)outs) : 20;
        if (row_h < 10)
            row_h = 10;
        int bar_fs = (int)(row_h * 0.65f);
        if (bar_fs < 8)
            bar_fs = 8;
        if (bar_fs > 18)
            bar_fs = 18;

        int label_w = col_w * 2 / 7;
        int pct_w = 44;
        int bar_x = rx_bars + label_w;
        int bar_max_w = col_w - label_w - pct_w - 4;

        for (size_t cls = 0; cls < outs && cls < (size_t)MAX_LABELS; cls++)
        {
            int by = bars_top + (int)cls * row_h;
            char lbl[LABEL_LEN + 4];
            if (cls < label_count)
                snprintf(lbl, sizeof(lbl), "%s", labels[cls]);
            else
                snprintf(lbl, sizeof(lbl), "Cls %zu", cls);
            DrawText(lbl, rx_bars + 4, by + (row_h - bar_fs) / 2, bar_fs, LIGHTGRAY);

            int fill_w = (int)(probs[cls] * bar_max_w);
            if (fill_w < 0)
                fill_w = 0;

            Color bar_col = (cls == pred) ? GREEN : DARKGRAY;
            DrawRectangle(bar_x, by + 2, fill_w, row_h - 4, bar_col);
            DrawRectangleLines(bar_x, by + 2, bar_max_w, row_h - 4, (Color){0x40, 0x40, 0x40, 0xFF});

            char pct[12];
            snprintf(pct, sizeof(pct), "%.0f%%", probs[cls] * 100.f);
            DrawText(pct, bar_x + bar_max_w + 4, by + (row_h - bar_fs) / 2, bar_fs, WHITE);
        }

        EndDrawing();
    }

    return 0;
}
