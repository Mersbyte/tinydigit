/*
 * tinydigit: classify a 28x28 grayscale PGM image of a handwritten digit
 * with a 784-64-10 ReLU network whose weights live in weights.h.
 *
 * C99 and the standard library only. The code itself never calls malloc:
 * the weights are const arrays, the input buffer is static and the
 * activations are on the stack. stdio may allocate its own buffers.
 */
#include <math.h>
#include <stdio.h>

#include "weights.h"

#define SIDE 28

#if N_IN != SIDE * SIDE
#error "weights.h does not match a 28x28 input"
#endif

/* PGM whitespace is only these four; isspace() would also accept \v and \f. */
static int is_pgm_space(int c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

/* If c starts a '#' comment, skip to the end of the line and return the '\n'
 * (or EOF) that ends it, so the newline acts as the separator. */
static int skip_comment(FILE *f, int c)
{
    if (c == '#')
        while (c != '\n' && c != EOF)
            c = getc(f);
    return c;
}

/*
 * Read a decimal number after optional whitespace and comments, then consume
 * exactly one separator: a whitespace character, a comment ending in '\n',
 * or EOF. Stores the separator in *sep. -1 on error. For P5 this means the
 * raster starts right after the separator that follows maxval.
 */
static long read_field(FILE *f, int *sep)
{
    long v = 0;
    int c, digits = 0;

    do
        c = skip_comment(f, getc(f));
    while (is_pgm_space(c));
    while (c >= '0' && c <= '9' && v < 100000) {
        v = v * 10 + (c - '0');
        digits++;
        c = getc(f);
    }
    *sep = c = skip_comment(f, c);
    if (digits == 0 || !(is_pgm_space(c) || c == EOF))
        return -1;
    return v;
}

/* Parse a P5 (binary) or P2 (ASCII) PGM into x, scaled to [0, 1]. */
static const char *parse_pgm(FILE *f, float x[N_IN])
{
    int type, c, sep;
    long width, height, maxval;

    if (getc(f) != 'P')
        return "not a PGM file";
    type = getc(f);
    if (type != '5' && type != '2')
        return "only P5 and P2 PGM files are supported";
    if (!is_pgm_space(skip_comment(f, getc(f))))
        return "bad PGM header";
    width = read_field(f, &sep);
    height = read_field(f, &sep);
    maxval = read_field(f, &sep);
    if (width < 0 || height < 0 || maxval < 0)
        return "bad PGM header";
    if (width != SIDE || height != SIDE)
        return "image must be 28x28";
    if (maxval < 1 || maxval > 255)
        return "maxval must be between 1 and 255";
    if (type == '5' && sep == '\r') {
        c = getc(f);
        if (c == '\n')
            return "PGM header uses CRLF line endings";
        ungetc(c, f);
    }

    for (int i = 0; i < N_IN; i++) {
        long v = type == '5' ? getc(f) : read_field(f, &sep);
        if (v < 0 || v > maxval)
            return "truncated or invalid pixel data";
        x[i] = (float)v / (float)maxval;
    }

    /* Anything after the pixels means the header or raster was misread. */
    if (type == '2')
        do
            c = skip_comment(f, getc(f));
        while (is_pgm_space(c));
    else
        c = getc(f);
    if (c != EOF)
        return "extra data after the image";
    return NULL;
}

static const char *load_pgm(const char *path, float x[N_IN])
{
    FILE *f = fopen(path, "rb");
    const char *err;

    if (!f)
        return "cannot open file";
    err = parse_pgm(f, x);
    fclose(f);
    return err;
}

/* Forward pass. Fills prob with softmax probabilities, returns the argmax. */
static int predict(const float x[N_IN], float prob[N_OUT])
{
    float hidden[N_HIDDEN];
    float max, sum = 0.0f;
    int best = 0;

    for (int i = 0; i < N_HIDDEN; i++) {
        float acc = b1[i];
        for (int j = 0; j < N_IN; j++)
            acc += W1[i][j] * x[j];
        hidden[i] = acc > 0.0f ? acc : 0.0f;
    }
    for (int i = 0; i < N_OUT; i++) {
        float acc = b2[i];
        for (int j = 0; j < N_HIDDEN; j++)
            acc += W2[i][j] * hidden[j];
        prob[i] = acc;
        if (acc > prob[best])
            best = i;
    }

    max = prob[best];
    for (int i = 0; i < N_OUT; i++) {
        prob[i] = expf(prob[i] - max);
        sum += prob[i];
    }
    for (int i = 0; i < N_OUT; i++)
        prob[i] /= sum;
    return best;
}

int main(int argc, char **argv)
{
    static float x[N_IN];
    const char *prog = argc > 0 && argv[0] ? argv[0] : "tinydigit";
    float prob[N_OUT];
    int status = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s image.pgm [image.pgm ...]\n", prog);
        return 2;
    }
    for (int i = 1; i < argc; i++) {
        const char *err = load_pgm(argv[i], x);
        int digit;

        if (err) {
            fprintf(stderr, "%s: %s\n", argv[i], err);
            status = 1;
            continue;
        }
        digit = predict(x, prob);
        if (argc > 2)
            printf("%s: ", argv[i]);
        printf("%d (%.1f%%)\n", digit, 100.0 * prob[digit]);
    }
    if (fflush(stdout) != 0 || ferror(stdout)) {
        fprintf(stderr, "%s: write error\n", prog);
        return 1;
    }
    return status;
}
