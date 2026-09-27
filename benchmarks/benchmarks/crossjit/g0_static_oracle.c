/* Benchmark-only static-C feasibility oracle for CrossJIT Generator G0. */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__GNUC__) || defined(__clang__)
#  define NOINLINE __attribute__((noinline))
#else
#  define NOINLINE
#endif

typedef void (*oracle_fn)(const double *const *, double *, size_t);

NOINLINE void
g0_oracle_linear_2(const double *const *inputs, double *restrict output,
                   size_t size)
{
    const double *a = inputs[0];
    const double *b = inputs[1];
    const double *c = inputs[2];
    for (size_t i = 0; i < size; i++) {
        double t0 = a[i] + b[i];
        double t1 = t0 * c[i];
        output[i] = t1;
    }
}

NOINLINE void
g0_oracle_linear_4(const double *const *inputs, double *restrict output,
                   size_t size)
{
    const double *a = inputs[0];
    const double *b = inputs[1];
    const double *c = inputs[2];
    const double *d = inputs[3];
    const double *e = inputs[4];
    for (size_t i = 0; i < size; i++) {
        double t0 = a[i] + b[i];
        double t1 = t0 * c[i];
        double t2 = t1 - d[i];
        double t3 = t2 / e[i];
        output[i] = t3;
    }
}

NOINLINE void
g0_oracle_linear_6(const double *const *inputs, double *restrict output,
                   size_t size)
{
    const double *a = inputs[0];
    const double *b = inputs[1];
    const double *c = inputs[2];
    const double *d = inputs[3];
    const double *e = inputs[4];
    const double *f = inputs[5];
    const double *g = inputs[6];
    for (size_t i = 0; i < size; i++) {
        double t0 = a[i] + b[i];
        double t1 = t0 * c[i];
        double t2 = t1 - d[i];
        double t3 = t2 / e[i];
        double t4 = t3 + f[i];
        double t5 = t4 * g[i];
        output[i] = t5;
    }
}

/* Volatile temporaries make this an independent scalar operation-order oracle. */
NOINLINE static void
reference(const double *const *inputs, double *output, size_t size,
          int region_length)
{
    for (size_t i = 0; i < size; i++) {
        volatile double t0 = inputs[0][i] + inputs[1][i];
        volatile double t1 = t0 * inputs[2][i];
        if (region_length == 2) {
            output[i] = t1;
            continue;
        }
        volatile double t2 = t1 - inputs[3][i];
        volatile double t3 = t2 / inputs[4][i];
        if (region_length == 4) {
            output[i] = t3;
            continue;
        }
        volatile double t4 = t3 + inputs[5][i];
        volatile double t5 = t4 * inputs[6][i];
        output[i] = t5;
    }
}

static uint64_t
bits(double value)
{
    uint64_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static int
same_value(double left, double right)
{
    if (isnan(left) && isnan(right)) {
        return 1;
    }
    return bits(left) == bits(right);
}

static uint64_t
clock_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
           (uint64_t)now.tv_nsec;
}

static double
median(double *values, int count)
{
    for (int i = 1; i < count; i++) {
        double value = values[i];
        int j = i;
        while (j > 0 && values[j - 1] > value) {
            values[j] = values[j - 1];
            j--;
        }
        values[j] = value;
    }
    return values[count / 2];
}

static int
validate(oracle_fn function, const double *const *inputs, double *output,
         double *expected, size_t size, int region_length)
{
    feclearexcept(FE_ALL_EXCEPT);
    function(inputs, output, size);
    int actual_fpe = fetestexcept(
        FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);
    feclearexcept(FE_ALL_EXCEPT);
    reference(inputs, expected, size, region_length);
    int expected_fpe = fetestexcept(
        FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);
    if (actual_fpe != expected_fpe) {
        fprintf(stderr, "FPE mismatch: actual=%d expected=%d\n",
                actual_fpe, expected_fpe);
        return 0;
    }
    for (size_t i = 0; i < size; i++) {
        if (!same_value(output[i], expected[i])) {
            fprintf(stderr, "value mismatch at %zu: 0x%016llx != 0x%016llx\n",
                    i, (unsigned long long)bits(output[i]),
                    (unsigned long long)bits(expected[i]));
            return 0;
        }
    }
    return 1;
}

int
main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr, "usage: %s CASE N SAMPLES TARGET_ELEMENTS ITERATION_CAP\n",
                argv[0]);
        return 2;
    }
    const char *case_id = argv[1];
    size_t size = (size_t)strtoull(argv[2], NULL, 10);
    int samples = atoi(argv[3]);
    size_t target_elements = (size_t)strtoull(argv[4], NULL, 10);
    size_t iteration_cap = (size_t)strtoull(argv[5], NULL, 10);
    int region_length;
    size_t input_count;
    oracle_fn function;
    if (strcmp(case_id, "linear_2") == 0) {
        region_length = 2;
        input_count = 3;
        function = g0_oracle_linear_2;
    }
    else if (strcmp(case_id, "linear_4") == 0) {
        region_length = 4;
        input_count = 5;
        function = g0_oracle_linear_4;
    }
    else if (strcmp(case_id, "linear_6") == 0) {
        region_length = 6;
        input_count = 7;
        function = g0_oracle_linear_6;
    }
    else {
        fprintf(stderr, "unsupported case: %s\n", case_id);
        return 2;
    }
    if (size == 0 || samples < 1) {
        return 2;
    }

    double *storage[7] = {0};
    const double *inputs[7] = {0};
    for (size_t input = 0; input < input_count; input++) {
        if (posix_memalign((void **)&storage[input], 64,
                           size * sizeof(double)) != 0) {
            fprintf(stderr, "allocation failed\n");
            return 2;
        }
        inputs[input] = storage[input];
        for (size_t i = 0; i < size; i++) {
            storage[input][i] = 1.0 + (double)i / (double)size +
                                (double)input * 0.125;
        }
    }
    double *output = NULL;
    double *expected = NULL;
    if (posix_memalign((void **)&output, 64, size * sizeof(double)) != 0 ||
        posix_memalign((void **)&expected, 64, size * sizeof(double)) != 0) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }
    if (!validate(function, inputs, output, expected, size, region_length)) {
        return 3;
    }
    static const double special[] = {
        NAN, INFINITY, -INFINITY, -0.0, 0.0, 1.0, -1.0, 2.0
    };
    for (size_t input = 0; input < input_count; input++) {
        for (size_t i = 0; i < sizeof(special) / sizeof(special[0]); i++) {
            storage[input][i] = special[(i + input) %
                (sizeof(special) / sizeof(special[0]))];
        }
    }
    if (!validate(function, inputs, output, expected,
                  sizeof(special) / sizeof(special[0]), region_length)) {
        return 3;
    }
    for (size_t input = 0; input < input_count; input++) {
        for (size_t i = 0; i < size; i++) {
            storage[input][i] = 1.0 + (double)i / (double)size +
                                (double)input * 0.125;
        }
    }

    size_t iterations = target_elements / size;
    if (iterations < 8) {
        iterations = 8;
    }
    if (iterations > iteration_cap) {
        iterations = iteration_cap;
    }
    for (int warmup = 0; warmup < 32; warmup++) {
        function(inputs, output, size);
    }
    double *sample_ns = calloc((size_t)samples, sizeof(double));
    if (sample_ns == NULL) {
        return 2;
    }
    for (int sample = 0; sample < samples; sample++) {
        uint64_t started = clock_ns();
        for (size_t iteration = 0; iteration < iterations; iteration++) {
            function(inputs, output, size);
        }
        uint64_t elapsed = clock_ns() - started;
        sample_ns[sample] = (double)elapsed / (double)iterations;
    }
    double process_median = median(sample_ns, samples);
    volatile double checksum = output[size / 2];
    printf("{\"case_identifier\":\"%s\",\"region_length\":%d,"
           "\"element_count\":%zu,\"iterations_per_sample\":%zu,"
           "\"sample_count\":%d,\"process_median_ns\":%.17g,"
           "\"correct\":true,\"fpe_parity\":true,\"checksum\":%.17g}\n",
           case_id, region_length, size, iterations, samples,
           process_median, checksum);

    free(sample_ns);
    free(expected);
    free(output);
    for (size_t input = 0; input < input_count; input++) {
        free(storage[input]);
    }
    return 0;
}
