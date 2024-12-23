#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
    __pp_vec_float x;
    __pp_vec_float result;
    __pp_vec_float zero = _pp_vset_float(0.f);
    __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i = 0; i < N; i += VECTOR_WIDTH) {

        // All ones
        maskAll = _pp_init_ones();

        // All zeros
        maskIsNegative = _pp_init_ones(0);

        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskAll); // x = values[i];

        // Set mask according to predicate
        _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

        // Execute instruction using mask ("if" clause)
        _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

        // Inverse maskIsNegative to generate "else" mask
        maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

        // Execute instruction ("else" clause)
        _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

        // Write results back to memory
        _pp_vstore_float(output + i, result, maskAll);
    }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
    //
    // PP STUDENTS TODO: Implement your vectorized version of
    // clampedExpSerial() here.
    //
    // Your solution should work for any value of
    // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
    //

    __pp_vec_float x, result, clamped_val = _pp_vset_float(9.999999f);
    __pp_vec_int y, zero = _pp_vset_int(0), one = _pp_vset_int(1);
    __pp_mask maskAll, maskAll_copy, maskOne;

    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        // init masks
        maskOne = _pp_init_ones(VECTOR_WIDTH);
        maskAll = _pp_init_ones((N - i) < VECTOR_WIDTH ? (N - i) : VECTOR_WIDTH);
        maskAll_copy = _pp_mask_and(maskAll, maskAll);

        // load
        _pp_vload_float(x, values + i, maskAll_copy);
        _pp_vload_int(y, exponents + i, maskAll_copy);
        _pp_vset_float(result, 0.f, maskOne);
        _pp_vset_float(result, 1.f, maskAll_copy);

        // exponentiation
        _pp_vgt_int(maskAll_copy, y, zero, maskAll_copy);
        while (_pp_cntbits(maskAll_copy)) {
            _pp_vmult_float(result, result, x, maskAll_copy);
            _pp_vsub_int(y, y, one, maskOne);
            _pp_vgt_int(maskAll_copy, y, zero, maskAll_copy);
        }

        // clamping
        _pp_vgt_float(maskAll_copy, result, clamped_val, maskOne);
        _pp_vmove_float(result, clamped_val, maskAll_copy);
        maskAll_copy = _pp_mask_or(maskAll, maskAll_copy);

        // store
        _pp_vstore_float(output + i, result, maskAll_copy);
    }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

    //
    // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
    //

    __pp_vec_float x, result = _pp_vset_float(0.f);
    __pp_mask maskAll;

    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        maskAll = _pp_init_ones(VECTOR_WIDTH);
        _pp_vload_float(x, values + i, maskAll);

        for (int j = 1; j < VECTOR_WIDTH; j <<= 1) {
            _pp_hadd_float(x, x);
            _pp_interleave_float(x, x);
        }

        result.value[0] += x.value[0];
    }

    return result.value[0];
}