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
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

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
  __pp_vec_float x;
  __pp_vec_int y;
  __pp_vec_float result;
  __pp_vec_float upperBound = _pp_vset_float(9.999999f);
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_mask maskAll, maskIsZero, maskIsNotZero;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    int maskSize = std::min(N - i, VECTOR_WIDTH); // to deal with the situation where N % VECTOR_WIDTH != 0

    // All ones except for the index >= maskSize
    maskAll = _pp_init_ones(maskSize);
    
    // All zeros
    maskIsZero = _pp_init_ones(0);
    
    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i]

    _pp_vload_int(y, exponents + i, maskAll); // y = exponents[i]

    _pp_vset_float(result, 1.f, maskAll); // result = 1.

    // Set mask according to predicate
    _pp_veq_int(maskIsZero, y, zero, maskAll); // if (y == 0) {
    
    // Execute instruction using mask ("if" clause)
    // no operation

    // Inverse maskIsZero to generate "else" mask
    maskIsNotZero = _pp_mask_not(maskIsZero); // } else {

    // Execute instruction using mask ("else" clause)
    while (_pp_cntbits(maskIsZero) < maskSize) {
      _pp_vmult_float(result, result, x, maskIsNotZero); // result *= x;
      _pp_vsub_int(y, y, one, maskIsNotZero); // count--;
      // update maskIsZero & maskIsNotZero
      _pp_veq_int(maskIsZero, y, zero, maskAll);  
      maskIsNotZero = _pp_mask_not(maskIsZero); 
    }

    __pp_mask maskClamp;
    _pp_vgt_float(maskClamp, result, upperBound, maskAll); // if (result > 9.999999f) {
    _pp_vmove_float(result, upperBound, maskClamp); // result = 9.999999f; }

    // Write result back to memory
    _pp_vstore_float(output + i, result, maskAll);
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
  // All ones
  __pp_mask maskAll = _pp_init_ones();

  __pp_vec_float result;
  int size;
  float sum = 0.0;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    size = VECTOR_WIDTH;
    _pp_vload_float(result, values + i, maskAll);
    
    while (size > 1) {
      // [0 1 2 3] -> [0+1 0+1 2+3 2+3]
      _pp_hadd_float(result, result);
      // [0+1 0+1 2+3 2+3] -> [0+1 2+3 0+1 2+3]
      _pp_interleave_float(result, result);

      size /= 2; // get the former half of the array
    }

    sum += result.value[0];
  }

  return sum;
}