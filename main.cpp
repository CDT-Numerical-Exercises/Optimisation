#include <iostream>
#include <gsl/gsl_vector.h>

// use bounds checking to make sure this doesn't go badly.
// GSL's bounds checking should be good.
void gsl_vector_arange(gsl_vector *a, double start, double end, double step=1.0) {
  double v = start;
  size_t i = 0;
  while (v <= end) {
    gsl_vector_set(a, i, v);
    v += step;
    ++i;
  }
}

void gsl_vector_linspace(gsl_vector *a, double start, double end,
                         bool endpoint = true) {
  size_t n = a->size;
  double stepsize;
  if (endpoint) stepsize = (end-start)/(n-1); else stepsize = (end-start)/n;

  double v = start;
  for (size_t i = 0; i < n; i++) {
    gsl_vector_set(a, i, v);
    v += stepsize;
  }
}

int main() {
  constexpr double stepsize = 0.05;
  constexpr double start = 1.0;
  constexpr double end = 2.0;
  constexpr size_t required_size = (end-start)/stepsize + 1;

  /*
    linspace equivalent
  */
  
  gsl_vector *a = gsl_vector_calloc(required_size);
  gsl_vector_linspace(a, start, end);

  std::cout << "[ ";
  for (int i = 0; i < a->size; ++i) {
    std::cout << gsl_vector_get(a, i) << " ";
  }
  std::cout << "]" << std::endl;
  gsl_vector_free(a);

  /*
    arange equivalent
  */

  a = gsl_vector_calloc(required_size);
  gsl_vector_arange(a, start, end+stepsize, stepsize);

  std::cout << "[ ";
  for (int i = 0; i < a->size; ++i) {
    std::cout << gsl_vector_get(a, i) << " ";
  }
  std::cout << "]" << std::endl;
  gsl_vector_free(a);

  
  return 0;
}
