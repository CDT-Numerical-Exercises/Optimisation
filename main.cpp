#include <iostream>
#include <gsl/gsl_vector.h>

int main() {
  gsl_vector *a = gsl_vector_calloc(3);
  for (int i = 0; i < a->size; ++i) {
    gsl_vector_set(a, i, i+1);
  }

  std::cout << "[ ";
  for (int i = 0; i < a->size; ++i) {
    std::cout << gsl_vector_get(a, i) << " ";
  }
  std::cout << "]" << std::endl;
  return 0;
}
