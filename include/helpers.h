#ifndef HELPERS_H
#define HELPERS_H 1

#include <cinttypes>
#include <random>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

void gsl_vector_arange(gsl_vector *a, const double start, const double end, const double step=1.0);

void gsl_vector_linspace(gsl_vector *a, const double start, const double end,
                         const bool endpoint = true);

void print_vector(const gsl_vector *a, const int width = 0);
void print_vector(const gsl_vector_view &a, const int width = 0);

void print_matrix(const gsl_matrix *A, const int width = 0);
void print_matrix(const gsl_matrix_view &A, const int width = 0);


extern std::default_random_engine generator;
template <typename T> T randint(const T min, const T max) {
  std::uniform_int_distribution<T> distribution(min,max);
  return distribution(generator);
}

template int randint<int>(const int, const int);
template long randint<long>(const long, const long);
template uint32_t randint<uint32_t>(const uint32_t, const uint32_t);
template uint64_t randint<uint64_t>(const uint64_t, const uint64_t);

#endif
