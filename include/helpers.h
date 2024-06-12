#ifndef HELPERS_H
#define HELPERS_H 1

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

void gsl_vector_arange(gsl_vector *a, const double start, const double end, const double step=1.0);

void gsl_vector_linspace(gsl_vector *a, const double start, const double end,
                         const bool endpoint = true);

void print_vector(const gsl_vector *a, const int width = 0);
void print_vector(const gsl_vector_view &a, const int width = 0);

void print_matrix(const gsl_matrix *A, const int width = 0);
void print_matrix(const gsl_matrix_view &A, const int width = 0);

#endif
