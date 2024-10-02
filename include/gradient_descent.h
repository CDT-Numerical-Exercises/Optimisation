#ifndef GRAD_DESC_H
#define GRAD_DESC_H 1

#include <vector>
#include <gsl/gsl_vector.h>

// define function types for the gradient descent input
typedef int iter_t;
typedef void (gradf_function)(const gsl_vector *x, gsl_vector *grad);
typedef double (learning_rate_function)(iter_t n, gsl_vector *x);

iter_t gradient_descent(gsl_vector *x0, gradf_function gradf, learning_rate_function lambda_n, std::vector<std::vector<double>> *path, double precision, iter_t maxsteps);

#endif
