#ifndef LINESEARCH_H
#define LINESEARCH_H 1

#include <gsl/gsl_vector.h>
#include "gradient_descent.h"

typedef double (function)(const gsl_vector *x);

double backtracking_linesearch(const gsl_vector *x, function f, gradf_function gradf, const double alpha0, const double tau, const double c);

#endif
