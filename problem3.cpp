#include <iostream>
#include <cstdlib>
#define _USE_MATH_DEFINES
#include <cmath>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gnuplot-iostream/gnuplot-iostream.h>
#include "helpers.h"

#include "gradient_descent.h"
#include "linesearch.h"

constexpr double alpha0 = 0.5;
constexpr double tau = 0.75;
constexpr double c = 0.9;

double f(const gsl_vector *X) {
  const double x = gsl_vector_get(X, 0);
  const double y = gsl_vector_get(X, 1);
  
  return sin(x+y) + (x - y)*(x - y) - 1.5*x + 2.5*y + 1;
}

void gradf(const gsl_vector *X, gsl_vector *grad) {
  // dx = cos(x+y) + 2x - 2y - 1.5
  // dy = cos(x+y) + 2y - 2x + 2.5
  const double x = gsl_vector_get(X, 0);
  const double y = gsl_vector_get(X, 1);
  const double cos_term = cos(x+y);

  gsl_vector_set(grad, 0, cos_term + 2*x - 2*y - 1.5);
  gsl_vector_set(grad, 1, cos_term - 2*x + 2*y + 2.5);
}

double lambda_n(iter_t n, gsl_vector *x) {
  return backtracking_linesearch(x, f, gradf, alpha0, tau, c);
  // return 0.2;
}

int main() {
  gsl_vector *x0 = gsl_vector_alloc(2);
  gsl_vector_set_all(x0, 0);

  std::vector<std::vector<double>> path;

  iter_t steps = gradient_descent(x0, gradf, lambda_n, &path, 1e-6, 100000);

  std::cout << "Gradient descent finished in " << steps << " steps." << std::endl;

  std::cout << "Found minima: (x,y) = (" << gsl_vector_get(x0, 0) << "," << gsl_vector_get(x0, 1) << ")" << std::endl;

  Gnuplot gp;
  gp << "plot '-' with linespoints\n";
  for (int i = 0; i < path.size(); ++i) {
    std::vector<double> X = path[i];
    gp << X[0] << " " << X[1] << "\n";
  }

  gsl_vector_free(x0);
}
