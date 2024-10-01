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

void gradf(gsl_vector *x, gsl_vector *grad) {
  // use y = x**2 + 1
  // gradient is just 2x
  gsl_vector_set(grad, 0, 2*gsl_vector_get(x, 0));
}

// constant
double lambda_n(iter_t n) {
  return 0.1;
}

int main() {
  gsl_vector *x0 = gsl_vector_alloc(1);
  gsl_vector_set_all(x0, 10);

  std::vector<std::vector<double>> path;

  iter_t steps = gradient_descent(x0, gradf, lambda_n, &path, 1e-10, 10000);

  std::cout << "Gradient descent finished in " << steps << " steps." << std::endl;

  std::cout << "Found minima: x = " << gsl_vector_get(x0, 0) << std::endl;

  Gnuplot gp;
  gp << "plot '-' with linespoints\n";
  for (int i = 0; i < path.size(); ++i) {
    double x = path[i][0];
    gp << x << " " << x*x + 1 << "\n";
  }

  gsl_vector_free(x0);
}
