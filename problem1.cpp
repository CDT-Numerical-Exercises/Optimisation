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

// these are constant, so we can write them here as arrays and use
// views to access them in the GSL way. This way we won't have to
// worry about allocating and freeing at the correct times.
const double A_arr[4] = {
  2, -1,
  -1, 3
};
const double b_arr[2] = {0, 1};

const gsl_matrix_const_view A_view = gsl_matrix_const_view_array(A_arr, 2, 2);
const gsl_vector_const_view b_view = gsl_vector_const_view_array(b_arr, 2);

void gradf(gsl_vector *x, gsl_vector *grad) {
  gsl_blas_dgemv(CblasNoTrans, 1, &A_view.matrix, x, 0, grad);
  gsl_blas_daxpy(-1, &b_view.vector, grad);
}

// special case
double lambda_n(iter_t n, gsl_vector *x) {
  // stack allocate the grad as we know it will be 2x1
  double grad[2];
  gsl_vector_view grad_vec_view = gsl_vector_view_array(grad, 2);

  // compute gradient
  gradf(x, &grad_vec_view.vector);

  // compute numerator
  // V.T @ V === V . V === |V|^2
  double numerator;
  gsl_blas_ddot(&grad_vec_view.vector, &grad_vec_view.vector, &numerator);

  double denominator[3]; // index 0 will hold the true denominator, indices 1-2 will hold the intermediate result
  gsl_vector_view den_inter = gsl_vector_view_array(denominator+1, 2);

  // compute denominator
  gsl_blas_dgemv(CblasNoTrans, 1, &A_view.matrix, &grad_vec_view.vector, 0, &den_inter.vector);
  gsl_blas_ddot(&grad_vec_view.vector, &den_inter.vector, denominator);

  return numerator / *denominator;
}

int main() {
  gsl_vector *x0 = gsl_vector_alloc(2);
  gsl_vector_set_all(x0, 1);

  std::vector<std::vector<double>> path;

  iter_t steps = gradient_descent(x0, gradf, lambda_n, &path, 1e-10, 10000);

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
