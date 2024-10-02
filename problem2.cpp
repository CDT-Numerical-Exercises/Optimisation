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
  1, 1,
  0, 1
};
const gsl_matrix_const_view A_view = gsl_matrix_const_view_array(A_arr, 2, 2);
// we will pre-compute this to save time later
double A_sum_arr[4];
gsl_matrix_view A_sum_view = gsl_matrix_view_array(A_sum_arr, 2, 2);

void compute_A_sum() {
  gsl_matrix_transpose_memcpy(&A_sum_view.matrix, &A_view.matrix);
  gsl_matrix_add(&A_sum_view.matrix, &A_view.matrix);
}

void gradf(const gsl_vector *x, gsl_vector *grad) {
  double denominator;
  double Ax[2];
  gsl_vector_view Ax_view = gsl_vector_view_array(Ax, 2);
  gsl_blas_dgemv(CblasNoTrans, 1, &A_view.matrix, x, 0, &Ax_view.vector);
  gsl_blas_ddot(x, &Ax_view.vector, &denominator);
  denominator += 1;

  gsl_blas_dgemv(CblasNoTrans, 1.0/denominator, &A_sum_view.matrix, x, 0, grad);
}

// constant
double lambda_n(iter_t n, gsl_vector *x) {
  return 0.1;
}

int main() {
  compute_A_sum();
  
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
