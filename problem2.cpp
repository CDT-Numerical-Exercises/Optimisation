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

constexpr double f(const double x, const double y) {
  return std::log(x*x + y*(x + y));
}

// constant
double lambda_n(iter_t n, gsl_vector *x) {
  return 0.1;
}

void do_and_plot_descent(Gnuplot &gp, gsl_vector *x0) {
  std::vector<std::vector<double>> path;
  iter_t steps = gradient_descent(x0, gradf, lambda_n, &path, 1e-10, 10000);

  std::cout << "Gradient descent finished in " << steps << " steps." << std::endl;
  std::cout << "Found minima: (x,y) = (" << gsl_vector_get(x0, 0) << "," << gsl_vector_get(x0, 1) << ")" << std::endl;

  for (int i = 0; i < path.size(); ++i) {
    std::vector<double> X = path[i];
    gp << X[0] << " " << X[1] << "\n";
  }
  gp << "e\n";
}

void get_point_on_circle(double &x0, double &y0, const double r=1, const double cx=0, const double cy=0) {
  const double theta = randreal(0., 2 * M_PI);
  x0 = cx + r*std::cos(theta);
  y0 = cy + r*std::sin(theta);
}

void generate_contours() {
  constexpr double XMIN = -1;
  constexpr double XMAX = 1;
  constexpr double YMIN = -1;
  constexpr double YMAX = 1;
  constexpr double XSTEP = 0.05;
  constexpr double YSTEP = 0.05;
  constexpr int XPOINTS = (XMAX - XMIN)/XSTEP;
  constexpr int YPOINTS = (YMAX - YMIN)/YSTEP;
  Gnuplot gp;
  gp << "set dgrid3d " << XPOINTS << "," << YPOINTS << " gauss\n";
  gp << "set contour base\n";
  gp << "set cntrparam linear\n";
  gp << "set cntrparam levels 10\n";
  gp << "set table 'contours.dat'\n";
  gp << "splot '-' u 1:2:3 with lines nosurface\n";
  for (double y = YMIN; y <= YMAX; y += YSTEP) {
    for (double x = XMIN; x <= XMAX; x += XSTEP) {
      const double z = f(x, y);
      gp << x << " " << y << " " << z << "\n";
    }
  }
  gp << "e\n";
}

constexpr int N_DESCENTS = 4;


double X[2];

int main() {
  compute_A_sum();
  helpers_rng::generator.seed(0xC0FFEE);
  
  Gnuplot gp;
  generate_contours();
  gp << "set key off\n";
  gp << "plot for [level=1:*] 'contours.dat' index level with lines, ";
  for (int i = 1; i < N_DESCENTS; ++i) {
    gp << "'-' with linespoints, ";
  }
  gp << "'-' with linespoints\n ";
  
  gsl_vector_view x0_view = gsl_vector_view_array(X, 2);
  gsl_vector *x0 = &x0_view.vector;

  
  
  gsl_vector_set_all(x0, 1);
  do_and_plot_descent(gp, x0);
  
  for (int i = 1; i < N_DESCENTS; ++i) {
    get_point_on_circle(X[0], X[1]);
    do_and_plot_descent(gp, x0);
  }
}
