#include <iostream>
#include <cstdlib>
#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gnuplot-iostream/gnuplot-iostream.h>
#include "helpers.h"

#include "gradient_descent.h"
#include "linesearch.h"

// precision to print the output to
constexpr int output_precision = 10;
// calculate to at least one decimal place more precision than we output
const double calculation_precision = pow(10, -(output_precision+2));

constexpr double alpha0 = 0.5;
constexpr double tau = 0.5;
constexpr double c = 1e-3;

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

// setup for sweeping the range in order to find the global minimum
constexpr double X_RANGE[2] = {-1, 4};
constexpr double Y_RANGE[2] = {-3, 3};
constexpr int TRIALS_X = 6;
constexpr int TRIALS_Y = 7;
constexpr int TOTAL_TRIALS = TRIALS_X * TRIALS_Y;
constexpr double STEPSIZE_X = (X_RANGE[1]-X_RANGE[0])/(TRIALS_X-1);
constexpr double STEPSIZE_Y = (Y_RANGE[1]-Y_RANGE[0])/(TRIALS_Y-1);

// cherry picked to produce a good plot
void generate_contours() {
  // constexpr double XMIN = X_RANGE[0];
  constexpr double XMIN = -2;
  constexpr double XMAX = X_RANGE[1];
  constexpr double YMIN = Y_RANGE[0];
  constexpr double YMAX = Y_RANGE[1];
  constexpr double XSTEP = 0.05;
  constexpr double YSTEP = 0.05;
  constexpr int XPOINTS = (XMAX - XMIN)/XSTEP;
  constexpr int YPOINTS = (YMAX - YMIN)/YSTEP;
  Gnuplot gp;
  gp << "set dgrid3d " << XPOINTS << "," << YPOINTS << " gauss\n";
  gp << "set contour base\n";
  gp << "set cntrparam linear\n";
  gp << "set cntrparam levels incremental -1.9, 1.5, 25\n";
  gp << "set table 'contours.dat'\n";
  gp << "splot '-' u 1:2:3 with lines nosurface\n";

  double X[2];
  gsl_vector_view x0_view = gsl_vector_view_array(X, 2);
  gsl_vector *x0 = &x0_view.vector;

  double &x = X[0];
  double &y = X[1];
  for (y = YMIN; y <= YMAX; y += YSTEP) {
    for (x = XMIN; x <= XMAX; x += XSTEP) {
      const double z = f(x0);
      // print_vector(x0);
      gp << x << " " << y << " " << z << "\n";
    }
  }
  gp << "e\n";
}

int main() {
  double x0_arr[2] = {X_RANGE[0], Y_RANGE[0]};
  double &X = x0_arr[0];
  double &Y = x0_arr[1];
  gsl_vector_view x0_view = gsl_vector_view_array(x0_arr, 2);
  gsl_vector *x0 = &x0_view.vector;

  std::vector<std::vector<double>> path;
  std::vector<std::vector<double>> best_path;
  double best_minimum = std::numeric_limits<double>::infinity();
  double best_minimum_coords[2] =
  { std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::quiet_NaN() };

  std::cout << "Searching " << X_RANGE[0] << " < x < " << X_RANGE[1] << ", " << Y_RANGE[0] << " < y < " << Y_RANGE[1] << " using " << TOTAL_TRIALS << " trials for the global minimum" << std::endl;
  for (int y = 0; y < TRIALS_Y; ++y) {
    for (int x = 0; x < TRIALS_X; ++x) {
      // x0 is modified by the gradient descent algorithm
      // we need to reset it directly at the start of each loop
      X = X_RANGE[0] + x*STEPSIZE_X;
      Y = Y_RANGE[0] + y*STEPSIZE_Y;
      // std::cout << "Starting at " << X << "," << Y << std::endl;

      // ensure the path is empty
      path.clear();

      iter_t steps = gradient_descent(x0, gradf, lambda_n, &path, calculation_precision, 100);

      // std::cout << "Gradient descent finished in " << steps << " steps."
      //           << std::endl;

      // evaluate the function for the found minima
      const double function_val = f(x0);
      if (function_val < best_minimum) {
        // only count it if it's inside the range
        if (X < X_RANGE[1] && X > X_RANGE[0] && Y < Y_RANGE[1] &&
            Y > Y_RANGE[0]) {
          best_minimum = function_val;
          best_minimum_coords[0] = X;
          best_minimum_coords[1] = Y;
          best_path = path;
          std::cout << "Found new best minima: f(x,y) = (" << X << ","
                    << Y << ") = " << function_val << std::endl;
        }
      }
    }
  }

  const auto default_precision{std::cout.precision()};
  std::cout << std::setprecision(output_precision);
  std::cout << "Best minima: f(" << best_minimum_coords[0] << ","
            << best_minimum_coords[1] << ") = " << best_minimum << std::endl;
  std::cout << std::setprecision(default_precision);

  generate_contours();
  Gnuplot gp;
  // gp << "set xrange [" << X_RANGE[0] << ":" << X_RANGE[1] << "]\n";
  // gp << "set yrange [" << Y_RANGE[0] << ":" << Y_RANGE[1] << "]\n";
  gp << "set key off\n";
  gp << "plot for [level=1:*] 'contours.dat' index level with lines, '-' with linespoints\n";
  for (int i = 0; i < best_path.size(); ++i) {
    std::vector<double> x = best_path[i];
    gp << x[0] << " " << x[1] << "\n";
  }
  gp << "e\n";
}
