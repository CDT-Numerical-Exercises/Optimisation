#include <iostream>
#include <cstdlib>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gnuplot-iostream/gnuplot-iostream.h>

// use bounds checking to make sure this doesn't go badly.
// GSL's bounds checking should be good.
void gsl_vector_arange(gsl_vector *a, double start, double end, double step=1.0) {
  double v = start;
  size_t i = 0;
  while (v <= end) {
    gsl_vector_set(a, i, v);
    v += step;
    ++i;
  }
}

void gsl_vector_linspace(gsl_vector *a, double start, double end,
                         bool endpoint = true) {
  size_t n = a->size;
  double stepsize;
  if (endpoint) stepsize = (end-start)/(n-1); else stepsize = (end-start)/n;

  double v = start;
  for (size_t i = 0; i < n; i++) {
    gsl_vector_set(a, i, v);
    v += stepsize;
  }
}

void print_vector(const gsl_vector *a, const int width = 0) {
  std::cout << "[ ";
  for (int i = 0; i < a->size; ++i) {
    std::cout << std::setw(width) << gsl_vector_get(a, i) << " ";
  }
  std::cout << "]" << std::endl;
}

void print_matrix(const gsl_matrix *A, const int width = 0) {
  for (int i = 0; i < A->size1; ++i) {
    for (int j = 0; j < A->size2; ++j) {
      std::cout << std::setw(width) << gsl_matrix_get(A, i, j) << " ";
    }
    std::cout << std::endl;
  }
}

void print_matrix(const gsl_matrix_view &A, const int width = 0) {
  print_matrix(&A.matrix, width);
}

constexpr double quadratic(const double x, const double a, const double b, const double c) {
  return a*x*x + b*x + c;
}

int main() {
  constexpr double stepsize = 0.05;
  constexpr double start = 1.0;
  constexpr double end = 2.0;
  constexpr size_t required_size = (end-start)/stepsize + 1;

  // 2.1.1 is trivial, skipping

  ///////////
  // 2.1.2 //
  ///////////

  {
  
  std::cout << "Problem 2.1.2" << std::endl;
  
  /*
    linspace equivalent
  */

  std::cout << "linspace: ";
  
  gsl_vector *a = gsl_vector_calloc(required_size);
  gsl_vector_linspace(a, start, end);

  std::cout << "[ ";
  for (int i = 0; i < a->size; ++i) {
    std::cout << gsl_vector_get(a, i) << " ";
  }
  std::cout << "]" << std::endl;
  gsl_vector_free(a);

  /*
    arange equivalent
  */

  std::cout << "arange:   ";

  a = gsl_vector_calloc(required_size);
  gsl_vector_arange(a, start, end+stepsize, stepsize);

  std::cout << "[ ";
  for (int i = 0; i < a->size; ++i) {
    std::cout << gsl_vector_get(a, i) << " ";
  }
  std::cout << "]" << std::endl;
  gsl_vector_free(a);

  std::cout << std::endl;

  }

  ///////////
  // 2.1.3 //
  ///////////

  {

  Gnuplot gp;
  gp << "plot '-' title '2x^2 + 3x + 4' with linespoints\n";
  // send the data to gnuplot manually.  I could put it into an array
  // of some description and use gp.send1d, but this avoids needing to
  // store the data, and I can instead generate it dynamically.
  for (double x = -5; x <= 5; x += 0.1) {
    gp << x << " " << quadratic(x, 2, 3, 4) << "\n";
  }

  }

  ///////////
  // 2.1.4 //
  ///////////

  {

  std::cout << "Problem 2.1.4" << std::endl;

  // make a vector of numbers from 1-100
  gsl_vector *a = gsl_vector_calloc(100);
  gsl_vector_arange(a, 1, 100);

  // use the Fisher–Yates algorithm to shuffle the vector
  for (int N = a->size; N > 1; --N) {
    // swap the Nth element with the kth element
    int k = rand() % N;
    gsl_vector_swap_elements(a, N-1, k);
  }

  // reverse the vector
  // we can't just use a view with stride -1 here; the stride must be 1 to make a matrix view later
  gsl_vector *b = gsl_vector_alloc(a->size);
  gsl_vector_memcpy(b, a);
  gsl_vector_reverse(b);

  // get matrix views into the vector
  gsl_matrix_view A = gsl_matrix_view_vector(a, 10, 10);
  gsl_matrix_view B = gsl_matrix_view_vector(b, 10, 10);

  print_matrix(A, 3);
  std::cout << std::endl;
  print_matrix(B, 3);

  gsl_vector_free(b);
  gsl_vector_free(a);
  
  }

  ///////////
  // 2.1.5 //
  ///////////

  {

  std::cout << std::endl << "Problem 2.1.5" << std::endl;

  // make a matrix that is 1 everywhere
  gsl_matrix *A = gsl_matrix_alloc(5, 5);
  gsl_matrix_set_all(A, 1);

  // get a view into the diagonal
  gsl_vector_view a = gsl_matrix_diagonal(A);
  gsl_vector_linspace(&a.vector, 10, 50);

  print_matrix(A, 2);

  gsl_matrix_free(A);

  }

  ///////////
  // 2.1.6 //
  ///////////

  {

  std::cout << std::endl << "Problem 2.1.6" << std::endl;

  gsl_matrix *A = gsl_matrix_alloc(5,5);
  for (int i = 0; i < A->size1; ++i) {
    for (int j = 0; j < A->size2; ++j) {
      gsl_matrix_set(A, i, j, 101 + (rand() % 100));
    }
  }
  print_matrix(A, 3);

  // find max and min
  double max = gsl_matrix_get(A, 0, 0);
  size_t max_loc[2] = { 0, 0 };
  double min = max;
  size_t min_loc[2] = { 0, 0 };
  for (int i = 0; i < A->size1; ++i) {
    for (int j = 0; j < A->size2; ++j) {
      double v = gsl_matrix_get(A, i, j);
      if (v < min) {
        min = v;
        min_loc[0] = i;
        min_loc[1] = j;
      }
      if (v > max) {
        max = v;
        max_loc[0] = i;
        max_loc[1] = j;
      }
    }
  }
  std::cout << "Minimum is " << min << ", found at (" << min_loc[0] << "," << min_loc[1] << ")" << std::endl;
  std::cout << "Maximum is " << max << ", found at (" << max_loc[0] << "," << max_loc[1] << ")" << std::endl;
  gsl_matrix_set(A, min_loc[0], min_loc[1], max);
  gsl_matrix_set(A, max_loc[0], max_loc[1], min);
  print_matrix(A, 3);

  }

  {
    
  std::cout << std::endl << "Problem 2.1.7" << std::endl;

  gsl_matrix *V = gsl_matrix_alloc(2, 3);

  gsl_vector_view v1 = gsl_matrix_row(V, 0);
  gsl_vector_view v2 = gsl_matrix_row(V, 1);

  gsl_vector *vs[2] = { &v1.vector, &v2.vector };
  for (size_t i = 0; i < 2; ++i) {
    gsl_vector *v = vs[i];
    for (size_t j = 0; j < v->size; ++j) {
      // you asked for any real numbers!
      uint32_t x = rand();
      gsl_vector_set(v, j, *(float*)(&x));
    }
    std::cout << "v" << i+1 << ": ";
    print_vector(v);
  }

  std::cout << "V: " << std::endl;
  print_matrix(V);
  
  }

  return 0;
}
