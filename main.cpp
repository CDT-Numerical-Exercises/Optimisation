#include <iostream>
#include <cstdlib>
#include <cmath>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gnuplot-iostream/gnuplot-iostream.h>

#include "helpers.h"

constexpr double quadratic(const double x, const double a, const double b, const double c) {
  return a*x*x + b*x + c;
}

void problem_2_1_2() {
  constexpr double stepsize = 0.05;
  constexpr double start = 1.0;
  constexpr double end = 2.0;
  constexpr size_t required_size = (end-start)/stepsize + 1;
  
  std::cout << "Problem 2.1.2" << std::endl;
  
  /*
    linspace equivalent
  */

  std::cout << "linspace: ";
  
  gsl_vector *a = gsl_vector_calloc(required_size);
  gsl_vector_linspace(a, start, end);
  print_vector(a);
  gsl_vector_free(a);

  /*
    arange equivalent
  */

  std::cout << "arange:   ";

  a = gsl_vector_calloc(required_size);
  gsl_vector_arange(a, start, end+stepsize, stepsize);
  print_vector(a);
  gsl_vector_free(a);

  std::cout << std::endl;

}

void problem_2_1_3() {
  Gnuplot gp;
  gp << "plot '-' title '2x^2 + 3x + 4' with linespoints\n";
  // send the data to gnuplot manually.  I could put it into an array
  // of some description and use gp.send1d, but this avoids needing to
  // store the data, and I can instead generate it dynamically.
  for (double x = -5; x <= 5; x += 0.1) {
    gp << x << " " << quadratic(x, 2, 3, 4) << "\n";
  }

}

void problem_2_1_4() {

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

void problem_2_1_5() {

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

void problem_2_1_6() {

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

void problem_2_1_7() {
    
  std::cout << std::endl << "Problem 2.1.7" << std::endl;

  gsl_matrix *V = gsl_matrix_alloc(2, 3);

  gsl_vector_view v1 = gsl_matrix_row(V, 0);
  gsl_vector_view v2 = gsl_matrix_row(V, 1);

  gsl_vector *vs[2] = { &v1.vector, &v2.vector };
  for (size_t i = 0; i < 2; ++i) {
    gsl_vector *v = vs[i];
    for (size_t j = 0; j < v->size; ++j) {
      // you asked for any real numbers!
      union { float f; uint32_t x; };
      do {
        x = rand();
      }
      while (!std::isfinite(f));
      gsl_vector_set(v, j, f);
    }
    std::cout << "v" << i+1 << ": ";
    print_vector(v);
  }

  std::cout << "V: " << std::endl;
  print_matrix(V);
  
}

void problem_2_1_8() {

  std::cout << std::endl << "Problem 2.1.8" << std::endl;

  gsl_matrix *M = gsl_matrix_alloc(3, 4);

  // fill with numbers between 0 and 1
  for (size_t i = 0; i < M->size1; ++i) {
    for (size_t j = 0; j < M->size2; ++j) {
      double r = (double)rand() / (double)RAND_MAX;
      gsl_matrix_set(M, i, j, r);
    }
  }

  std::cout << "M: " << std::endl;
  print_matrix(M);

  gsl_vector_view a[M->size1];
  for (size_t i = 0; i < M->size1; ++i) {
    a[i] = gsl_matrix_row(M, i);
    std::cout << "a" << i << ": ";
    print_vector(a[i]);
  }

  gsl_matrix_view B1 = gsl_matrix_submatrix(M, 0, 0, 3, 2);
  std::cout << "B1: " << std::endl;
  print_matrix(B1);

  gsl_matrix_view B2 = gsl_matrix_submatrix(M, 0, 2, 3, 2);
  std::cout << "B2: " << std::endl;
  print_matrix(B2);
}

int main() {
  // 2.1.1 is trivial, skipping
  problem_2_1_2();
  problem_2_1_3();
  problem_2_1_4();
  problem_2_1_5();
  problem_2_1_6();
  problem_2_1_7();
  problem_2_1_8();

  return 0;
}
