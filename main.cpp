#include <iostream>
#include <cstdlib>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_randist.h>
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
    int k = randint(0, N-1);
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
  for (size_t i = 0; i < A->size1; ++i) {
    for (size_t j = 0; j < A->size2; ++j) {
      gsl_matrix_set(A, i, j, 101 + randint(0, 99));
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

  gsl_matrix_free(A);
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
      gsl_vector_set(v, j, any_random_real());
    }
    std::cout << "v" << i+1 << ": ";
    print_vector(v);
  }

  std::cout << "V: " << std::endl;
  print_matrix(V);

  gsl_matrix_free(V);
}

void problem_2_1_8() {

  std::cout << std::endl << "Problem 2.1.8" << std::endl;

  gsl_matrix *M = gsl_matrix_alloc(3, 4);

  // fill with numbers between 0 and 1
  for (size_t i = 0; i < M->size1; ++i) {
    for (size_t j = 0; j < M->size2; ++j) {
      gsl_matrix_set(M, i, j, randreal<double>(0, 1));
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
  std::cout << std::endl;

  gsl_matrix_view B2 = gsl_matrix_submatrix(M, 0, 2, 3, 2);
  std::cout << "B2: " << std::endl;
  print_matrix(B2);

  gsl_matrix_free(M);
}

void problem_2_1_9() {

  std::cout << std::endl << "Problem 2.1.9" << std::endl;

  size_t N = randint(5, 9);
  std::cout << "N = " << N << std::endl;
  gsl_matrix *M = gsl_matrix_alloc(3, N);

  for (size_t i = 0; i < M->size1; ++i) {
    for (size_t j = 0; j < M->size2; ++j) {
      gsl_matrix_set(M, i, j, norm<double>(0, 1));
    }
  }

  std::cout << "M: " << std::endl;
  print_matrix(M);
  std::cout << std::endl;

  gsl_vector_view v = gsl_matrix_column(M, N-2);
  std::cout << "v: ";
  print_vector(v);

  gsl_matrix_free(M);
}

void problem_2_1_10() {
  std::cout << std::endl << "Problem 2.1.10" << std::endl;

  gsl_matrix *A = gsl_matrix_alloc(4, 3);
  for (size_t i = 0; i < A->size1; ++i) {
    for (size_t j = 0; j < A->size2; ++j) {
      gsl_matrix_set(A, i, j, randint<int>(1, 9));
    }
  }
  std::cout << "A: " << std::endl;
  print_matrix(A);

  // if we create the vector data as an array, we can easily use the
  // same storage for both.
  //
  // an alternative would be to create the column vector as a matrix
  // first, then get a vector view into it via
  // gsl_matrix_column. However, one advantage of allocating the
  // vector this way is that it exists solely on the stack (rather
  // than heap) and does not need to be freed (so no memory leaks).
  double b_arr[4];
  for (size_t j = 0; j < 4; ++j) {
    b_arr[j] = randint<int>(1, 9);
  }
  gsl_vector_view b = gsl_vector_view_array(b_arr, 4);
  std::cout << "b: ";
  print_vector(b);

  gsl_vector *y1 = gsl_vector_calloc(3);

  // using Level 2 BLAS (matrix-vector)
  //   for all intents and purposes, the vector acts like a column matrix
  //   gemv then does A @ x
  gsl_blas_dgemv(CblasTrans, 1, A, &b.vector, 0, y1);
  std::cout << "y1: ";
  print_vector(y1);
  std::cout << "Dims: " << y1->size << "x1" << std::endl;

  // using Level 3 BLAS (matrix-matrix, treating the vector as a column matrix)
  //   here, we then get gemm to do B^T @ A
  gsl_matrix *y2 = gsl_matrix_calloc(1, 3);
  gsl_matrix_view B = gsl_matrix_view_array(b_arr, 4, 1);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, &B.matrix, A, 0, y2);
  std::cout << "y2: ";
  print_matrix(y2);
  std::cout << "Dims: " << y2->size1 << "x" << y2->size2 << std::endl;

  gsl_matrix_free(A);
  gsl_vector_free(y1);
  gsl_matrix_free(y2);
}

void setup_histogram(Gnuplot &gp, double min, double max, int n, double fill_opacity = 0.0) {
  // script for histogram based on https://stackoverflow.com/a/19596160
  gp << "Min = " << std::fixed << min << "\n" // make sure Min and Max are floats, else the width will be broken
    "Max = " << std::fixed << max << "\n"
    "n = " << n << "\n"
    "width = (Max-Min)/n # binwidth\n"
    "bin(x) = width*(floor((x-Min)/width)+0.5) + Min\n"
    "set boxwidth width\n"
    "set xr [Min:Max]\n"
    "set style fill solid " << std::fixed << fill_opacity << "\n";
}

void problem_2_1_11() {

  std::cout << std::endl << "Problem 2.1.11" << std::endl;

  constexpr size_t N = 500000;
  constexpr size_t P = 5;

  gsl_matrix *M = gsl_matrix_alloc(N, P);
  for (size_t i = 0; i < M->size1; ++i) {
    for (size_t j = 0; j < M->size2; ++j) {
      gsl_matrix_set(M, i, j, norm<double>(0, 1));
    }
  }

  std::cout << "Matrix created!" << std::endl;

  Gnuplot gp;
  
  double means[N];
  double vars[N];
  for (size_t i = 0; i < M->size1; ++i) {
    gsl_vector_view v = gsl_matrix_row(M, i);
    means[i] = gsl_stats_mean(v.vector.data, v.vector.stride, v.vector.size);
    vars[i] = gsl_stats_variance_m(v.vector.data, v.vector.stride, v.vector.size, means[i]); // use the pre-calculated mean instead of recalculating
  }

  gp << "set multiplot layout 2, 1\n";
  setup_histogram(gp, -2.5, 2.5, 11, 0.5);
  gp << "plot" << gp.file1d(means) << "using (bin($1)):(1.0) smooth freq with boxes lc rgb'red' title 'mean'\n";
  setup_histogram(gp, 0, 5, 11, 0.5);
  gp << "plot" << gp.file1d(vars) << "using (bin($1)):(1.0) smooth freq with boxes lc rgb'green' title 'variance'\n";
  gp << "unset multiplot\n";

  gsl_matrix_free(M);
}

void problem_2_1_12() {
  std::cout << std::endl << "Problem 2.1.12" << std::endl;

  constexpr size_t num_ints = 1000;
  constexpr int min_num = 0;
  constexpr int max_num = 100;
  constexpr size_t num_bins = (max_num - min_num + 1);

  uint bins[num_bins] = {0};
  int L[num_ints]; // likely 4kB. Should be fine to hold in stack
  for (size_t i = 0; i < num_ints; ++i) {
    const int x = randint(min_num, max_num);
    L[i] = x;
    ++bins[min_num+x];
  }

  uint biggest_count = 0;
  int mode1 = -1;
  uint second_biggest_count = 0;
  int mode2 = -1; // = p
  for (size_t i = 0; i < num_bins; ++i) {
    if (bins[i] > second_biggest_count) {
      if (bins[i] > biggest_count) { // i.e. second_biggest_count < biggest_count < bins[i]
        second_biggest_count = biggest_count;
        mode2 = mode1;
        biggest_count = bins[i];
        mode1 = i + min_num;
      } else { // i.e. second_biggest_count < bins[i] < biggest_count
        second_biggest_count = bins[i];
        mode2 = i + min_num;
      }
    }
  }

  std::cout << "Biggest number: " << mode1 << " (" << biggest_count << " instances)" << std::endl;
  std::cout << " Second number: " << mode2 << " (" << second_biggest_count << " instances)" << std::endl;

  std::vector<int> Lltp(0);
  std::vector<int> Lgtep(0);

  for (size_t i = 0; i < num_ints; ++i) {
    const int x = L[i];
    if (x < mode2) Lltp.push_back(x);
    else Lgtep.push_back(x);
  }

  // print the vectors
  std::cout << "Less than p: ";
  for (int x : Lltp) {
    std::cout << x << " ";
  }
  std::cout << std::endl << "Gte p: ";
  for (int x : Lgtep) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
}

void problem_2_1_13() {
  std::cout << std::endl << "Problem 2.1.13" << std::endl;

  constexpr double r = 1;
  constexpr double cx = 1;
  constexpr double cy = 1;

  Gnuplot gp;
  gp << "set size ratio -1\n";
  gp << "set yr [" << std::fixed << cy - r << ":" << std::fixed << cy + r << "]\n";
  gp << "set xr [" << std::fixed << cx - r << ":" << std::fixed << cx + r << "]\n";
  gp << "set object 1 circle at " << std::fixed << cx << "," << std::fixed << cy << " size first " << std::fixed << r << " fc rgb'purple'\n";
  gp << "set style fill solid 0.5\n";
  gp << "plot '-' using 1:2 with filledcurves\n";

  double x0;
  double y0;
  for (size_t i = 0; i < 3; ++i) {
    const double theta = randreal(0., 2 * M_PI);
    const double x = cx + r*std::cos(theta);
    const double y = cy + r*std::sin(theta);
    gp << std::fixed << x << " " << std::fixed << y << "\n";
    if (i == 0) {
      x0 = x;
      y0 = y;
    }
  }
  gp << std::fixed << x0 << " " << std::fixed << y0 << "\n";
}

int coin_toss_experiment(const int N) {
  int heads = 0;
  for (int i = 0; i < N; ++i) {
    heads += randint(0,1);
  }
  return heads;
}

void problem_2_1_14() {
  std::cout << std::endl << "Problem 2.1.14" << std::endl;
  
  Gnuplot gp;

  constexpr int N = 10000;
  constexpr int M = 100;

  constexpr double ax_min = 4800;
  constexpr double ax_max = 5200;
  constexpr int n_bins = 25;
  constexpr int pdf_samples = 1000;

  constexpr double binwidth = (ax_max-ax_min)/(double)n_bins;

  setup_histogram(gp, ax_min, ax_max, n_bins, 0.5);
  gp << "plot '-' using (bin($1)):(1.0/" << std::fixed << (double)M * binwidth << ") smooth freq with boxes lc rgb'green' title '', '' using 1:2 w lines title 'Gaussian PDF'\n";
  for (int i = 0; i < M; ++i) {
    gp << coin_toss_experiment(N) << "\n";
  }
  gp << "e\n"; // indicates end of first data file

  constexpr double mu = (double)N/2;
  const double sigma = std::sqrt((double)N/4);
  for (int i = 0; i < pdf_samples; ++i) {
    double x = ax_min + (double)i*(ax_max - ax_min)/(double)pdf_samples;
    gp << std::fixed << x << " " << std::fixed << gsl_ran_gaussian_pdf(x - mu, sigma) << "\n";
  }
  gp << "e\n";
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
  problem_2_1_9();
  problem_2_1_10();
  problem_2_1_11();
  problem_2_1_12();
  problem_2_1_13();
  problem_2_1_14();

  return 0;
}
