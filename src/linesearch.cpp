#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "helpers.h"
#include "gradient_descent.h"
#include "linesearch.h"
#include <iostream>

// returns the search direction in vector p. p should be
// pre-allocated, and should be the same size as x.
// also returns the gradient (to aid in calculating m) as grad,
// which again should be pre-allocated and the same size as x.
void get_search_direction(const gsl_vector *x, gradf_function gradf, gsl_vector *p, gsl_vector *grad) {
  gradf(x, p);
  gsl_vector_memcpy(grad, p);
  double length = gsl_blas_dnrm2(p);
  gsl_vector_scale(p, -1./length);
}

// implemented based on descriptions of the algorithm found at:
//  - https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods#Backtracking_Line_Search
//  - https://en.wikipedia.org/wiki/Backtracking_line_search#Algorithm
double backtracking_linesearch(const gsl_vector *x, function f, gradf_function gradf, const double alpha0, const double tau, const double c) {
  gsl_vector *p = gsl_vector_alloc(x->size);
  gsl_vector *grad = gsl_vector_alloc(x->size);

  get_search_direction(x, gradf, p, grad);
  double m;
  gsl_blas_ddot(grad, p, &m);
  // p and m should now stay the same, as we will never change x

  double alpha = alpha0;
  double newalpha = alpha;
  double diff;
  const double t = c * m;
  gsl_vector *xp = gsl_vector_alloc(x->size);
  do {
    alpha = newalpha;
    // compute diff
    const double fx = f(x);
    gsl_vector_memcpy(xp, p);
    gsl_vector_scale(xp, alpha);
    gsl_vector_add(xp, x);
    const double fxp = f(xp);

    diff = fxp - fx;
    newalpha = tau*alpha;
    std::cout << "alpha: " << alpha << ", diff: " << diff << std::endl;
  } while (diff > alpha*t);

  return alpha;
}
