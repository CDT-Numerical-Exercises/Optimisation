#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "helpers.h"
#include "gradient_descent.h"
#include "linesearch.h"
#ifdef VERBOSE_LINESEARCH
#include <iostream>
#endif

// returns the search direction in vector p. p should be
// pre-allocated, and should be the same size as x.
// also returns the gradient (to aid in calculating m) as grad,
// which again should be pre-allocated and the same size as x.
void get_search_direction(const gsl_vector *x, gradf_function gradf, gsl_vector *p, gsl_vector *grad) {
  gradf(x, p);
  gsl_vector_memcpy(grad, p);

  // some sources (e.g. [1, 3]) suggest using the unit vector grad as
  // the search direction, i.e. -∇f/|∇f|. However, to apply this to
  // gradient descent, it only makes sense to use p = -∇f (as
  // suggested in [4]).
  //
  // The reason for this is as follows: the gradient descent algorithm
  // finds the minimum by stepping to x' = x - λ∇f on each
  // iteration. The linesearch algorithm is essentially testing values
  // of α in order to find one where x' = x + αp will produce an
  // efficient descent according to the Armijo rule. If we let p =
  // -∇f, then α directly corresponds to λ, and we can feed it back
  // into the gradient descent algorithm. If we use a different p,
  // such as p = -∇f/|∇f|, the linesearch algorithm will still find a
  // suitable step size, but it will be specific to this search
  // vector.
  //
  // The unit vector version can be used when the linesearch algorithm
  // forms the basis of the minimisation; that is, we iteratively
  // minimise the function by setting x' = x + αp, where p does not
  // necessarily equal -∇f. This algorithm is effectively the same as
  // gradient descent, but allows for a different choice of descent
  // direction in order to improve the efficiency of the algorithm
  // [1]. To reiterate, however, for α to be valid as the step size in
  // standard gradient descent, the choice of p = -∇f is required.
  gsl_vector_scale(p, -1.);
}

// implemented based on descriptions of the algorithm found at [1], [2]
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
    #ifdef VERBOSE_LINESEARCH
    std::cout << "alpha: " << alpha << ", diff: " << diff << ", alpha*t: " << alpha*t << std::endl;
    #endif
  } while (diff > alpha*t);

  gsl_vector_free(p);
  gsl_vector_free(grad);
  gsl_vector_free(xp);

  return alpha;
}

/*
  References

[1] https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods
[2] https://en.wikipedia.org/wiki/Backtracking_line_search#Algorithm
[3] https://sites.math.washington.edu/~burke/crs/516/notes/backtracking.pdf
[4] https://en.wikipedia.org/wiki/Wolfe_conditions

*/
