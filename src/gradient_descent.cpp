#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "helpers.h"
#include "gradient_descent.h"

void do_descent(gsl_vector *x, gsl_vector *gradf, double lambda) {
  // we can be clever here
  // BLAS level 1 allows us to do calculations of the form
  //  y' = ax + y
  // Let a = -lambda
  //     x = gradf
  //     y = x
  // Now we can write the gradient descent in a BLAS-compatible
  // way to take advantage of its efficiency
  // The compiler should be smart enough to inline this function :)
  gsl_blas_daxpy(-lambda, gradf, x);
}

// implementation of gradient descent to find the minimum of a function
// Args:
//  - x0: starting point. Should be possible to feed this directly into gradf.
//  - gradf: function defining the vector gradient of the function. Takes the x vector as input, as well as a pointer to a vector to store the output. This should be pre-allocated, and should be the same size as x.
//  - lambda_n: function defining the learning rate for step n. The simplest example would be a constant learning rate function. However, this allows the learning rate to be defined algebraically (as can be done for some special case functions) or for a linesearch function to be implemented.
//  - path: pointer to a (standard library) vector for tracking the path taken by the gradient descent. If this is a null pointer, the path will not be tracked. Otherwise, this should be a pointer to a 2D double vector. The first axis corresponds to the step, and the second axis corresponds to xn for that step.
//  - precision: if x does not change by more than this amount between two steps, the gradient descent is considered finished.
//  - maxsteps: maximum number of gradient descent steps to perform before returning.
//
// x0 will be updated to reflect the located minimum. The return value of the function is the number of steps taken to reach this minimum. If this is equal to maxsteps, it is likely that the gradient descent did not converge in time.
iter_t gradient_descent(gsl_vector *x0, gradf_function gradf, learning_rate_function lambda_n, std::vector<std::vector<double>> *path, double precision, iter_t maxsteps) {
  double step_size;
  iter_t n = 0;
  gsl_vector *x_prev = gsl_vector_alloc(x0->size);
  gsl_vector *grad = gsl_vector_alloc(x0->size);
  double lambda;
  do {
    // copy the current value of x0 to x_prev for calculating the step size later
    gsl_vector_memcpy(x_prev, x0);

    // determine lambda for this iteration
    lambda = lambda_n(n);

    // determine the gradient for the current x0
    gradf(x0, grad);

    // append to the path (if necessary)
    if (path != NULL) {
      std::vector<double> x_Vec;
      for (int i = 0; i < x0->size; ++i) {
        x_Vec.push_back(gsl_vector_get(x0, i));
      }
      path->push_back(x_Vec);
    }

    // calculate new x
    do_descent(x0, grad, lambda);
    ++n;

    // calculate the step size
    // we update x_prev because we'll just memcpy it again at the
    // start of the loop and we take the norm anyway
    gsl_vector_sub(x_prev, x0);
    step_size = gsl_vector_length(x_prev);

  } while (step_size > precision && n < maxsteps);

  // add the last step to the path
  if (path != NULL) {
    std::vector<double> x_Vec;
    for (int i = 0; i < x0->size; ++i) {
      x_Vec.push_back(gsl_vector_get(x0, i));
    }
    path->push_back(x_Vec);
  }

  gsl_vector_free(x_prev);
  gsl_vector_free(grad);
  return n;
}
