#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float* matrix_dot(const float *X, const float *Y, size_t X_start, size_t X_end,
                  size_t k, size_t n)
{
    float *C = new float[(X_end - X_start) * n];
    for (size_t i = X_start; i < X_end; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            float c = 0;
            for (size_t z = 0; z < k; z++)
            {
                c += X[i * k + z] * Y[z * n + j];
            }
            C[(i - X_start) * n + j] = c;
        }
    }
    return C;
}


void matrix_softmax(float *X, size_t m, size_t n) {
    float *_sum = new float[m * 1];
    
    for(size_t i = 0; i < m; i++) {
        float cur_sum = 0;
        for(size_t j = 0; j < n; j++) {
            X[i * n + j] = exp(X[i * n + j]);
            cur_sum += X[i * n + j];
        }
        _sum[i] = cur_sum;
    }
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            X[i * n + j] /= _sum[i];
        }
    }
    delete _sum;
}

float* matrix_eye(const unsigned char *y, size_t y_start, size_t y_end, size_t k) 
{
    float *I = new float[(y_end - y_start) * k];
    for(size_t i = y_start; i < y_end; i++) {
        for(size_t j = 0; j < k; j++) {
            if(j==y[i]) I[(i - y_start) * k + j] = 1;
            else I[(i - y_start) * k + j] = 0;
        }
    }
    return I;
}

float* matrix_transpose(const float *X, size_t X_start, size_t X_end, size_t n) {
    size_t m = X_end - X_start;
    float *XT = new float[n * m];
    for(size_t i = X_start; i < X_end; i++) {
        for(size_t j = 0; j < n; j++) {
            XT[j * m + i - X_start] = X[i * n + j];
        }
    }
    return XT;
}

void matrix_subtraction(float *X, float *Y, size_t m, size_t n) 
{
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            X[i * n + j] -= Y[i * n + j]; 
        }
    }
}

void matrix_mul(float *X, float t, size_t m, size_t n) 
{
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            X[i * n + j] *= t;
        }
    }
}
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of exmaples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
      for(size_t i = 0; i < m; i += batch) {
          // X_batch : batch x n
          // thata : n x k
          // Z = X_batch \dot theta batch x k
          float *Z = matrix_dot(X, theta, i, i + batch, n, k);
          matrix_softmax(Z, batch, k);
          float *I = matrix_eye(y, i, i+batch, k);
          float *XT = matrix_transpose(X, i, i + batch, n);
          matrix_subtraction(Z, I, batch, k);
          float *g = matrix_dot(XT, Z, 0, n, batch, k);
          
          matrix_mul(g, (float)lr/(float)batch, n, k);
          matrix_subtraction(theta, g, n, k);
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
