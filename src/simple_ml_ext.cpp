#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

#include <cstdlib>

namespace py = pybind11;

float* mmproductFloat(const float *A, const float *B, size_t m, size_t n, size_t k) {
    float* result = (float*)malloc(sizeof(float)*m*k);
    memset(result, 0, sizeof(float)*m*k);
    for(size_t i = 0; i < m; i++) { // first dim of norm_z
        for(size_t j = 0; j < k; j++) { // second dim of norm_z
            for(size_t ni = 0; ni < n; ni++) {
                result[i*k + j] += A[i*n + ni] * B[ni*k + j];
            }
        }
    }
    return result;
}

float* transpose(const float *X, size_t n, size_t X_start, size_t X_end) {
    size_t batch_size = X_end-X_start;
    
    float* trans = (float*)malloc(sizeof(float)*n*batch_size);
    memset(trans, 0, sizeof(sizeof(float)*n*batch_size));
    for(size_t i = 0; i < n; i++) { // first dim of norm_z
        for(size_t j = 0; j < batch_size; j++) { // second dim of norm_z
            trans[i*batch_size + j] = X[(j+X_start)*n + i];
        }
    }
    return trans;
}

float* softmax(
    const float *X,
    float *theta, size_t m, size_t n, size_t k,
    size_t X_start, size_t X_end)
{
    size_t batch_size = X_end-X_start;
    
    float* norm_z = (float*)malloc(sizeof(float)*(batch_size)*k); // size[batch, k]
    // fill 0
    memset(norm_z, 0, sizeof(float)*(batch_size)*k);
    // X * theta
    for(size_t i = 0; i < batch_size; i++) { // first dim of norm_z
        for(size_t j = 0; j < k; j++) { // second dim of norm_z
            for(size_t ni = 0; ni < n; ni++) {
                norm_z[i*k + j] += X[(i+X_start)*n + ni] * theta[ni*k + j];
            }
        }
        // 计算exp 和 sum
        float sum = 0.0;
        for(size_t j = 0; j < k; j++) {
            norm_z[i*k + j] = exp(norm_z[i*k + j]);
            sum += norm_z[i*k + j];
        }
        for(size_t j = 0; j < k; j++) {
            norm_z[i*k + j] /= sum; // normalized(exp)
        }
    }

    return norm_z;
}

void softmax_regression_epoch_batch(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  size_t X_start, size_t X_end, float lr)
{
    size_t batch = X_end - X_start;
    // normalized softmax predict  m*k
    float* norm_z = softmax(X, theta, m, n, k, X_start, X_end); // size[batch, k];
    // one_hot;
    for(size_t i = 0; i < batch; i++) {
        norm_z[i*k + (uint8_t)y[i+X_start]] -= 1.0;
    }
    float* trans_x = transpose(X, n, X_start, X_end); // size[n, batch];

    float* gradient = mmproductFloat(trans_x, norm_z, n, batch, k); // size[n, k];

    for(size_t ni = 0; ni < n; ni++) {
        for(size_t ki = 0; ki < k; ki++) {
            theta[ni*k + ki] -= lr * (gradient[ni*k + ki] / (float)batch);
        }
    }

    std::free(norm_z);
    std::free(gradient);
    std::free(trans_x);
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
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t i = 0;
    while(i + batch <= m) {
        softmax_regression_epoch_batch(X, y, theta, m, n, k, i, i+batch, lr);
        i += batch;
    }

    if (i < m) {
        softmax_regression_epoch_batch(X, y, theta, m, n, k, i, m, lr);
    }
    /// END YOUR CODE
}