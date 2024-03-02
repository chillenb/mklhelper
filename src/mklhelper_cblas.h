#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <stdexcept>
#include "mkl.h"

namespace nb = nanobind;

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

static inline CBLAS_LAYOUT get_layout(nb::ndarray<> &a) {
    if (a.ndim() != 2) {
        throw std::runtime_error("Matrix must be 2D");
    }
    if (a.stride(1) == 1)
        return CblasRowMajor;
    else if (a.stride(0) == 1)
        return CblasColMajor;
    else
        throw std::runtime_error("Matrix must be contiguous");
}

template <typename T>
static inline CBLAS_LAYOUT get_layout(nb::ndarray<T> &a) {
    if (a.ndim() != 2) {
        throw std::runtime_error("Matrix must be 2D");
    }
    if (a.stride(1) == 1)
        return CblasRowMajor;
    else if (a.stride(0) == 1)
        return CblasColMajor;
    else
        throw std::runtime_error("Matrix must be contiguous");
}




// void __cblas_helper_dgemm(nb::ndarray<> &a, nb::ndarray<> &b, nb::ndarray<> &c, double alpha, double beta);
// void __cblas_helper_zgemm(nb::ndarray<> &a, nb::ndarray<> &b, nb::ndarray<> &c, std::complex<double> alpha, std::complex<double> beta);

static inline void internalgemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                  CBLAS_TRANSPOSE TransB, MKL_INT M, MKL_INT N, MKL_INT K,
                  std::complex<double> alpha,
                  std::complex<double> *A,
                  MKL_INT lda,
                  std::complex<double> *B,
                  MKL_INT ldb,
                  std::complex<double> beta,
                  std::complex<double> *C,
                  MKL_INT ldc)
{
    cblas_zgemm(Layout, TransA, TransB, M, N, K, (void *)&alpha, (void *)A, lda,
                (void *)B, ldb, (void *)&beta, (void *)C, ldc);
}

static inline void internalgemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                  CBLAS_TRANSPOSE TransB, MKL_INT M, MKL_INT N, MKL_INT K,
                  std::complex<float> alpha,
                  std::complex<float> *A,
                  MKL_INT lda,
                  std::complex<float> *B,
                  MKL_INT ldb,
                  std::complex<float> beta,
                  std::complex<float> *C,
                  MKL_INT ldc)
{
    cblas_cgemm(Layout, TransA, TransB, M, N, K, (void *)&alpha, (void *)A, lda,
                (void *)B, ldb, (void *)&beta, (void *)C, ldc);
}

static inline void internalgemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                  CBLAS_TRANSPOSE TransB, MKL_INT M, MKL_INT N, MKL_INT K,
                  double alpha,
                  double *A,
                  MKL_INT lda,
                  double *B,
                  MKL_INT ldb,
                  double beta,
                  double *C,
                  MKL_INT ldc)
{
    cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}


static inline void internalgemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                  CBLAS_TRANSPOSE TransB, MKL_INT M, MKL_INT N, MKL_INT K,
                  float alpha,
                  float *A,
                  MKL_INT lda,
                  float *B,
                  MKL_INT ldb,
                  float beta,
                  float *C,
                  MKL_INT ldc)
{
    cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}



template <typename T>
void __cblas_helper_gemm(nb::ndarray<T> &a, nb::ndarray<T> &b, nb::ndarray<T> &c, T alpha, T beta, char outputContig) {
        if (!(a.dtype() == nb::dtype<T>() && a.ndim() == 2))
            throw std::runtime_error("Matrix must be 2D");
        if (!(b.dtype() == nb::dtype<T>() && b.ndim() == 2))
            throw std::runtime_error("Matrix must be 2D");
        if (!(c.dtype() == nb::dtype<T>() && c.ndim() == 2))
            throw std::runtime_error("Matrix must be 2D");

        if (outputContig != 'C' && outputContig != 'F')
            throw std::runtime_error("outputContig must be 'C' or 'F': " __FILE__ ":" STR(__LINE__));

        MKL_INT m = a.shape(0);
        MKL_INT n = b.shape(1);
        MKL_INT k = a.shape(1);

        CBLAS_TRANSPOSE opa, opb;
        MKL_INT lda, ldb, ldc;
        CBLAS_LAYOUT layout_c;

        if(outputContig == 'C') {
            opa = (get_layout(a) == CblasRowMajor) ? CblasNoTrans : CblasTrans;
            opb = (get_layout(b) == CblasRowMajor) ? CblasNoTrans : CblasTrans;
            lda = (opa == CblasNoTrans) ? k : m;
            ldb = (opb == CblasNoTrans) ? n : k;
            ldc = n;
            layout_c = CblasRowMajor;
        } else {
            opa = (get_layout(a) == CblasColMajor) ? CblasNoTrans : CblasTrans;
            opb = (get_layout(b) == CblasColMajor) ? CblasNoTrans : CblasTrans;
            lda = (opa == CblasNoTrans) ? m : k;
            ldb = (opb == CblasNoTrans) ? k : n;
            ldc = m;
            layout_c = CblasColMajor;
        }



        if(k != b.shape(0))
            throw std::runtime_error("Matrix dimensions do not match");

        internalgemm(layout_c, opa, opb, m, n, k, alpha, (T*) a.data(), lda, (T*) b.data(), ldb, beta, (T*) c.data(), ldc);
}


/*


template <typename T>
void __cblas_helper_trsm(nb::ndarray<T> &a, nb::ndarray<T> &b, T alpha, char uplo) {
        if (!(a.dtype() == nb::dtype<T>() && a.ndim() == 2))
            throw std::runtime_error("Matrix must be 2D");
        if (!(b.dtype() == nb::dtype<T>() && x.ndim() == 2))
            throw std::runtime_error("Matrix must be 2D");
        if (!(c.dtype() == nb::dtype<T>() && b.ndim() == 2))
            throw std::runtime_error("Matrix must be 2D");

        CBLAS_LAYOUT layout_a = get_layout(a);
        CBLAS_LAYOUT layout_b = get_layout(b);
        CBLAS_LAYOUT layout_x = get_layout(x);

        // left: A * X = B, X = B / A
        // op(A) = transpose if col major
        //

        // right: AT * XT = BT

        MKL_INT m = a.shape(0);
        MKL_INT n = b.shape(1);
        MKL_INT k = a.shape(1);

        const MKL_INT lda = (opa == CblasNoTrans) ? k : m;
        const MKL_INT ldb = (opb == CblasNoTrans) ? n : k;
        const MKL_INT ldc = n;

        if(k != b.shape(0))
            throw std::runtime_error("Matrix dimensions do not match");

        internalgemm(CblasRowMajor, opa, opb, m, n, k, alpha, (T*) a.data(), lda, (T*) b.data(), ldb, beta, (T*) c.data(), ldc);
}
*/