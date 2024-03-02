#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <stdexcept>
#include "mkl.h"
#include "mklhelper_cblas.h"

namespace nb = nanobind;

using namespace nb::literals;



NB_MODULE(mklhelper_ext, m) {
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
    m.def("inspect", [](nb::ndarray<> &a) {
        printf("Array data pointer : %p\n", a.data());
        printf("Array dimension : %zu\n", a.ndim());
        for (size_t i = 0; i < a.ndim(); ++i) {
            printf("Array dimension [%zu] : %zu\n", i, a.shape(i));
            printf("Array stride    [%zu] : %zd\n", i, a.stride(i));
        }
        printf("Device ID = %u (cpu=%i, cuda=%i)\n", a.device_id(),
            int(a.device_type() == nb::device::cpu::value),
            int(a.device_type() == nb::device::cuda::value)
        );
        printf("Array dtype: int16=%i, uint32=%i, float32=%i\n",
            a.dtype() == nb::dtype<int16_t>(),
            a.dtype() == nb::dtype<uint32_t>(),
            a.dtype() == nb::dtype<float>()
        );
    });
    m.def("dgemm", __cblas_helper_gemm<double>, "a"_a, "b"_a, "c"_a, "alpha"_a=1.0, "beta"_a=0.0, "outputContig"_a='C');
    m.def("zgemm", __cblas_helper_gemm<std::complex<double>>, "a"_a, "b"_a, "c"_a, "alpha"_a=std::complex<double>(1.0, 0.0), "beta"_a=std::complex<double>(0.0, 0.0), "outputContig"_a='C');
    m.def("sgemm", __cblas_helper_gemm<float>, "a"_a, "b"_a, "c"_a, "alpha"_a=1.0f, "beta"_a=0.0f, "outputContig"_a='C');
    m.def("cgemm", __cblas_helper_gemm<std::complex<float>>, "a"_a, "b"_a, "c"_a, "alpha"_a=std::complex<float>(1.0f, 0.0f), "beta"_a=std::complex<float>(0.0f, 0.0f), "outputContig"_a='C');
}