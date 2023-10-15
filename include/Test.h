//
// Created by Josh Shiells on 08/08/2023.
//

#ifndef NEURALNETWORKSIM_TEST_H
#define NEURALNETWORKSIM_TEST_H
#include "CL/sycl.hpp"
void test() {
    // Generate random 0-1 input array
    sycl::queue q;
    const int N = 2;
    std::cout << "Device : " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    auto *data = static_cast<int *>(sycl::malloc_shared(N * sizeof(int), q));  // Size: 2

    for (int i = 0; i < N; i++) data[i] = i;

    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { data[i] *= 2; }).wait();

    for (int i = 0; i < N; i++) {
        std::cout << data[i] << std::endl;
    }
    sycl::free(data, q);
}

#endif //NEURALNETWORKSIM_TEST_H
