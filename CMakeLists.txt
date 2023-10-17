cmake_minimum_required(VERSION 3.0)
project(NeuralNetworkSim)
find_package(IntelSYCL REQUIRED)
cmake_policy(SET CMP0079 NEW)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-unnamed-lambda")

include_directories($ENV{DPCPP_HOME}/llvm/build/include)
include_directories($ENV{DPCPP_HOME}/llvm/build/include/sycl/)
include_directories($ENV{DPCPP_HOME}/llvm/build/lib)
set(CMAKE_BUILD_TYPE Debug)
#add_compile_options(-fsycl -fsycl-unnamed-lambda)
add_link_options(-fsycl -fsycl-unnamed-lambda)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} src/main.cpp src/objects/DenseLayer.cpp include/DenseLayer.h include/Neuron.h src/objects/Model.cpp include/Model.h include/Structs.h include/NN.h)
# Add third party libraries.
add_subdirectory(libs)

#target_link_libraries(${PROJECT_NAME} Test)
target_include_directories(${PROJECT_NAME} PUBLIC include)
