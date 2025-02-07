cmake_minimum_required(VERSION 3.30)
project(cuda_02_fractal CUDA)

set(CMAKE_CXX_STANDARD 17 )

#Habilitar OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

enable_language(CXX CUDA)
find_package(fmt)
find_package(SFML)
add_executable(cuda_02_fractal main.cpp
kernel.cu)

target_link_libraries(cuda_02_fractal fmt::fmt sfml::sfml)

target_include_directories(cuda_02_fractal PUBLIC
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(cuda_02_fractal PROPERTIES
CUDA_SEPARABLE_COMPILATION ON)

