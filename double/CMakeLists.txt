cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_ARCHITECTURES 89 90)
project(getrf VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(GEMMUL8_DIR "/home/ruinique/GEMMul8/GEMMul8")
set(GEMMUL8_INCLUDE_DIR "${GEMMUL8_DIR}/include")
set(GEMMUL8_LIB_DIR "${GEMMUL8_DIR}/lib")
add_library(gemmul8 STATIC IMPORTED)
set_target_properties(gemmul8 PROPERTIES
  IMPORTED_LOCATION "${GEMMUL8_LIB_DIR}/libgemmul8.a"
  INTERFACE_INCLUDE_DIRECTORIES "${GEMMUL8_INCLUDE_DIR}"
)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 89 90)
endif()

# 判断是否有 CUDA 环境
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA 环境已找到")
else()
    message(STATUS "CUDA 环境未找到")
endif()

add_executable(${PROJECT_NAME} getrf.cu)

target_include_directories(${PROJECT_NAME} PUBLIC ${GEMMUL8_INCLUDE_DIR})

# 链接 CUDA 库
target_link_libraries(${PROJECT_NAME} 
    gemmul8
    CUDA::cuda_driver 
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cusolver
    CUDA::curand)

target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -g>)