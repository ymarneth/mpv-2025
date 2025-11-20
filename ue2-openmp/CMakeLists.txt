cmake_minimum_required(VERSION 3.10)
project(ue2_openmp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP! Using detected settings.")

        # First program
        add_executable(parallelTrapezoidalRule parallelTrapezoidalRule.cpp)
        target_link_libraries(parallelTrapezoidalRule PUBLIC OpenMP::OpenMP_CXX)

        # Second program
        add_executable(parallelMatrixVectorMultiplication parallelMatrixVectorMultiplication.cpp)
        target_link_libraries(parallelMatrixVectorMultiplication PUBLIC OpenMP::OpenMP_CXX)

        # Third program
        add_executable(kMeansClustering kMeansClustering.cpp)
        target_link_libraries(kMeansClustering PUBLIC OpenMP::OpenMP_CXX)

else()
        message(WARNING "OpenMP not found. Forcing -fopenmp manually.")

        add_executable(parallelTrapezoidalRule parallelTrapezoidalRule.cpp)
        target_compile_options(parallelTrapezoidalRule PUBLIC -fopenmp)
        target_link_options(parallelTrapezoidalRule PUBLIC -fopenmp)

        add_executable(parallelMatrixVectorMultiplication parallelMatrixVectorMultiplication.cpp)
        target_compile_options(parallelMatrixVectorMultiplication PUBLIC -fopenmp)
        target_link_options(parallelMatrixVectorMultiplication PUBLIC -fopenmp)

        add_executable(kMeansClustering kMeansClustering.cpp)
        target_compile_options(kMeansClustering PUBLIC -fopenmp)
        target_link_options(kMeansClustering PUBLIC -fopenmp)
endif()
