#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <format>
#include <omp.h>

double *multiplySequential(const double *A, const double *x, const int rows, const int cols) {
    auto *result = new double[rows];

    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }

        result[i] = sum;
    }

    return result;
}

double *multiplyParallel(const double *A, const double *x, const int rows, const int cols, int num_threads) {
    auto *result = new double[rows];

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }
        result[i] = sum;
    }

    return result;
}

double measureMatrixMultiplicationTime(const double *A, const double *x, const int rows, const int cols,
                                       const int num_threads, const bool parallel = true) {
    double *result;

    const auto start = std::chrono::high_resolution_clock::now();

    if (parallel)
        result = multiplyParallel(A, x, rows, cols, num_threads);
    else
        result = multiplySequential(A, x, rows, cols);

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    const auto time_ms = duration.count() / 1000.0;

    delete[] result;

    return time_ms;
}

void write_header(std::ostream &output) {
    output <<
            "Run Index;Rows;Cols;Threads;Implementation;Trial;Time (ms);Avg. Sequential Time (ms);Avg. Parallel Time (ms); Speedup (x); Efficiency (%)\n";
}

void write_state(std::ostream &output, const int runIndex, const int rows, const int cols, const int thread_num,
                 const bool parallel, const int trial, const double time) {
    const std::string impl = parallel ? "parallel" : "sequential";
    output << runIndex << ';'
            << rows << ';'
            << cols << ';'
            << thread_num << ';'
            << impl << ';'
            << trial << ';'
            << time << '\n';
}

void write_summary(std::ostream &output, const int rows, const int cols, const int thread_num, const double avgSeqTime,
                   const double avgParallelTime, const double speedup, const double efficiency) {
    output << ';'
            << rows << ';'
            << cols << ';'
            << thread_num << ';'
            << ';'
            << ';'
            << ';'
            << avgSeqTime << ';'
            << avgParallelTime << ';'
            << speedup << ';'
            << efficiency << '\n';
    output << '\n';
}

void runExperiment(const int rows, const int cols, const int trials, const int num_threads, int &runIndex,
                   std::ostream &out_file) {
    auto *A = new double[rows * cols];
    for (long long i = 0; i < static_cast<long long>(rows) * cols; i++) {
        A[i] = 1.0;
    }

    auto *x = new double[cols];
    for (int i = 0; i < cols; i++) {
        x[i] = 1.0;
    }

    double seqTotalTime = 0.0;
    double parallelTotalTime = 0.0;

    for (int i = 0; i < trials; i++) {
        const double seqTime = measureMatrixMultiplicationTime(A, x, rows, cols, num_threads, false);
        seqTotalTime += seqTime;
        write_state(out_file, runIndex, rows, cols, num_threads, false, i + 1, seqTime);
        runIndex++;
    }

    for (int j = 0; j < trials; j++) {
        const double parallelTime = measureMatrixMultiplicationTime(A, x, rows, cols, num_threads, true);
        parallelTotalTime += parallelTime;
        write_state(out_file, runIndex, rows, cols, num_threads, true, j + 1, parallelTime);
        runIndex++;
    }

    const double avgSeqTime = seqTotalTime / trials;
    const double avgParallelTime = parallelTotalTime / trials;

    const double speedup = avgSeqTime / avgParallelTime;
    const double efficiency = speedup / num_threads * 100;

    write_summary(out_file, rows, cols, num_threads, avgSeqTime, avgParallelTime, speedup, efficiency);

    delete[] A;
    delete[] x;
}

int main() {
    std::fstream out_file{"parallelMatrixMultiplication.csv", std::ios::out};
    write_header(out_file);

    const auto runIndex = new int(0);

    for (int numThreadsValues[] = {1, 2, 4, 8, 16, 32}; const int numThreads: numThreadsValues) {
        constexpr int trials = 50;

        runExperiment(8000000, 8, trials, numThreads, *runIndex, out_file);

        runExperiment(8000, 8000, trials, numThreads, *runIndex, out_file);

        runExperiment(8, 8000000, trials, numThreads, *runIndex, out_file);
    }

    delete runIndex;

    return 0;
}
