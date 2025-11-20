#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <format>
#include <random>
#include <omp.h>

constexpr int N = 100000;
constexpr int d = 2;
double a = -100.0, b = 100.0;
int K = 5;
int max_iterations = 100;


void write_header(std::ostream &output) {
    output <<
            "Run Index;Threads;Implementation;Trial;Time (ms);Avg. Sequential Time (ms);Avg. Parallel Time (ms);Speedup (x);Efficiency (%)\n";
}

void write_state(std::ostream &output, int runIndex, int threads, const bool parallel,
                 int trial, double time_ms) {
    const std::string impl = parallel ? "parallel" : "sequential";
    output << runIndex << ';'
            << threads << ';'
            << impl << ';'
            << trial << ';'
            << time_ms << '\n';
}

void write_summary(std::ostream &output, int threads, double avgSeq, double avgPar,
                   double speedup, double efficiency) {
    output << ';'
            << threads << ';'
            << ';'
            << ';'
            << ';'
            << avgSeq << ';'
            << avgPar << ';'
            << speedup << ';'
            << efficiency << '\n';
}


double *initializeSamples() {
    auto *samples = new double[N * d];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> cluster_center_dist(a, b);
    std::normal_distribution<double> cluster_spread(0.0, 10.0); // cluster spread controls tightness

    std::vector<std::array<double, d> > centers(K);
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < d; ++j) {
            centers[k][j] = cluster_center_dist(gen);
        }
    }

    for (int i = 0; i < N; ++i) {
        int cluster_id = i % K;
        for (int j = 0; j < d; ++j) {
            samples[i * d + j] = centers[cluster_id][j] + cluster_spread(gen);
        }
    }

    return samples;
}

double *initializeCentroids(const double *samples) {
    auto *centroids = new double[K * d];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, N - 1);

    for (int k = 0; k < K; k++) {
        const int idx = dist(gen);
        for (int j = 0; j < d; j++) {
            centroids[k * d + j] = samples[idx * d + j];
        }
    }

    return centroids;
}

void assignDataPointsToCentroids(const double *samples, const double *centroids, int *assignments,
                                 const int num_samples,
                                 const int num_centroids, const int dimensions) {
    for (int i = 0; i < num_samples; i++) {
        double min_distance = std::numeric_limits<double>::max();
        int closest_centroid = -1;

        for (int k = 0; k < num_centroids; k++) {
            double distance = 0.0;
            for (int j = 0; j < dimensions; j++) {
                const double diff = samples[i * dimensions + j] - centroids[k * dimensions + j];
                distance += diff * diff;
            }

            if (distance < min_distance) {
                min_distance = distance;
                closest_centroid = k;
            }
        }

        assignments[i] = closest_centroid;
    }
}

void assignDataPointsToCentroidsParallel(const double *samples, const double *centroids, int *assignments,
                                         const int num_samples,
                                         const int num_centroids, const int dimensions, const int num_threads) {
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_samples; i++) {
        double min_distance = std::numeric_limits<double>::max();
        int closest_centroid = -1;

        for (int k = 0; k < num_centroids; k++) {
            double distance = 0.0;
            for (int j = 0; j < dimensions; j++) {
                const double diff = samples[i * dimensions + j] - centroids[k * dimensions + j];
                distance += diff * diff;
            }

            if (distance < min_distance) {
                min_distance = distance;
                closest_centroid = k;
            }
        }

        assignments[i] = closest_centroid;
    }
}

void updateCentroids(const double *samples, double *centroids, const int *assignments, const int num_samples,
                     const int num_centroids, const int dimensions) {
    std::vector counts(num_centroids, 0);
    std::vector new_centroids(num_centroids * dimensions, 0.0);

    for (int i = 0; i < num_samples; ++i) {
        const int cluster = assignments[i];
        counts[cluster]++;
        for (int j = 0; j < dimensions; ++j) {
            new_centroids[cluster * dimensions + j] += samples[i * dimensions + j];
        }
    }

    for (int k = 0; k < num_centroids; ++k) {
        if (counts[k] > 0) {
            for (int j = 0; j < dimensions; ++j) {
                centroids[k * dimensions + j] = new_centroids[k * dimensions + j] / counts[k];
            }
        }
    }
}

void runKMeansSequential(double *samples, double *centroids, int *assignments,
                         int num_samples, int num_centroids, int dimensions,
                         double tol = 1e-4) {
    std::vector<double> old_centroids(num_centroids * dimensions);

    for (int iter = 0; iter < max_iterations; iter++) {
        std::copy_n(centroids, num_centroids * dimensions, old_centroids.begin());

        assignDataPointsToCentroids(samples, centroids, assignments,
                                    num_samples, num_centroids, dimensions);

        updateCentroids(samples, centroids, assignments,
                        num_samples, num_centroids, dimensions);

        double shift = 0.0;
        for (int k = 0; k < num_centroids; k++) {
            for (int j = 0; j < dimensions; j++) {
                const double diff = centroids[k * dimensions + j] - old_centroids[k * dimensions + j];
                shift += diff * diff;
            }
        }
        if (shift < tol) break;
    }
}

void runKMeansParallel(double *samples, double *centroids, int *assignments,
                       int num_samples, int num_centroids, int dimensions,
                       int runIndex, int trial, std::ostream &out_file,
                       int num_threads, double tol = 1e-4) {
    std::vector<double> old_centroids(num_centroids * dimensions);

    for (int iter = 0; iter < max_iterations; iter++) {
        std::copy_n(centroids, num_centroids * dimensions, old_centroids.begin());

        assignDataPointsToCentroidsParallel(samples, centroids, assignments,
                                            num_samples, num_centroids, dimensions, num_threads);

        updateCentroids(samples, centroids, assignments,
                        num_samples, num_centroids, dimensions);

        double shift = 0.0;
        for (int k = 0; k < num_centroids; k++) {
            for (int j = 0; j < dimensions; j++) {
                const double diff = centroids[k * dimensions + j] - old_centroids[k * dimensions + j];
                shift += diff * diff;
            }
        }

        if (shift < tol) break;
    }
}

double measureKMeansRun(double *samples, double *centroids, int *assignments, int runIndex, int t,
                                       std::ostream &out_file, int num_threads, bool parallel) {
    const auto start = std::chrono::high_resolution_clock::now();

    if (parallel)
        runKMeansParallel(samples, centroids, assignments, N, K, d, runIndex, t, out_file, num_threads);
    else
        runKMeansSequential(samples, centroids, assignments, N, K, d, runIndex);

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    const auto time_ms = duration.count() / 1000.0;

    return time_ms;
}

void runExperiment(double *samples, int trials, int num_threads, int &runIndex,
                   std::ostream &out_file) {
    auto *assignments = new int[N];

    double seqTotalTime = 0.0;
    double parallelTotalTime = 0.0;

    for (int i = 0; i < trials; i++) {
        auto *centroids_seq = initializeCentroids(samples);
        const double seqTime =
            measureKMeansRun(samples, centroids_seq, assignments, runIndex, i, out_file, num_threads, false);
        delete[] centroids_seq;
        seqTotalTime += seqTime;
        write_state(out_file, runIndex, num_threads, false, i + 1, seqTime);
        runIndex++;
    }

    for (int j = 0; j < trials; j++) {
        auto *centroids_par = initializeCentroids(samples);
        const double parallelTime =
            measureKMeansRun(samples, centroids_par, assignments, runIndex, j, out_file, num_threads, true);
        delete[] centroids_par;
        parallelTotalTime += parallelTime;
        write_state(out_file, runIndex, num_threads, true, j + 1, parallelTime);
        runIndex++;
    }

    const double avgSeqTime = seqTotalTime / trials;
    const double avgParallelTime = parallelTotalTime / trials;

    const double speedup = avgSeqTime / avgParallelTime;
    const double efficiency = speedup / num_threads * 100;

    write_summary(out_file, num_threads, avgSeqTime, avgParallelTime, speedup, efficiency);

    delete[] assignments;
}

int main() {
    std::fstream out_file{"parallelKMeans.csv", std::ios::out};
    write_header(out_file);

    auto *samples = initializeSamples();

    const auto runIndex = new int(0);

    for (const int num_threads: {1, 2, 4, 8, 16, 32}) {
        constexpr int trials = 50;

        runExperiment(samples, trials, num_threads, *runIndex, out_file);
    }

    delete runIndex;
    delete[] samples;

    return 0;
}
