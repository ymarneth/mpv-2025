#include <iostream>
#include <iomanip>
#include <omp.h>

double f(double x);

double trap(double a, double b, long long N, int num_threads) {
    double h = (b - a) / N;
    double sum = (f(a) + f(b)) / 2.0;

    #pragma omp parallel for num_threads(num_threads) reduction(+:sum)
    for (long long i = 1; i < N; i++) {
        double x_i = a + i * h;
        sum += f(x_i);
    }

    sum *= h;
    return 4 * sum;
}

int main(int argc, char **argv) {
    long long samples = 1000000000;
    int num_threads = 4;

    if (argc == 3) {
        samples = std::atoi(argv[1]);
        num_threads = std::atoi(argv[2]);
    }

    double result = trap(0, 1, samples, num_threads);
    std::cout << "Approx integral: " << std::setprecision(15) << result << std::endl;
    return 0;
}

double f(const double x) {
    return 1.0 / (1.0 + x * x);
}
