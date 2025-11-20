#include <iostream>
#include <mpi.h>
#include <random>
#include <format>

// PS C:\projects\semester_3\mpv\ue3-mpi\x64\Debug> mpiexec -n 4 .\ue3-mpi.exe

int main__simple(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int M = 8000000, N = 8;

    std::default_random_engine gen{ 42 };
    std::uniform_real_distribution<double> dist{ -1.0, +1.0 };

    // broadcast x
    double* x = new double[N];

    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            x[i] = dist(gen);
        }

    }

    MPI_Bcast(
        x,
        N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // scatter A
    double* A = nullptr;
    if (rank == 0) {
        A = new double[M * N];
        for (int i = 0; i < M * N; i++)
        {
            A[i] = dist(gen);
        }
    }

    int rows_per_processor = M / comm_size;

    double* local_A = new double[rows_per_processor * N];

    MPI_Scatter(
        A,
        rows_per_processor * N,
        MPI_DOUBLE,
        local_A,
        rows_per_processor * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        delete[] A;
    }

    // calculate local y
    std::cout << "P " << rank << " starts calculating...\n" << std::flush;

    double* local_y = new double[rows_per_processor];

    for (int i = 0; i < rows_per_processor; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < N; j++) {
            local_y[i] += local_A[i * N + j] * x[j];
        }
    }

    delete[] local_A;
    delete[] x;

    // gather local results to root process
    double* y = nullptr;
    if (rank == 0) {
        y = new double[M];
    }

    MPI_Gather(
        local_y,
        rows_per_processor,
        MPI_DOUBLE,
        y,
        rows_per_processor,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    delete[] local_y;

    // print result on root process
    if (rank == 0) {
        std::cout << "Result: " << std::flush;
        for (int i = 0; i < std::min(10, M); i++) {
            std::cout << y[i] << " ";
        }
    }

    delete[] y;

    MPI_Finalize();

    return 0;
}
