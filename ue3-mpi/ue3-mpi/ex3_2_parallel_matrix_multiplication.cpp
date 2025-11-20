#include <iostream>
#include <mpi.h>
#include <random>
#include <format>

// PS C:\projects\semester_3\mpv\ue3-mpi\x64\Debug> mpiexec -n 4 .\ue3-mpi.exe

int main__extended(int argc, char** argv)
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

    int base_rows = M / comm_size;
    int remainder_rows = M % comm_size;

    int* rows_for_process = new int[comm_size];
    int* sendcounts = new int[comm_size];
    int* displs = new int[comm_size];

    displs[0] = 0;
    for (int p = 0; p < comm_size; p++) {
        rows_for_process[p] = base_rows + (p < remainder_rows ? 1 : 0);
        sendcounts[p] = rows_for_process[p] * N;
        if (p > 0) {
            displs[p] = displs[p - 1] + sendcounts[p - 1];
        }
    }

    int local_rows = rows_for_process[rank];
    double* local_A = new double[local_rows * N];

    MPI_Scatterv(
        A,
        sendcounts,
        displs,
        MPI_DOUBLE,
        local_A,
        sendcounts[rank],
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        delete[] A;
    }

    // calculate local y
    std::cout << "P " << rank << " starts calculating...\n" << std::flush;

    double* local_y = new double[rows_for_process[rank]];

    for (int i = 0; i < rows_for_process[rank]; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < N; j++) {
            local_y[i] += local_A[i * N + j] * x[j];
        }
    }

    delete[] local_A;
    delete[] x;

    // gather local results to root process
    double* y = (rank == 0) ? new double[M] : nullptr;
    
    int* recvcounts_y = new int[comm_size];
    int* displs_y = new int[comm_size];

    displs_y[0] = 0;
    for (int p = 0; p < comm_size; p++) {
        recvcounts_y[p] = rows_for_process[p];
        if (p > 0) {
            displs_y[p] = displs_y[p - 1] + recvcounts_y[p - 1];
        }
    }

    MPI_Gatherv(
        /*sendbuf*/ local_y,
        /*sendcount*/ local_rows,
        /*sendtype*/ MPI_DOUBLE,
        /*recvbuf*/ y,
        /*recvcounts*/ recvcounts_y,
        /*displs*/ displs_y,
        /*recvtype*/ MPI_DOUBLE,
        /*root*/ 0,
        /*comm*/ MPI_COMM_WORLD
    );

    delete[] local_y;
    delete[] rows_for_process;
    delete[] sendcounts;
    delete[] displs;
    delete[] recvcounts_y;
    delete[] displs_y;

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
