#include <iostream>
#include <mpi.h>
#include <random>
#include <format>
#include <chrono>

int main__potato(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    auto time_seed = static_cast<unsigned int>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );

    std::default_random_engine gen{ time_seed };
    std::uniform_int_distribution<int> rank_dist{ 0, comm_size - 1 };

    bool game_running = true;
    int potato = -1;

    if (rank == 0) {
        std::cout << "=== Starting hot potato game\n" << std::flush;

        std::uniform_int_distribution<int> potato_dist{ 20, 50 };

        int original_potato = potato_dist(gen);
        
        int receiver;
        do {
            receiver = rank_dist(gen);
        } while (receiver == rank);

        std::cout << "=== Generated potato: " << original_potato << "\n" << std::flush;
        std::cout << "=== Sending potato to rank: " << receiver << "\n" << std::flush;

        MPI_Send(&original_potato, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
        potato = original_potato;
    }
    
    while (game_running) {
        MPI_Recv(
            &potato,
            1,
            MPI_INT,
            /*from*/MPI_ANY_SOURCE,
            /*tag*/0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        if (potato == -1)
        {
            game_running = false;
            break;
        }

        std::cout << "=== Rank " << rank << " received potato: " << potato << "\n" << std::flush;

        if (potato > 0) {
            potato -= 1;

            int receiver;
            do {
                receiver = rank_dist(gen);
            } while (receiver == rank);

            std::cout << "=== Sending potato to rank: " << receiver << "\n" << std::flush;

            MPI_Send(
                &potato,
                1,
                MPI_INT,
                receiver,
                0,
                MPI_COMM_WORLD
            );
        }
        else if (potato == 0) {
            game_running = false;
            std::cout << "Process " << rank << " has lost!" << std::flush;

            int end_signal = -1;
            for (int i = 0; i < comm_size; ++i) {
                if (i != rank)
                    MPI_Send(&end_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();

    return 0;
}
