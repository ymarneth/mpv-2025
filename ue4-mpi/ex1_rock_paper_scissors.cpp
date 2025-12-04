#include <iostream>
#include <mpi.h>
#include <random>

int main__rock(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int ROUNDS = 10;
    const int ROCK = 0, PAPER = 1, SCISSORS = 2;

    int score = 0;

    std::default_random_engine gen{42 + (unsigned int)rank};
    std::uniform_int_distribution<int> dist{ROCK, SCISSORS};

    MPI_Comm cart_comm;
    int dims[1] = {size};
    int periods[1] = {1};
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart_comm);

    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int left_opponent, right_opponent;
    MPI_Cart_shift(cart_comm, 0, 1, &left_opponent, &right_opponent);

    for (int rounds = 0; rounds < ROUNDS; rounds++)
    {
        int choice = dist(gen);
        int opponent_choice;

        // send...

        /*
        MPI_Ssend(
            &choice,
            1,
            MPI_INT,
            right_opponent,
            0,
            cart_comm
        );

        MPI_Recv(
            &opponent_choice,
            1,
            MPI_INT,
            left_opponent,
            0,
            cart_comm,
            MPI_STATUS_IGNORE
        );
        */

        /*
        MPI_Sendrecv(
            &choice,
            1,
            MPI_INT,
            right_opponent,
            0,
            &opponent_choice,
            1,
            MPI_INT,
            left_opponent,
            0,
            cart_comm,
            MPI_STATUS_IGNORE
            );
            */

        MPI_Request reqs[2];

        MPI_Issend(
            &choice,
            1,
            MPI_INT,
            right_opponent,
            0,
            cart_comm,
            &reqs[0]);

        MPI_Irecv(
            &opponent_choice,
            1,
            MPI_INT,
            left_opponent,
            0,
            cart_comm,
            &reqs[1]);

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        constexpr int score_lookup[3][3] = {
            {0, -1, +1}, // ROCK versus others
            {+1, 0, -1}, // PAPER versus others
            {-1, +1, 0}  // SCISSORS versus others
        };

        score += score_lookup[choice][opponent_choice];
    }

    std::cout << "Process " << rank << ": " << score << std::endl
              << std::flush;

    MPI_Finalize();

    return 0;
}
