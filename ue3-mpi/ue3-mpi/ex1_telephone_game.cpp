#include <iostream>
#include <mpi.h>
#include <random>

int main__telephone(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD , &comm_size);
    
    const int MAX_LENGTH = 100;

    if (rank == 0) {
        std::cout << "Process 0: Enter a message:\n";

        char original_message[MAX_LENGTH];
        std::cin.getline(original_message, MAX_LENGTH);


        MPI_Send(
            original_message,
            std::strlen(original_message) + 1, // with string terminator for c :/
            MPI_CHAR,
            /*to*/1, 
            /*tag*/0,
            MPI_COMM_WORLD
        );

        char final_message[MAX_LENGTH];
        MPI_Recv(
            final_message, 
            MAX_LENGTH, 
            MPI_CHAR,
            comm_size - 1, 
            0, 
            MPI_COMM_WORLD, 
            MPI_STATUSES_IGNORE
        );
        
        std::cout << "Final Message:\n" << final_message << std::flush;
    } else {
        char message[MAX_LENGTH];
        MPI_Recv(
            message,
            MAX_LENGTH,
            MPI_CHAR,
            /*from*/rank - 1,
            /*tag*/0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        std::default_random_engine gen { 42 + (unsigned int)rank };
        std::uniform_int_distribution<int> index_dist{ 0, (int)std::strlen(message) - 1 };
        std::uniform_int_distribution<int> char_dist{ (int)'a', (int)'z' };

        int idx = index_dist(gen);
        char new_char = (char)char_dist(gen);

        message[idx] = new_char;

        MPI_Send(
            message,
            std::strlen(message) + 1,
            MPI_CHAR,
            (rank + 1) % comm_size,
            0,
            MPI_COMM_WORLD
        );
    }

    MPI_Finalize();

    return 0;
}
