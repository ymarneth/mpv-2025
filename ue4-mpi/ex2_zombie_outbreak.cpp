#include <iostream>
#include <mpi.h>
#include <random>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ROUNDS = 10;
    
    const int HUMAN = 0, ZOMBIE = 1;
    const int STAY = 0, UP = 1, DOWN = 2, LEFT = 3, RIGHT = 4;
    
    const int GLOBAL_ROWS = 4;
    const int GLOBAL_COLS = 4;

    std::default_random_engine gen{ 42 + (unsigned int)rank };
    std::uniform_int_distribution<int> dist{ HUMAN, ZOMBIE};
    std::uniform_int_distribution<int> move_dist{0, 3}; // 0=stay, 1=up, 2=down, 3=left, 4=right

    // Create process grid topology
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0}; // No wraparound
    MPI_Comm grid_comm;

    MPI_Cart_create(
        MPI_COMM_WORLD,
        2, // 2D grid
        dims,
        periods,
        /*reorder*/ 1,
        &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    
    // Calculate local grid dimensions for this process
    int local_rows = GLOBAL_ROWS / dims[0];
    int local_cols = GLOBAL_COLS / dims[1];
    
    if (coords[0] < GLOBAL_ROWS % dims[0]) local_rows++;
    if (coords[1] < GLOBAL_COLS % dims[1]) local_cols++;
    
    int start_row = coords[0] * (GLOBAL_ROWS / dims[0]) + std::min(coords[0], GLOBAL_ROWS % dims[0]);
    int start_col = coords[1] * (GLOBAL_COLS / dims[1]) + std::min(coords[1], GLOBAL_COLS % dims[1]);

    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, 1, &left, &right);   // horizontal neighbors
    MPI_Cart_shift(grid_comm, 0, 1, &up, &down);      // vertical neighbors

    int** local_grid = new int*[local_rows + 2];
    int** new_local_grid = new int*[local_rows + 2];
    
    for (int i = 0; i < local_rows + 2; i++) {
        local_grid[i] = new int[local_cols + 2];
        new_local_grid[i] = new int[local_cols + 2];
    }

    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= local_cols; j++) {
            local_grid[i][j] = dist(gen);
        }
    }
    
    if (rank == 0) {
        std::cout << "Grid dimensions: " << GLOBAL_ROWS << "x" << GLOBAL_COLS << std::endl;
        std::cout << "Process grid: " << dims[0] << "x" << dims[1] << std::endl;
        std::cout << "Starting zombie outbreak simulation with " << size << " processes" << std::endl;
    }
    
    std::cout << "Process " << rank << " (" << coords[0] << "," << coords[1] << ") manages " 
              << local_rows << "x" << local_cols << " section starting at (" 
              << start_row << "," << start_col << ")" << std::endl;

    for (int round = 0; round < ROUNDS; round++)
    {
        MPI_Request requests[8];
        int req_count = 0;
        
        if (left != MPI_PROC_NULL) {
            for (int i = 1; i <= local_rows; i++) {
                MPI_Isend(&local_grid[i][1], 1, MPI_INT, left, 0, grid_comm, &requests[req_count++]);
                MPI_Irecv(&local_grid[i][0], 1, MPI_INT, left, 0, grid_comm, &requests[req_count++]);
            }
        }
        
        if (right != MPI_PROC_NULL) {
            for (int i = 1; i <= local_rows; i++) {
                MPI_Isend(&local_grid[i][local_cols], 1, MPI_INT, right, 0, grid_comm, &requests[req_count++]);
                MPI_Irecv(&local_grid[i][local_cols + 1], 1, MPI_INT, right, 0, grid_comm, &requests[req_count++]);
            }
        }
        
        if (up != MPI_PROC_NULL) {
            MPI_Isend(&local_grid[1][1], local_cols, MPI_INT, up, 1, grid_comm, &requests[req_count++]);
            MPI_Irecv(&local_grid[0][1], local_cols, MPI_INT, up, 1, grid_comm, &requests[req_count++]);
        }
        
        if (down != MPI_PROC_NULL) {
            MPI_Isend(&local_grid[local_rows][1], local_cols, MPI_INT, down, 1, grid_comm, &requests[req_count++]);
            MPI_Irecv(&local_grid[local_rows + 1][1], local_cols, MPI_INT, down, 1, grid_comm, &requests[req_count++]);
        }
        
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= local_cols; j++) {
                int zombie_neighbors = 0;
                
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0) continue;
                        if (local_grid[i + di][j + dj] == ZOMBIE) {
                            zombie_neighbors++;
                        }
                    }
                }
                
                new_local_grid[i][j] = local_grid[i][j];
                if (local_grid[i][j] == HUMAN && zombie_neighbors >= 2) {
                    new_local_grid[i][j] = ZOMBIE;
                }
            }
        }
        
        // Random movement within local boundaries
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= local_cols; j++) {
                int move = move_dist(gen);
                int new_i = i, new_j = j;
                
                switch (move) {
                    case 1: new_i = std::max(1, i - 1); break;          // up
                    case 2: new_i = std::min(local_rows, i + 1); break; // down
                    case 3: new_j = std::max(1, j - 1); break;          // left
                    case 4: new_j = std::min(local_cols, j + 1); break; // right
                    default: break;
                }
       
            }
        }
       
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= local_cols; j++) {
                local_grid[i][j] = new_local_grid[i][j];
            }
        }
        
        if (rank == 0) {
            std::cout << "Round " << round + 1 << " completed" << std::endl;
        }
    }

    for (int i = 0; i < local_rows + 2; i++) {
        delete[] local_grid[i];
        delete[] new_local_grid[i];
    }
    delete[] local_grid;
    delete[] new_local_grid;

    MPI_Finalize();

    return 0;
}
