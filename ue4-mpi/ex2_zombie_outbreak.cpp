#include <iostream>
#include <mpi.h>
#include <random>
#include <algorithm>
#include <vector>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ROUNDS = 100;

    const int HUMAN = 0, ZOMBIE = 1;
    const int STAY = 0, UP = 1, DOWN = 2, LEFT = 3, RIGHT = 4;

    const int GLOBAL_ROWS = 128;
    const int GLOBAL_COLS = 128;

    std::default_random_engine gen{ 42 + (unsigned int)rank };
    std::uniform_int_distribution<int> dist{ HUMAN, ZOMBIE };
    std::uniform_int_distribution<int> move_dist{ 0, 4 }; // 0=stay,1=up,2=down,3=left,4=right

    // Create process grid topology
    int dims[2] = { 0, 0 };
    MPI_Dims_create(size, 2, dims);

    int periods[2] = { 0, 0 }; // No wraparound
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder*/ 1, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    // Calculate local grid dimensions (block distribution with remainder)
    int local_rows = GLOBAL_ROWS / dims[0];
    int local_cols = GLOBAL_COLS / dims[1];

    if (coords[0] < GLOBAL_ROWS % dims[0]) local_rows++;
    if (coords[1] < GLOBAL_COLS % dims[1]) local_cols++;

    int start_row = coords[0] * (GLOBAL_ROWS / dims[0]) + std::min(coords[0], GLOBAL_ROWS % dims[0]);
    int start_col = coords[1] * (GLOBAL_COLS / dims[1]) + std::min(coords[1], GLOBAL_COLS % dims[1]);

    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, 1, &left, &right); // horizontal neighbors
    MPI_Cart_shift(grid_comm, 0, 1, &up, &down);    // vertical neighbors

    // Allocate with halo: (local_rows + 2) x (local_cols + 2)
    std::vector<int> backing((local_rows + 2) * (local_cols + 2), HUMAN);
    std::vector<int> new_backing((local_rows + 2) * (local_cols + 2), HUMAN);

    auto idx = [&](int i, int j) { return i * (local_cols + 2) + j; };

    int* local_grid = backing.data();
    int* new_local_grid = new_backing.data();

    // Initialize interior randomly
    for (int i = 1; i <= local_rows; ++i)
        for (int j = 1; j <= local_cols; ++j)
            local_grid[idx(i, j)] = dist(gen);

    if (rank == 0) {
        std::cout << "Grid dimensions: " << GLOBAL_ROWS << "x" << GLOBAL_COLS << std::endl;
        std::cout << "Process grid: " << dims[0] << "x" << dims[1] << std::endl;
        std::cout << "Starting zombie outbreak simulation with " << size << " processes" << std::endl;
    }

    std::cout << "Process " << rank << " (" << coords[0] << "," << coords[1] << ") manages "
        << local_rows << "x" << local_cols << " section starting at ("
        << start_row << "," << start_col << ")" << std::endl;

    // Create derived datatypes for halos
    MPI_Datatype col_type, row_type;
    
    MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    MPI_Type_contiguous(local_cols, MPI_INT, &row_type);
    MPI_Type_commit(&row_type);

    const int TAG_LEFT = 10;
    const int TAG_RIGHT = 11;
    const int TAG_UP = 20;
    const int TAG_DOWN = 21;

    for (int round = 0; round < ROUNDS; ++round)
    {
        // Halo exchange using derived types and non-blocking calls ---
        std::vector<MPI_Request> requests;
        requests.reserve(8);

        if (left != MPI_PROC_NULL) {
            // Send leftmost interior column to right halo of left neighbor
            MPI_Request req;
            MPI_Isend(&local_grid[idx(1, 1)], 1, col_type, left, TAG_LEFT, grid_comm, &req);
            requests.push_back(req);

            // Receive their rightmost col into our left halo
            MPI_Irecv(&local_grid[idx(1, 0)], 1, col_type, left, TAG_RIGHT, grid_comm, &req);
            requests.push_back(req);
        }
        
        if (right != MPI_PROC_NULL) {
            MPI_Request req;
            MPI_Isend(&local_grid[idx(1, local_cols)], 1, col_type, right, TAG_RIGHT, grid_comm, &req);
            requests.push_back(req);
            MPI_Irecv(&local_grid[idx(1, local_cols + 1)], 1, col_type, right, TAG_LEFT, grid_comm, &req);
            requests.push_back(req);
        }

        if (up != MPI_PROC_NULL) {
            MPI_Request req;
            MPI_Isend(&local_grid[idx(1, 1)], 1, row_type, up, TAG_UP, grid_comm, &req);
            requests.push_back(req);
            MPI_Irecv(&local_grid[idx(0, 1)], 1, row_type, up, TAG_DOWN, grid_comm, &req);
            requests.push_back(req);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Request req;
            MPI_Isend(&local_grid[idx(local_rows, 1)], 1, row_type, down, TAG_DOWN, grid_comm, &req);
            requests.push_back(req);
            MPI_Irecv(&local_grid[idx(local_rows + 1, 1)], 1, row_type, down, TAG_UP, grid_comm, &req);
            requests.push_back(req);
        }

        if (!requests.empty())
            MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        // Infection update
        for (int i = 1; i <= local_rows; ++i) {
            for (int j = 1; j <= local_cols; ++j) {
                int zombie_neighbors = 0;
                if (local_grid[idx(i - 1, j)] == ZOMBIE) ++zombie_neighbors; // up
                if (local_grid[idx(i + 1, j)] == ZOMBIE) ++zombie_neighbors; // down
                if (local_grid[idx(i, j - 1)] == ZOMBIE) ++zombie_neighbors; // left
                if (local_grid[idx(i, j + 1)] == ZOMBIE) ++zombie_neighbors; // right

                new_local_grid[idx(i, j)] = local_grid[idx(i, j)];
                if (local_grid[idx(i, j)] == HUMAN && zombie_neighbors >= 2) {
                    new_local_grid[idx(i, j)] = ZOMBIE;
                }
            }
        }

        // Movement in local_grid
        std::vector<int> moved_backing((local_rows + 2) * (local_cols + 2), -1);
        int* moved_local_grid = moved_backing.data();

        for (int i = 1; i <= local_rows; ++i) {
            for (int j = 1; j <= local_cols; ++j) {
                int val = new_local_grid[idx(i, j)]; // use post-infection value
                int mv = move_dist(gen);
                int ti = i, tj = j;

                switch (mv) {
                case UP:    ti = std::max(1, i - 1); break;
                case DOWN:  ti = std::min(local_rows, i + 1); break;
                case LEFT:  tj = std::max(1, j - 1); break;
                case RIGHT: tj = std::min(local_cols, j + 1); break;
                default: break; // stay
                }

                if (moved_local_grid[idx(ti, tj)] == -1) {
                    moved_local_grid[idx(ti, tj)] = val; // first writer wins
                }
                else {
                    // do nothing --> collision
                }
            }
        }

        // For any cell not written, fall back to the pre-movement value
        for (int i = 1; i <= local_rows; ++i)
            for (int j = 1; j <= local_cols; ++j)
                if (moved_local_grid[idx(i, j)] == -1)
                    moved_local_grid[idx(i, j)] = new_local_grid[idx(i, j)];

        // Compute local change comparing moved_local_grid to previous local_grid
        int local_changed = 0;
        for (int i = 1; i <= local_rows; ++i) {
            for (int j = 1; j <= local_cols; ++j) {
                if (moved_local_grid[idx(i, j)] != local_grid[idx(i, j)]) {
                    local_changed = 1;
                    break;
                }
            }
            if (local_changed) break;
        }

        // Global convergence check
        int global_changed = 0;
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, grid_comm);

        // Copy moved grid back into local_grid
        for (int i = 1; i <= local_rows; ++i)
            for (int j = 1; j <= local_cols; ++j)
                local_grid[idx(i, j)] = moved_local_grid[idx(i, j)];

        if (rank == 0) {
            std::cout << "Round " << round + 1 << " completed (global_changed=" << global_changed << ")" << std::endl;
        }

        if (!global_changed) {
            if (rank == 0) std::cout << "Converged at round " << round + 1 << std::endl;
            break;
        }
    }

    // Cleanup
    MPI_Type_free(&col_type);
    MPI_Type_free(&row_type);

    MPI_Finalize();
    return 0;
}
