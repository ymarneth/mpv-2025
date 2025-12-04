# Assignment: MPI Part 2

## Exercise 1: Rock, Paper, Scissors

### Init the game

As seen in the previous assignment, `MPI` needs to be initialized. The command-line arguments (`argc`, `argv`) are passed to MPI to allow for processing runtime options later on and the `rank` as well as the `comm_size` are passed a pointer.

To initialize the game, the number of rounds is set, the legal moves are defined and the score is initialized to zero.

```cpp
MPI_Init(&argc, &argv);

int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

const int ROUNDS = 10;
const int ROCK = 0, PAPER = 1, SCISSORS = 2;

int score = 0;

std::default_random_engine gen{42 + (unsigned int)rank};
std::uniform_int_distribution<int> dist{ROCK, SCISSORS};
```

### Prepare Cartesian communicator

In order to not calculate the neighbors manually, a cartesian communicator is created. This provides a convenient way to organize processes in a logical topology and automatically determine neighbor relationships.`MPI_Cart_create` creates a new communicator with a cartesian topology overlay on the existing `MPI_COMM_WORLD`. The topology defines how processes are logically arranged.

In this case `dims[1] = {size}` creates a 1-dimensional grid with a size of the number of available processes.

`periods[1] = {1}` enables periodic boundaries, meaning the topology wraps around. This creates a ring topology where the last process connects back to the first, so for example with 4 processes: `[0] - [1] - [2] - [3] - [0] - ...`.

`MPI_Cart_shift` can then automatically calculate the neighbors. `Dimension 0` hereby describes the axis along which to shift. Since it is a 1-dimensional grid, there is only one axis available. So `left_opponent` describes the process rank to the left with a displacement of -1 and `right_opponend` describes the process rank to the right with a displacement of +1.

```cpp
MPI_Comm cart_comm;
int dims[1] = {size};
int periods[1] = {1};
MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart_comm);

int cart_rank;
MPI_Comm_rank(cart_comm, &cart_rank);

int left_opponent, right_opponent;
MPI_Cart_shift(cart_comm, 0, 1, &left_opponent, &right_opponent);
```

This creates a tournament structure where each process plays against its left neighbor, forming pairs: (0↔3), (1↔0), (2↔1), (3↔2). The periodic boundary ensures every process has exactly one opponent with any number of processes.

### Version 1: Blocking Communication

In the first version, standard blocking send (`MPI_Ssend`) and receive (`MPI_Recv`) calls are used. This means that each process waits until the send or receive operation is complete, creating the risk of a deadlock. If all processes call `MPI_Ssend` simultaneously, each could be waiting for the other to receive.

```cpp
for (int rounds = 0; rounds < ROUNDS; rounds++)
{
    int choice = dist(gen);
    int opponent_choice;

    // send...

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

    constexpr int score_lookup[3][3] = {
        {0, -1, +1}, // ROCK versus others
        {+1, 0, -1}, // PAPER versus others
        {-1, +1, 0}  // SCISSORS versus others
    };

    score += score_lookup[choice][opponent_choice];
}
```

### Version 2: `MPI_Sendrecv` to avoid Deadlocks

`MPI_Sendrecv` combines send and receive in a single atomic operation. This approach avoids deadlocks, since sending and receiving happen simultaneously, processes do not wait indefinitely.

```cpp
for (int rounds = 0; rounds < ROUNDS; rounds++)
{
    int choice = dist(gen);
    int opponent_choice;

    // send...

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
        

    constexpr int score_lookup[3][3] = {
        {0, -1, +1}, // ROCK versus others
        {+1, 0, -1}, // PAPER versus others
        {-1, +1, 0}  // SCISSORS versus others
    };

    score += score_lookup[choice][opponent_choice];
}
```

### Version 3: Non-blocking communication

The third version uses non-blocking send (`MPI_Issend`) and receive (`MPI_Irecv`) calls.

Non-blocking behavior: These calls return immediately, allowing the program to perform other work or post multiple communication operations concurrently.

Synchronization: MPI_Waitall is used to ensure that all non-blocking operations complete before using the received data.

Advantages: Avoids deadlocks and can overlap computation with communication in more complex programs.

```cpp
for (int rounds = 0; rounds < ROUNDS; rounds++)
{
    int choice = dist(gen);
    int opponent_choice;

    // send...

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
```

### Finishing the game

In the end, the scores per processes is printed and `MPI` is shut down with `MPI_Finalize`.

```cpp
std::cout << "Process " << rank << ": " << score << std::endl
          << std::flush;

MPI_Finalize();
```

## Exercise 2: Zombie Outbreak
This task was done together by Yvonne Marneth and Jana Burns-Balog.

### Initializing the simulation
In this step we initialize MPI as usual, getting the rank and the size. We define our states for humans and zombies and for moving directions. Our global grid is 128x128. Every process gets its own random generator for initialising humans & zombies and for the movements.

```cpp
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
std::uniform_int_distribution<int> move_dist{ 0, 4 };
```

### Create process grid topology
In this step `MPI_Dims_create` calculates a distribution for the grids among the processes using the size. With `MPI_Cart_create` a new communicator is created. We have no wrap-around. `MPI_Cart_coords` gives every process its 2d coordinates in the grid.

```cpp
int dims[2] = { 0, 0 };
MPI_Dims_create(size, 2, dims);

int periods[2] = { 0, 0 }; // No wraparound
MPI_Comm grid_comm;
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder*/ 1, &grid_comm);

int coords[2];
MPI_Cart_coords(grid_comm, rank, 2, coords);
```

### Calculate local grid dimensions (block distribution with remainder)
The local row and col variables are the size of the tile of the current process. If the global variable isn't divisible by the dimensions, the rest is distributed on the first processes in the dimension. `start_row` and `start_col` are the global start indices of the current tile in the global grid. `MPI_Cart_shift` gives us the neighbors of a dimension. This is how we exchange halos with the neighbors.

```cpp
 int local_rows = GLOBAL_ROWS / dims[0];
 int local_cols = GLOBAL_COLS / dims[1];

 if (coords[0] < GLOBAL_ROWS % dims[0]) local_rows++;
 if (coords[1] < GLOBAL_COLS % dims[1]) local_cols++;

 int start_row = coords[0] * (GLOBAL_ROWS / dims[0]) + std::min(coords[0], GLOBAL_ROWS % dims[0]);
 int start_col = coords[1] * (GLOBAL_COLS / dims[1]) + std::min(coords[1], GLOBAL_COLS % dims[1]);

 int left, right, up, down;
 MPI_Cart_shift(grid_comm, 1, 1, &left, &right);
 MPI_Cart_shift(grid_comm, 0, 1, &up, &down);
```

### Allocate with halo
In this step every process saves its own tile with a halo. The size is `(local_rows + 2) * (local_cols + 2)`, where the + 2 are the halo. `idx(i, j)` converts 2D coordinates `(i, j)` into a 1D index in the flattened array. `i * (local_cols + 2)`jumps to the start of row i and + j selects the column inside that row. We use `i = 1..local_rows`and `j = 1..local_cols` for interior cells and `i = 0 / local_rows + 1)` and `j = 0 / local_cols + 1` are the halo cells. The two for loops randomly initialise the inside cells as human or zombies. The halos are initially human and are later filled by the neighbors.

```cpp
 std::vector<int> backing((local_rows + 2) * (local_cols + 2), HUMAN);
 std::vector<int> new_backing((local_rows + 2) * (local_cols + 2), HUMAN);

 auto idx = [&](int i, int j) { return i * (local_cols + 2) + j; };

 int* local_grid = backing.data();
 int* new_local_grid = new_backing.data();

 for (int i = 1; i <= local_rows; ++i)
    for (int j = 1; j <= local_cols; ++j)
        local_grid[idx(i, j)] = dist(gen);
```

### Derived data types for halos
`col_type` describes a column in the `local_grid`. We have `local_rows` elements with 1 int in each case and a distance of `local_cols+2`to the next element (length of a cell). `row_type` describes a continuous row with `local_cols` elements. 

```cpp
MPI_Datatype col_type, row_type;

MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_INT, &col_type);
MPI_Type_commit(&col_type);

MPI_Type_contiguous(local_cols, MPI_INT, &row_type);
MPI_Type_commit(&row_type);
```

### Round loops
We loop through the rounds, in each exchanging the halos, calcuating the infection, simulating movements and testing for convergence.

#### Halo exchange
The first step is to exchange halos with the neighbors. The left neighbor get the left inside column (col=1). From the left we receive the right inside column of the neighbor and save it in our own left halo (col=0). The same for right, up, and down. We use `MPI_Isend`, `MPI_Irecv` and `MPI_Waitall` in order to not block while exchanging in all four directions at the same time.

```cpp
std::vector<MPI_Request> requests;
requests.reserve(8);

if (left != MPI_PROC_NULL) {
    MPI_Request req;
    MPI_Isend(&local_grid[idx(1, 1)], 1, col_type, left, TAG_LEFT, grid_comm, &req);
    requests.push_back(req);
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
    ...
}
if (down != MPI_PROC_NULL) {
    MPI_Request req;
    MPI_Isend(&local_grid[idx(local_rows, 1)], 1, row_type, down, TAG_DOWN, grid_comm, &req);
    ...
}
if (!requests.empty())
    MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);
```

#### Infection
Now that we have all relevant cells in our `local_grid`, we check if any humans get infected. For every inside cell we check the four neighbors. Humans with at least two zombie neighbors become zombies. Zombies stay zombies.

```cpp
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
```

#### Movement
We create a new grid for the movements, filling it with -1 to indicate that nobody has moved there yet. Then a random movement is chosen for every cell. `ti` and `tj` are the end coordinates. With min and max we make sure, that these aren't outside of the local tiles. If there is a collision, the first one wins, the other doesn't move. In the last step unfilled cells are set back to their original value.

```cpp
std::vector<int> moved_backing((local_rows + 2) * (local_cols + 2), -1);
        int* moved_local_grid = moved_backing.data();

        for (int i = 1; i <= local_rows; ++i) {
            for (int j = 1; j <= local_cols; ++j) {
                int val = new_local_grid[idx(i, j)];
                int mv = move_dist(gen);
                int ti = i, tj = j;

                switch (mv) {
                case UP:    ti = std::max(1, i - 1); break;
                case DOWN:  ti = std::min(local_rows, i + 1); break;
                case LEFT:  tj = std::max(1, j - 1); break;
                case RIGHT: tj = std::min(local_cols, j + 1); break;
                default: break; // STAY
                }

                if (moved_local_grid[idx(ti, tj)] == -1) {
                    moved_local_grid[idx(ti, tj)] = val;
                }
                else {
                }
            }
        }

        for (int i = 1; i <= local_rows; ++i)
            for (int j = 1; j <= local_cols; ++j)
                if (moved_local_grid[idx(i, j)] == -1)
                    moved_local_grid[idx(i, j)] = new_local_grid[idx(i, j)];
```

#### Testing for convergence
Every process tests, if something has changed in its own tile. With `MPI_Allreduce`and `MPI_LOR` we combine all local flags into a global one. After that the new values are transferred to the `local_grid` and the end condition is checked.

```cpp
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

int global_changed = 0;
MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, grid_comm);

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
```
