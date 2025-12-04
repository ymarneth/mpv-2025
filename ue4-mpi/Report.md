# Assignment: MPI Part 2

## Exercise 1: Rock, Paper, Scissors

### Init the game

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

### Prepare Cart

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

### Version 1: Blocking Communication

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

```cpp
std::cout << "Process " << rank << ": " << score << std::endl
          << std::flush;

MPI_Finalize();
```

## Exercise 2: Zombie Outbreak

### Initializing the simulation

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
std::uniform_int_distribution<int> move_dist{ 0, 4 }; // 0=stay,1=up,2=down,3=left,4=right
```

### Create process grid topology

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

```cpp
 int local_rows = GLOBAL_ROWS / dims[0];
 int local_cols = GLOBAL_COLS / dims[1];

 if (coords[0] < GLOBAL_ROWS % dims[0]) local_rows++;
 if (coords[1] < GLOBAL_COLS % dims[1]) local_cols++;

 int start_row = coords[0] * (GLOBAL_ROWS / dims[0]) + std::min(coords[0], GLOBAL_ROWS % dims[0]);
 int start_col = coords[1] * (GLOBAL_COLS / dims[1]) + std::min(coords[1], GLOBAL_COLS % dims[1]);

 int left, right, up, down;
 MPI_Cart_shift(grid_comm, 1, 1, &left, &right); // horizontal neighbors
 MPI_Cart_shift(grid_comm, 0, 1, &up, &down);    // vertical neighbors
```

### Allocate with halo

```cpp
 std::vector<int> backing((local_rows + 2) * (local_cols + 2), HUMAN);
 std::vector<int> new_backing((local_rows + 2) * (local_cols + 2), HUMAN);

 auto idx = [&](int i, int j) { return i * (local_cols + 2) + j; };

 int* local_grid = backing.data();
 int* new_local_grid = new_backing.data();
```
