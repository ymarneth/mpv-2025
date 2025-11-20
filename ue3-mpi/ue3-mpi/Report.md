# Assignment: MPI

## Exercise 1: Telephone Game

### Initializing MPI

First, the MPI environment must be initialized. The command-line arguments (`argc`, `argv`) allow MPI to process runtime options later on.

`MPI_Comm_rank` returns the `rank` of the calling process within the communicator, so it can be used to identify which process the code is currently run on. `MPI_Comm_size` returns the total number of processes that are available.

```cpp
MPI_Init(&argc, &argv);

int rank, comm_size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD , &comm_size);

const int MAX_LENGTH = 100;
```

### Role of root (Rank 0)

The process with rank 0 is the root process. It is first used to obtain a message from the user for the telephone game. Then, it sends this message to the process with rank 1.

In order to receive the final message, it also listens to a message after sending the first message, which it prints to the console. `MPI_Recv` effectively blocks the process until the final message arrives.

```cpp
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
    ...
}
```

### Role of other ranks

The other processes each reveive the message from the previous process with a lower rank and randomize it by replacing a char. Then they send it to the next process using point-to-point communication.

By doing `(rank + 1) % comm_size` when sending the message to the next process, it is ensured that the process with the highest rank sends the message back to the root process as the final message.

```cpp
if (rank == 0) {
    ...
}
else {
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
```

### Finalizing

`MPI_Finalize` cleans up the MPI environment and must be called by all processes before the program exits.

```cpp
MPI_Finalize();
```

### Output

```sh
PS C:\projects\semester_3\mpv\ue3-mpi\x64\Debug> mpiexec -n 4 .\ue3-mpi.exe
Process 0: Enter a message:
Hello world!
Final Message:
Hmllo worllh
```

## Exercise 2: Hot Potato

### Initializing the game

The MPI environment is again initialized like in the previous exercise, however this time the goal is to implement a Hot Potato game. To generate the initial potato, a seed is prepared for a random start number based on the current time.

Since this time the receiver of each message is also random, a uniform int distribution is also already prepared.

```cpp
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
```

### Role of root (Rank 0)

Rank 0 then generates the potato using the `default_random_engine` that was previously prepared to a number between 20 and 50 and chooses a random receiver

```cpp
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
```

### Receiving the potato

```cpp
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
}
```

### Passing the potato along

```cpp
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
} else if (potato == 0) {
    ...
}
```

### Ending the game

```cpp
if (potato > 0) {
    ...
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

MPI_Finalize();
```

### Output

```sh
PS C:\projects\semester_3\mpv\ue3-mpi\x64\Debug> mpiexec -n 4 .\ue3-mpi.exe
=== Starting hot potato game
=== Generated potato: 39
=== Sending potato to rank: 3
=== Rank 3 received potato: 39
=== Sending potato to rank: 2
=== Rank 2 received potato: 38
=== Sending potato to rank: 0
=== Rank 0 received potato: 37
=== Sending potato to rank: 2
=== Rank 2 received potato: 36
=== Sending potato to rank: 3
=== Rank 3 received potato: 35
=== Sending potato to rank: 1
=== Rank 1 received potato: 34
=== Sending potato to rank: 3
=== Rank 3 received potato: 33
=== Sending potato to rank: 1
=== Rank 1 received potato: 32
=== Sending potato to rank: 3
=== Rank 3 received potato: 31
=== Sending potato to rank: 2
=== Rank 2 received potato: 30
=== Sending potato to rank: 1
=== Rank 1 received potato: 29
=== Sending potato to rank: 0
=== Rank 0 received potato: 28
=== Sending potato to rank: 2
=== Rank 2 received potato: 27
=== Sending potato to rank: 3
=== Rank 3 received potato: 26
=== Sending potato to rank: 0
=== Rank 0 received potato: 25
=== Sending potato to rank: 1
=== Rank 1 received potato: 24
=== Sending potato to rank: 2
=== Rank 2 received potato: 23
=== Sending potato to rank: 3
=== Rank 3 received potato: 22
=== Sending potato to rank: 1
=== Rank 1 received potato: 21
=== Sending potato to rank: 0
=== Rank 0 received potato: 20
=== Sending potato to rank: 1
=== Rank 1 received potato: 19
=== Sending potato to rank: 0
=== Rank 0 received potato: 18
=== Sending potato to rank: 3
=== Rank 3 received potato: 17
=== Sending potato to rank: 0
=== Rank 0 received potato: 16
=== Sending potato to rank: 3
=== Rank 3 received potato: 15
=== Sending potato to rank: 1
=== Rank 1 received potato: 14
=== Sending potato to rank: 2
=== Rank 2 received potato: 13
=== Sending potato to rank: 1
=== Rank 1 received potato: 12
=== Sending potato to rank: 0
=== Rank 0 received potato: 11
=== Sending potato to rank: 2
=== Rank 2 received potato: 10
=== Sending potato to rank: 0
=== Rank 0 received potato: 9
=== Sending potato to rank: 3
=== Rank 3 received potato: 8
=== Sending potato to rank: 0
=== Rank 0 received potato: 7
=== Sending potato to rank: 3
=== Rank 3 received potato: 6
=== Sending potato to rank: 2
=== Rank 2 received potato: 5
=== Sending potato to rank: 1
=== Rank 1 received potato: 4
=== Sending potato to rank: 2
=== Rank 2 received potato: 3
=== Sending potato to rank: 1
=== Rank 1 received potato: 2
=== Sending potato to rank: 0
=== Rank 0 received potato: 1
=== Sending potato to rank: 3
=== Rank 3 received potato: 0
Process 3 has lost
```

## Exercise 3: Parallel Matrix Multiplication

### Part 1: Rows match number of processes

#### Initialization

```cpp
MPI_Init(&argc, &argv);

int rank, comm_size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

int M = 8000000, N = 8;

std::default_random_engine gen{ 42 };
std::uniform_real_distribution<double> dist{ -1.0, +1.0 };
```

#### Broadcast x

```cpp
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
```

#### Scatter A

```cpp
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
```

#### Calculate local y

```cpp
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
```

#### Gather local results to root process

```cpp
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
```

#### Finalize

```cpp
if (rank == 0) {
    std::cout << "Result: " << std::flush;
    for (int i = 0; i < std::min(10, M); i++) {
        std::cout << y[i] << " ";
    }
}

delete[] y;

MPI_Finalize();
```

### Part 2: Rows don't match number of processes

####  Handle scatter with `MPI_Scatterv`

```cpp
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
```
#### Handle gather with `MPI_Gatherv`

```cpp
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
```

## Exercise 4: NBody Simulation