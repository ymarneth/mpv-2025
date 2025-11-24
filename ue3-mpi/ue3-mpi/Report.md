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

Rank 0 then generates the potato using the `default_random_engine` that was previously prepared to a number between 20 and 50 and chooses a random receiver using `MPI_Send`. To ensure the potato is not sent the same rank, it is wrapped in a do-while loop.

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

Then, all processes, including rank 0, wait to receive the potato. Since the reveiver of the patato is randomly chosen, `MPI_ANY_SOURCE` is used for the `from` property. The `MPI_Recv` is also wrapped in a while-loop to keep the processes ready to receive the potato multiple times in one game. This is controlled by the `game_running` variable. If the potato is sent with a value of `-1` the game is ended and the loop is broken. -1 was chosen as the end signal and ensures all processes stop waiting to receive the potato.

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

However, as long as the potato value is greater than 0, the receiver simply reduces the value by 1 and passes it along to the next randomly chosen process. This happens in the same manner as in the beginning for process 0.

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

As soon as the potato value reaches 0, the loser of the Hot Potato game is announced, the `game_running` is set to false and the ending signal is sent to all other processes, except to the rank itself, allowing the program to gracefully stop. `MPI_Finalize` is called last to ensure that the MPI environment is cleaned up.

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

In this first version of the matrix-vector multiplication, the total number of matrix rows matches the number of MPI processes, allowing the program to evenly distribute the the rows to the processes. Each process receives exactly the same number of rows.

#### Initialization

As in the previous exercises, the MPI environment is initialized before any communication can be established. When this is done, the dimensions of the matrix and vector are then defined. Here, `M` is the number of rows of the matrix and is set to a very large value, while `N`, the number of columns, is very small. This setup creates a tall and narrow matrix, allowing MPI to optimize the runtime effectively.

To generate reproducible data, a fixed seed is used for the random-number generator. A uniform real distribution in the range `[-1.0, +1.0]` is chosen for both the matrix entries and the vector elements.

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

Before the multiplication process can begin, every process needs to be able to access the same vector `x`. Since only the root process with rank 0 generates the vector, it must be distributed to all other processes using `MPI_Bcast`. After this call completes, every process holds an identical copy of the vector, ensuring that the local matrix-vector multiplications performed later are consistent.

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

The matrix `A` must be divided among the available processes so that each process computes only a portion of the final result. In this first version, the total number of rows `M` is chosen to divide evenly among the numer of process, so every process gets the same number of rows.

The root process begins by generating the full matrix of `A` and fills it with random values. All other processes leave their pointar as `nullptr`, since they will only receive their portion of the matrix.

The number of rows each process should handle is computed as `rows_perprocessor = M / comm_size`. Then each process allocates a local buffer called `local_A`, which is large enough to store its share of the matrix. Since each row contains `N` entries, the local buffer size is `rows_per_processor * N`.

The actual distribution of the matrix is performed via `MPI_Scatter`, which sends contiguous blocks of data from the root process to every other process, including root itself. After scattering, the root process no longer needs the full matrix and can free the memory.

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

Once each process has received the necessary data, it can begin computing its part of the final output vector `y`. A local result array `local_y` is allocated with one entry for each row assigned to the process. The matrix-vector multiplication is then performed row by row. For every local row, a dot product between the corresponding matrix row and the vector `x` is computed.

After the computation is finished, and since the matrix slice `local_A` and the broadcasted vector `x` are no longer needed, both buffers are freed.

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

After every process computes its partial result, the final step is to collect these partial vectors into a single output vector on the root process. This is done using `MPI_Gather`, the counterpart of `MPI_Scatter`.

The root process first allocates enough memory to hold the entire result vector `y`. Then, `MPI_Gather` collects each `local_y` buffer and places them in the correct order in `y`. After the gather operation is done, each process frees its `local_y` buffer. Only the root process keeps the full vector `y`.

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

In the end, the root process prints a small subset of the result vector as a quick confirmation that the operation produced useful output. Printing the entire result would be excessive for matrices of this size.

Once printing is complete, the root process frees the memory used for the full result vector. All processes then call `MPI_Finalize` to cleanly shut down the MPI environment.

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

Since the number of rows `M` in the matrix `A` may not always divide evenly between the number of available processes, the simple `MPI_Scatter` approach is insufficient since it requires that every process receives the same number of elements. To support any matrix size, a more flexible scatter operation is required.

####  Handle scatter with `MPI_Scatterv`

The approach begins by computing how many rows each process should receive. Most processes get `base_rows = M / comm_size`, while the first `remainder_rows = M % comm_size` processes reveive one additional row each. This ensures that the total number of rows is distributed as evenly as possible.

To effectively scatter this, `MPI_Scatterv` is used in place of the simple `MPI_Scatter`. This requires three arrays to be constructed: `sendcounts` defines how many elements each process receives, `displs` where each block starts within the original matrix and `rows_for_process` the number of rows assigned to each process.

`MPI_Scatterv` then distributes the parts of the amtrix accordingly. Unlike the basic scatter, this function allows each process to receive a differtly sized segment of data.

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

After each process has computed its portion of the output vector, the partial results must again be assembled back into the final result array `y` on the root process. Since the number of rows assigned to each process may differ, the basic `MPI_Gather` can also not be used here, so `MPI_Gatherv` is used here.

It needs two arrays to define how the results should be placed in the final output: `recvcounts_y` defines how many output entries ach process contributes and `displs_y` the position where each block of results should be stored. These arrays are build based on `rows_for_process`, ensuring that the gathered values appear in the correct order in the global result vector.

After gathering, the temporary buffers and bookkeeping arrays are freed.

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

In the last assignment the N-body simulation was implemented using OpenMP with shared memory. Parts of this code are reused here to implement the simple version of the simulation with MPI. This report will highlight differences between the previous version and how MPI integrates in this reworked implementation.

### Replacing the Particle struct and creating a MPI Dataype for `Vector`

Previously, the simulation stored particle data like position, velocity and mass in a single struct like this:

```cpp
struct Particle {
    Vector Position;
    Vector Velocity;
    double Mass;
};
```

For the MPI implementaiton, this was removed since MPI cannot reliably transmit arbitrary C++ structs. Instead, ithe simulation now stores the three properties in separate arrays:

- `Vector positions[]`
- `Vector velocities[]`
- `double masses[]`

For the `Vector` struct, a custom MPI datatype was created for the Vector struct which originally simply stored `x`, `y` and `z` coordinates like this:

```cpp
struct Vector {
    double X;
    double Y;
    double Z;
};
```

Because MPI needs to understand the layout of the transmitted data, a derived MPI datatype is created for the `Vector`. This datatype describes the struct field addresses and ensures arrays of `Vector` behave correctly during MPI communication operations. The implementation follows the code from the hint in the assignment slide:

```cpp
struct Vector {
  double X;
  double Y;
  double Z;
};

MPI_Datatype mpi_vector_type;

void build_vector_type() {
  int blocklens[3] = { 1, 1, 1 };
  MPI_Aint disps[3];
  MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

  Vector probe{};
  MPI_Aint base;
  MPI_Get_address(&probe, &base);
  MPI_Get_address(&probe.X, &disps[0]);
  MPI_Get_address(&probe.Y, &disps[1]);
  MPI_Get_address(&probe.Z, &disps[2]);
  disps[0] -= base; disps[1] -= base; disps[2] -= base;

  MPI_Datatype tmp;
  MPI_Type_create_struct(3, blocklens, disps, types, &tmp);
  // Ensure arrays of Vector advance by sizeof(Vector)
  MPI_Type_create_resized(tmp, 0, sizeof(Vector), &mpi_vector_type);
  MPI_Type_commit(&mpi_vector_type);
  MPI_Type_free(&tmp);
}
```

With this MPI type, arrays of Vector now can safely be sent using MPI collective functions like `MPI_Bcast` and `MPI_Scatterv`.

### Initialization on Rank 0

Rank 0 is the root process and therefore responsible to initialize all shared ressources. In this simulation it generates the masses, the initial positions and the initial velocities of the particles and writes this initial state to the output file:

```cpp
void generate_initial_state(Vector* positions, Vector* velocities, double* masses, int n) {
    std::default_random_engine generator{ 42 };

    std::lognormal_distribution<double> mass_distribution{ std::log(1e25), 0.8 };
    std::uniform_real_distribution<double> position_distribution{ -1e11, +1e11 };
    std::normal_distribution<double> velocity_distribution{ 0.0, 1e2 };

    for (int q = 0; q < n; q++) {
        masses[q] = mass_distribution(generator);

        positions[q].X = position_distribution(generator);
        positions[q].Y = position_distribution(generator);
        positions[q].Z = position_distribution(generator);

        velocities[q].X = velocity_distribution(generator);
        velocities[q].Y = velocity_distribution(generator);
        velocities[q].Z = velocity_distribution(generator);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    build_vector_type();

    int rank = 0, comm_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    ...

    // global buffers (only filled on rank 0).
    Vector* global_positions = new Vector[n_particles];
    Vector* global_velocities = new Vector[n_particles];
    double* masses = new double[n_particles];

    std::ofstream out_file{ "n_body_simple_mpi.csv", std::ios::out };
    if (rank == 0) {
        write_header(out_file);
        
        generate_initial_state(global_positions, global_velocities, masses, n_particles);
        
        write_state(out_file, 0, 0.0, global_positions, masses, n_particles);
    }

    ...
}
```

### Distribution of Particles

While all processes receive the complete inital system in order to compute the forces in the first timestep, the particle range is then divided into blocks for updating. The distribution follows the same logic as in the exercise about matrix-vector multiplication:

```cpp
// distribute particles to processes
int base_particles = n_particles / comm_size;
int remaining_particles = n_particles % comm_size;
int local_n = base_particles + (rank < remaining_particles ? 1 : 0);

int* counts = new int[comm_size];
int* displs = new int[comm_size];
displs[0] = 0;
for (int p = 0; p < comm_size; ++p) {
    counts[p] = base_particles + (p < remaining_particles ? 1 : 0);
    if (p > 0) displs[p] = displs[p - 1] + counts[p - 1];
}

...

// local buffers
Vector* local_positions = new Vector[local_n];
Vector* local_velocities = new Vector[local_n];
Vector* local_forces = new Vector[local_n];
```

All processes then receive the global initial state via broadcast:

```cpp
MPI_Bcast(masses, n_particles, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// broadcast initial global_positions and velocities so every rank can compute forces in first step
MPI_Bcast(global_positions, n_particles, mpi_vector_type, 0, MPI_COMM_WORLD);
MPI_Bcast(global_velocities, n_particles, mpi_vector_type, 0, MPI_COMM_WORLD);
```

And finally, each process reveives its local slice of their particle positions and velocities using `MPI_Scatterv`:

```cpp
// copy initial local slices from global arrays (local view)
MPI_Scatterv(global_positions, counts, displs, mpi_vector_type,
    local_positions, counts[rank], mpi_vector_type, 0, MPI_COMM_WORLD);

MPI_Scatterv(global_velocities, counts, displs, mpi_vector_type,
    local_velocities, counts[rank], mpi_vector_type, 0, MPI_COMM_WORLD);
```

### Local computation

After initialization and distribution, each process is now responsible for computing the dynamics of its assigned subset of particles. Even though particles are divided among the ranks, every process still requires the full set of global positions and masses in order to compute the gravitational forces between them. This is why all ranks maintain a synchronized copy of the global arrays, while only updating the particles they are responsible for.

Each process iterates its assigned particles and calculates the force acting on a particle. To correctly update a particle, its global index is needed, which can be calculated by simply combining the displacement of the process with the local loop counter like `int global_index = displs[rank] + i`.

The function `compute_local_forces` uses the full `global_positions` and `masses` to compute the gravitational influence of all bodies, but only stores results for the rank's local particles.

```cpp
for (int step = 1; step <= n_steps; ++step) {
    double current_time = step * delta_t;

    // compute forces for local bodies using the current global_positions & masses
    for (int i = 0; i < local_n; ++i) {
        int global_index = displs[rank] + i; // global index
        compute_local_forces(global_index, global_positions, masses, local_forces[i], n_particles);
    }

    // update local velocities and positions for local bodies
    for (int i = 0; i < local_n; ++i) {
        // v += a * dt  ; a = F/m
        local_velocities[i].X += local_forces[i].X / masses[displs[rank] + i] * delta_t;
        local_velocities[i].Y += local_forces[i].Y / masses[displs[rank] + i] * delta_t;
        local_velocities[i].Z += local_forces[i].Z / masses[displs[rank] + i] * delta_t;

        // s += v * dt
        local_positions[i].X += local_velocities[i].X * delta_t;
        local_positions[i].Y += local_velocities[i].Y * delta_t;
        local_positions[i].Z += local_velocities[i].Z * delta_t;
    }

    ...
}
```

### Gather results on root

Once the forces are computed, each process updates the velocity and postion of its local particles. Then, all ranks refresh their copy of the global particle arrays, so that the next force computation can use the latest state using `MPI_Allgatherv`, which gathers the blocks from all processes and distributes the complete result back to everyone.

Only the root process writes the simulation states to the output file for later visualization to avoid duplicated output.

```cpp
for (int step = 1; step <= n_steps; ++step) {
    
    ...

    // Gather results on root
    MPI_Allgatherv(local_positions, counts[rank], mpi_vector_type,
        global_positions, counts, displs, mpi_vector_type, MPI_COMM_WORLD);

    MPI_Allgatherv(local_velocities, counts[rank], mpi_vector_type,
        global_velocities, counts, displs, mpi_vector_type, MPI_COMM_WORLD);

    if (rank == 0 && (step % 100 == 0)) {
        write_state(out_file, step, current_time, global_positions, masses, n_particles);
    }
}
```

### Cleanup

After the simulation computation is complete and all results have been written, each process frees the memory it allocated during the setup phase:

```cpp
delete[] counts;
delete[] displs;
delete[] local_positions;
delete[] local_velocities;
delete[] local_forces;
delete[] global_positions;
delete[] global_velocities;
delete[] masses;
```

Additionally, the custom MPI datatype created for representing the `Vector` struct must be freed:

```cpp
MPI_Type_free(&mpi_vector_type);
```

And lastly, the program terminates the MPI environment with

```cpp
MPI_Finalize();
```
