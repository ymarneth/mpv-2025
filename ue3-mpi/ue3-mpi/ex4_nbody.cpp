#include <iostream>
#include <random>
#include <fstream>
#include <chrono>
#include <mpi.h>

struct Vector {
    double X;
    double Y;
    double Z;
};

const double G = 6.6743e-11;

int n_steps = 100000;
int n_particles = 25;

double delta_t = 1e4; // hours

MPI_Datatype mpi_vector_type;
MPI_Datatype mpi_particle_type;

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

void compute_local_forces(int global_index, const Vector* global_positions, const double* masses, Vector& forces, int n_total) {
    forces.X = forces.Y = forces.Z = 0.0;

    const Vector& pos_i = global_positions[global_index];

    for (int k = 0; k < n_total; ++k) {
        if (k == global_index) continue;

        const Vector& pos_k = global_positions[k];

        double dx = pos_i.X - pos_k.X;
        double dy = pos_i.Y - pos_k.Y;
        double dz = pos_i.Z - pos_k.Z;

        double r2 = dx * dx + dy * dy + dz * dz;
        double r1 = std::sqrt(r2);
        double r3 = r2 * r1 + 1e-30; // avoid division by zero

        double mass_force = masses[k] / r3;

        forces.X += mass_force * dx;
        forces.Y += mass_force * dy;
        forces.Z += mass_force * dz;
    }

    forces.X *= -G;
    forces.Y *= -G;
    forces.Z *= -G;
}

void write_header(std::ostream& output) {
    output << "Step;Time;Particle;Position_X;Position_Y;Position_Z;Mass\n";
}

void write_state(std::ostream& output, int step, double time, const Vector* positions, const double* masses, int n) {
    for (int q = 0; q < n; q++) {
        output << step << ";" << time << ";" << q << ";"
            << positions[q].X << ";" << positions[q].Y << ";" << positions[q].Z << ";"
            << masses[q] << "\n";
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    build_vector_type();

    int rank = 0, comm_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

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

    // global buffers (only filled on rank 0).
    Vector* global_positions = new Vector[n_particles];
    Vector* global_velocities = new Vector[n_particles];
    double* masses = new double[n_particles];

    // local buffers
    Vector* local_positions = new Vector[local_n];
    Vector* local_velocities = new Vector[local_n];
    Vector* local_forces = new Vector[local_n];

    std::ofstream out_file{ "n_body_simple_mpi.csv", std::ios::out };
    if (rank == 0) {
        write_header(out_file);
        
        generate_initial_state(global_positions, global_velocities, masses, n_particles);
        
        write_state(out_file, 0, 0.0, global_positions, masses, n_particles);
    }

    MPI_Bcast(masses, n_particles, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // broadcast initial global_positions and velocities so every rank can compute forces in first step
    MPI_Bcast(global_positions, n_particles, mpi_vector_type, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_velocities, n_particles, mpi_vector_type, 0, MPI_COMM_WORLD);

    // copy initial local slices from global arrays (local view)
    MPI_Scatterv(global_positions, counts, displs, mpi_vector_type,
        local_positions, counts[rank], mpi_vector_type, 0, MPI_COMM_WORLD);

    MPI_Scatterv(global_velocities, counts, displs, mpi_vector_type,
        local_velocities, counts[rank], mpi_vector_type, 0, MPI_COMM_WORLD);

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

        // Gather results on root
        MPI_Allgatherv(local_positions, counts[rank], mpi_vector_type,
            global_positions, counts, displs, mpi_vector_type, MPI_COMM_WORLD);

        MPI_Allgatherv(local_velocities, counts[rank], mpi_vector_type,
            global_velocities, counts, displs, mpi_vector_type, MPI_COMM_WORLD);

        if (rank == 0 && (step % 100 == 0)) {
            write_state(out_file, step, current_time, global_positions, masses, n_particles);
        }
    }

    delete[] counts;
    delete[] displs;
    delete[] local_positions;
    delete[] local_velocities;
    delete[] local_forces;
    delete[] global_positions;
    delete[] global_velocities;
    delete[] masses;

    MPI_Type_free(&mpi_vector_type);
    MPI_Finalize();
    
    return 0;
}
