#include <iostream>
#include <random>
#include <fstream>
#include <chrono>

struct Vector {
    double X;
    double Y;
    double Z;
};

struct Particle {
    double m;
    Vector s; // position
    Vector v; // velocity
};

const double G = 6.6743e-11;

int n_steps = 100000;
int n_particles = 25;

double delta_t = 1e4; // hours

void generate_initial_state(Particle* particles) {
    std::default_random_engine generator{ 42 };

    std::lognormal_distribution<double> mass_distribution{ std::log(1e25), 0.8 };
    std::uniform_real_distribution<double> position_distribution{ -1e11, +1e11 };
    std::normal_distribution<double> velocity_distribution{ 0.0, 1e2 };

    for (int q = 0; q < n_particles; q++) {
        particles[q].m = mass_distribution(generator);

        particles[q].s.X = position_distribution(generator);
        particles[q].s.Y = position_distribution(generator);
        particles[q].s.Z = position_distribution(generator);

        particles[q].v.X = velocity_distribution(generator);
        particles[q].v.Y = velocity_distribution(generator);
        particles[q].v.Z = velocity_distribution(generator);
    }
}

void calculate_forces_simple(int q, Particle* particles, Vector* forces) {
    forces[q].X = forces[q].Y = forces[q].Z = 0.0;

    for (int k = 0; k < n_particles; k++) {
        if (k == q) continue;

        double dx = particles[q].s.X - particles[k].s.X;
        double dy = particles[q].s.Y - particles[k].s.Y;
        double dz = particles[q].s.Z - particles[k].s.Z;

        double r2 = dx * dx + dy * dy + dz * dz;
        double r1 = sqrt(r2);
        double r3 = r2 * r1;

        double mass_force = particles[k].m / r3;

        forces[q].X += mass_force * dx;
        forces[q].Y += mass_force * dy;
        forces[q].Z += mass_force * dz;
    }

    forces[q].X *= -G;
    forces[q].Y *= -G;
    forces[q].Z *= -G;
}

void update_position(int q, Particle* particles, Vector* forces) {
    particles[q].s.X += particles[q].v.X * delta_t;
    particles[q].s.Y += particles[q].v.Y * delta_t;
    particles[q].s.Z += particles[q].v.Z * delta_t;

    particles[q].v.X += forces[q].X / particles[q].m * delta_t;
    particles[q].v.Y += forces[q].Y / particles[q].m * delta_t;
    particles[q].v.Z += forces[q].Z / particles[q].m * delta_t;
}

void write_header(std::ostream& output) {
    output << "Step;Time;Particle;Position_X;Position_Y;Position_Z;Mass\n";
}

void write_state(std::ostream& output, int step, double time, Particle* particles) {
    for (int q = 0; q < n_particles; q++) {
        output << step << ";" << time << ";" << q << ";"
            << particles[q].s.X << ";" << particles[q].s.Y << ";" << particles[q].s.Z << ";"
            << particles[q].m << "\n";
    }
}

void write_header_stats(std::ostream& output) {
    output << "Version;Implementation;Thread Number;Trial;Time (ms);Avg Sequential Time (ms);Speedup (x);Efficiency (%)\n";
}

void write_run_stats(std::ostream& output, bool parallel, int thread_num, int trial, double time) {
    const std::string version = "Simple";
    const std::string impl = parallel ? "parallel" : "sequential";

    output << version << ';'
        << impl << ';'
        << thread_num << ';'
        << trial << ';'
        << time << ';' << '\n';
}

void write_run_stats_summary(std::ostream& output, bool parallel, int thread_num, double avgSeqTime) {
    const std::string version = "Simple";
    const std::string impl = parallel ? "parallel" : "sequential";

    output << version << ';'
        << impl << ';'
        << thread_num << ';'
        << ';'
        << ';'
        << avgSeqTime
        << '\n';
}

void run_simulation_simple() {
    auto* particles = new Particle[n_particles];
    generate_initial_state(particles);

    std::ofstream out_file{ "n_body_simple.csv", std::ios::out };
    write_header(out_file);
    write_state(out_file, 0, 0.0, particles);

    auto* forces = new Vector[n_particles];

    for (int step = 1; step <= n_steps; step++) {
        double current_time = step * delta_t;

        // calculate forces
        for (int q = 0; q < n_particles; q++) {
            calculate_forces_simple(q, particles, forces);
        }

        // position updates
        for (int q = 0; q < n_particles; q++) {
            update_position(q, particles, forces);
        }

        // write results
        if (step % 100 == 0) {
            write_state(out_file, step, current_time, particles);
        }
    }

    delete[] particles;
    delete[] forces;
}

void run_simulation_simple_parallel(int thread_num) {
    auto* particles = new Particle[n_particles];
    generate_initial_state(particles);

    std::ofstream out_file{ "n_body_simple.csv", std::ios::out };
    write_header(out_file);
    write_state(out_file, 0, 0.0, particles);

    auto* forces = new Vector[n_particles];

    for (int step = 1; step <= n_steps; step++) {
        double current_time = step * delta_t;

        // calculate forces
#pragma omp parallel for num_threads(thread_num)
        for (int q = 0; q < n_particles; q++) {
            calculate_forces_simple(q, particles, forces);
        }

        // position updates
        for (int q = 0; q < n_particles; q++) {
            update_position(q, particles, forces);
        }

        // write results
        if (step % 100 == 0) {
            write_state(out_file, step, current_time, particles);
        }
    }

    delete[] particles;
    delete[] forces;
}

double measure_time(bool parallel, int thread_num) {
    auto start = std::chrono::high_resolution_clock::now();

        if (parallel)
            run_simulation_simple_parallel(thread_num);
        else
            run_simulation_simple();

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void run_experiment(bool parallel, int thread_num, int trials, std::ostream& output) {
    double total_time = 0.0;

    for (int i = 0; i < trials; i++) {
        double time_ms = measure_time(parallel, thread_num);
        write_run_stats(output, parallel, thread_num, i + 1, time_ms);
        total_time += time_ms;
    }

    double average_time = total_time / trials;
    write_run_stats_summary(output, parallel, thread_num, average_time);
}

int main__nbody() {
    constexpr int trials = 10;

    std::ofstream out_file{ "n_body_stats.csv", std::ios::out };
    write_header_stats(out_file);

    std::cout << "=== Running Sequential Simple Version ===" << std::endl;
    run_experiment(false, 1, trials, out_file);

    //for (int numThreadsValues[] = { 2, 4, 8 }; const int numThreads : numThreadsValues) {
    //    std::cout << std::format("=== Running Parallel Simple Version ({} threads) ===\n", numThreads);
    //    run_experiment(true, numThreads, trials, out_file);
    //}

    return 0;
}
