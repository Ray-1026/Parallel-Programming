#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define ll long long
const double OFFSET = 2.0 / RAND_MAX;

ll monte_carlo(ll tosses, unsigned int my_rank)
{
    unsigned int local_seed = time(NULL) + my_rank;

    ll local_sum = 0;
    for (ll t = 0; t < tosses; t++) {
        double x = rand_r(&local_seed) * OFFSET - 1;
        double y = rand_r(&local_seed) * OFFSET - 1;
        if (x * x + y * y <= 1.0)
            local_sum++;
    }

    return local_sum;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    ll avg_tosses = (tosses + world_size - 1) / world_size;
    ll local_sum = monte_carlo(avg_tosses, world_rank);

    if (world_rank > 0) {
        // TODO: MPI workers
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0) {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size - 1];
        MPI_Status stats[world_size - 1];
        ll recv_sums[world_size - 1];

        for (int i = 1; i < world_size; i++)
            MPI_Irecv(&recv_sums[i - 1], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &requests[i - 1]);

        MPI_Waitall(world_size - 1, requests, stats);

        for (int i : recv_sums)
            local_sum += i;
    }

    if (world_rank == 0) {
        // TODO: PI result
        pi_result = 4.0 * local_sum / static_cast<double>(tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
