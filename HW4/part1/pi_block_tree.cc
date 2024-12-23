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
    ll local_sum = monte_carlo(avg_tosses, world_rank), recv_sum;

    // TODO: binary tree redunction
    for (int power = 2; power <= world_size; power <<= 1) {
        if (world_rank & (power - 1)) {
            MPI_Send(&local_sum, 1, MPI_LONG_LONG, world_rank - (power >> 1), 0, MPI_COMM_WORLD);
            break;
        }

        MPI_Recv(&recv_sum, 1, MPI_LONG_LONG, world_rank + (power >> 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_sum += recv_sum;
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
