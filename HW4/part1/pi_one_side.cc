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

int fnz(ll *schedule, ll *oldschedule, int size)
{
    int diff = 0;

    for (int i = 0; i < size; i++)
        diff |= (schedule[i] != oldschedule[i]);

    if (diff) {
        ll res = 0;
        for (int i = 0; i < size; i++) {
            res += (schedule[i] != 0);
            oldschedule[i] = schedule[i];
        }
        return (res == size - 1);
    }
    return 0;
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

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    ll avg_tosses = (tosses + world_size - 1) / world_size;
    ll local_sum = monte_carlo(avg_tosses, world_rank);

    if (world_rank == 0) {
        // Master
        ll *old_schedule = new ll[world_size];
        ll *schedule;
        MPI_Alloc_mem(sizeof(ll) * world_size, MPI_INFO_NULL, &schedule);

        for (int i = 0; i < world_size; i++) {
            schedule[i] = 0;
            old_schedule[i] = -1;
        }

        MPI_Win_create(schedule, world_size * sizeof(ll), sizeof(ll), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        int ready = 0;
        while (!ready) {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            ready = fnz(schedule, old_schedule, world_size);
            MPI_Win_unlock(0, win);
        }

        for (int i = 1; i < world_size; i++)
            local_sum += schedule[i];

        MPI_Free_mem(schedule);
        free(old_schedule);
    }
    else {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&local_sum, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0) {
        // TODO: handle PI result
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