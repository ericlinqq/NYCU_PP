#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

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
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status[world_size-1];
    int tag = 0;

    unsigned int seed = (unsigned int) time(NULL) * (world_rank + 1);
    long long num_of_tosses = (long long) tosses / world_size;
    long long num_in_circle = 0;

    for (long long num = 0; num < num_of_tosses; num++) {
        double x = (double) rand_r(&seed) / RAND_MAX;
        double y = (double) rand_r(&seed) / RAND_MAX;
        if (x*x + y*y <= 1) num_in_circle++;
    }

    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Send(&num_in_circle, 1, MPI_LONG_LONG, 0, tag, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size-1];
        long long buffer[world_size-1];
        for (int i = 1; i < world_size; i++)
            MPI_Irecv(&buffer[i-1], 1, MPI_LONG_LONG, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &requests[i-1]);

        MPI_Waitall(world_size - 1, requests, status);

        for (int i = 1; i < world_size; i++)
            num_in_circle += buffer[i-1];
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 * num_in_circle / (double) tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
