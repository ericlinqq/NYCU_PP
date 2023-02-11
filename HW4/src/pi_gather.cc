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

    unsigned int seed = (unsigned int) time(NULL) * (world_rank + 1);
    long long num_of_tosses = (long long) tosses / world_size;
    long long num_in_circle = 0;

    for (long long num = 0; num < num_of_tosses; num++) {
        double x = (double) rand_r(&seed) / RAND_MAX;
        double y = (double) rand_r(&seed) / RAND_MAX;
        if (x*x + y*y <= 1) num_in_circle++;
    }

    // TODO: use MPI_Gather
    long long *buffer;
    if (world_rank == 0)
        buffer = (long long*) malloc(world_size*sizeof(long long));
    MPI_Gather(&num_in_circle, 1, MPI_LONG_LONG, buffer, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);


    if (world_rank == 0)
    {
        // TODO: PI result
        for (int i = 1; i < world_size; i++)
            num_in_circle += buffer[i];
        free(buffer);
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
