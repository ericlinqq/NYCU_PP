#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
    // Get rank and size
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // MASTER: read the n, m, l from datafile 
    if (world_rank == 0)
        std::cin >> *n_ptr >> *m_ptr >> *l_ptr; 

    // Broadcast n_ptr, m_ptr, l_ptr
    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    *a_mat_ptr = (int *) calloc((*n_ptr) * (*m_ptr), sizeof(int));
    *b_mat_ptr = (int *) calloc((*m_ptr) * (*l_ptr), sizeof(int));

    // MASTER: read the matrix A and B from datafile
    if (world_rank == 0) {
        // read the value of elements
        for (int count = 0; count < (*n_ptr) * (*m_ptr); count++)
            std::cin >> (*a_mat_ptr)[count]; 

        // read the value of elements
        for (int count = 0; count < (*m_ptr) * (*l_ptr); count++)
            std::cin >> (*b_mat_ptr)[count];
    }

    // assign workload to every process
    int *start = (int *) calloc(world_size, sizeof(int));
    int *end = (int *) calloc(world_size, sizeof(int));

    int max_used_process_num = (*n_ptr < world_size) ? *n_ptr : world_size; // actual amount of processes being used
    const float BLOCK_SIZE = 1.0 * (*n_ptr) / max_used_process_num; // average workload for every process

    for (int i = 0; i < max_used_process_num - 1; i++) {
        start[i] = BLOCK_SIZE * i; // start idx for rank i
        end[i] = BLOCK_SIZE * (i + 1); // end idx for rank i
    }

    start[max_used_process_num - 1] = 
        (max_used_process_num == 1) ? 0 : end[max_used_process_num - 2]; // start idx for the last

    end[max_used_process_num - 1] = *n_ptr; // end idx for the last
    
    // sendcounts: number of elements to send to each processor
    // displs: the displacement from which to take the outgoing data to processor
    int *sendcounts = (int *) calloc(world_size, sizeof(int));
    int *displs = (int *) calloc(world_size, sizeof(int));

    for (int i = 0; i < max_used_process_num; i++) {
        sendcounts[i] = (end[i] - start[i]) * (*m_ptr);
        displs[i] = start[i] * (*m_ptr);
    }
 
    // Scatter matrix A to each process according to the workload assignment
    MPI_Scatterv(*a_mat_ptr, sendcounts, displs, MPI_INT, *a_mat_ptr,
                 sendcounts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast matrix B to every process
    MPI_Bcast(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, 0, MPI_COMM_WORLD);
}               

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    // Get rank and size
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // assign workload to every process
    int *start = (int *) calloc(world_size, sizeof(int));
    int *end = (int *) calloc(world_size, sizeof(int));

    int max_used_process_num = (n < world_size) ? n : world_size; // actual amount of processes being used
    const float BLOCK_SIZE = 1.0 * n / max_used_process_num; // average workload for every process

    for (int i = 0; i < max_used_process_num - 1; i++) {
        start[i] = BLOCK_SIZE * i; // start idx for rank i
        end[i] = BLOCK_SIZE * (i + 1); // end idx for rank i
    }

    start[max_used_process_num - 1] = 
        (max_used_process_num == 1) ? 0 : end[max_used_process_num - 2]; // start idx for the last

    end[max_used_process_num - 1] = n; // end idx for the last

    // Compute the matrix C
    int *c_mat = (int *) calloc(n * l, sizeof(int));

    if (world_rank < max_used_process_num) {
        for (int i = 0; i < end[world_rank] - start[world_rank]; i++) {
            for (int k = 0; k < m; k++) {
                for (int j = 0; j < l; j++)
                    c_mat[i * l + j] += a_mat[i * m + k] * b_mat[k * l + j]; 
            }
        }
    }

    // sendcounts: number of elements to send to each processor
    // displs: the displacement from which to take the outgoing data to processor
    int *sendcounts = (int *) calloc(world_size, sizeof(int));
    int *displs = (int *) calloc(world_size, sizeof(int));
    
    for (int i = 0; i < max_used_process_num; i++) {
        sendcounts[i] = (end[i] - start[i]) * l;
        displs[i] = start[i] * l;
    }

    // Gather computation result from each process according to the workload assignment
    MPI_Gatherv(c_mat, sendcounts[world_rank], MPI_INT, c_mat, sendcounts, displs,
                MPI_INT, 0, MPI_COMM_WORLD);

    // Master: print matrix C
    if (world_rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++)
                printf("%d ", c_mat[i * l + j]);
            printf("\n");
        }
    }

    free(c_mat);
}

void destruct_matrices(int *a_mat, int *b_mat) {
    free(a_mat);
    free(b_mat);
}