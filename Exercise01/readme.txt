================================================================================
SE3082 - Parallel Computing | Lab 07
Exercise 01: Broadcast + Send/Recv (Given Program)
================================================================================

1. OBJECTIVE & STRATEGY:
--------------------------------------------------------------------------------
This is the baseline program provided in the lab specification.
Root (rank 0) fills the array with integers from 1 to 1,000,000 and broadcasts
the ENTIRE array to all processes using MPI_Bcast.
Each process computes the sum of its own portion (N / size elements).
Non-root processes send their partial sums back to root using MPI_Send.
Root receives and accumulates the partial sums using a loop over MPI_Recv.

Key Observations & Inefficiencies:
- Every process allocates the full 1,000,000-element array (4 MB each).
  Across P processes, this consumes 4 * P MB of memory instead of just 4 MB total.
- Root receives from each process sequentially one by one in an O(P) linear loop.

2. SOURCE CODE (sum_bcast.c):
--------------------------------------------------------------------------------
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *array = (int *)malloc(N * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Rank %d failed to allocate array\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    double start = MPI_Wtime();

    MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = N / size;
    int start_idx = rank * chunk_size;
    int end_idx = start_idx + chunk_size;

    long long local_sum = 0;
    for (int i = start_idx; i < end_idx; i++)
        local_sum += array[i];

    printf("  Rank %d: summed indices [%d, %d) => local_sum = %lld\n",
           rank, start_idx, end_idx, local_sum);

    if (rank != 0) {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    } else {
        long long total_sum = local_sum;
        for (int r = 1; r < size; r++) {
            long long recv_sum;
            MPI_Recv(&recv_sum, 1, MPI_LONG_LONG, r, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += recv_sum;
        }
        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Bcast] Total sum   = %lld\n", total_sum);
        printf("[Bcast] Expected    = %lld\n", expected);
        printf("[Bcast] Correct?    = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Bcast] Time        = %.4f sec\n", elapsed);
    }

    free(array);
    MPI_Finalize();
    return 0;
}

3. COMPILATION & EXECUTION:
--------------------------------------------------------------------------------
Command:
  mpicc -Wall -O2 -o sum_bcast sum_bcast.c
  mpirun -np 4 ./sum_bcast

Sample Output (4 processes):
  Root filled array with values 1 to 1000000
    Rank 0: summed indices [0, 250000) => local_sum = 31250125000
    Rank 1: summed indices [250000, 500000) => local_sum = 93750125000
    Rank 2: summed indices [500000, 750000) => local_sum = 156250125000
    Rank 3: summed indices [750000, 1000000) => local_sum = 218750125000

  [Bcast] Total sum   = 500000500000
  [Bcast] Expected    = 500000500000
  [Bcast] Correct?    = YES
  [Bcast] Time        = 0.0051 sec
