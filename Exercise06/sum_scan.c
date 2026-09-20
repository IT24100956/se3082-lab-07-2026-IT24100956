#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;

    /* Only root allocates the full array */
    int *array = NULL;
    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        if (array == NULL) {
            fprintf(stderr, "Root failed to allocate full array\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    /* All processes allocate only their small local chunk */
    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));
    if (local_chunk == NULL) {
        fprintf(stderr, "Rank %d failed to allocate local_chunk\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double start = MPI_Wtime();

    /* Distribute chunks using Scatter */
    MPI_Scatter(array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Local summation */
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    /*
     * SCAN: Prefix reduction. Each rank receives the cumulative sum of all
     * contributions from rank 0 through its own rank.
     */
    long long prefix_sum = 0;
    MPI_Scan(&local_sum, &prefix_sum, 1, MPI_LONG_LONG,
             MPI_SUM, MPI_COMM_WORLD);

    /* The sum of all chunks strictly before this process */
    long long sum_before_me = prefix_sum - local_sum;

    /* Verification formula: sum of 1 to K is K*(K+1)/2 */
    long long K = (long long)(rank + 1) * chunk_size;
    long long expected_prefix = K * (K + 1) / 2;
    int is_correct = (prefix_sum == expected_prefix);

    printf("  Rank %d: local_sum = %lld, prefix_sum = %lld, sum_before_me = %lld [Check: %s]\n",
           rank, local_sum, prefix_sum, sum_before_me, is_correct ? "PASSED" : "FAILED");

    /* Last rank verifies against the grand total (sum of 1 to N) */
    if (rank == size - 1) {
        double elapsed = MPI_Wtime() - start;
        long long expected_total = (long long)N * (N + 1) / 2;
        printf("\n[Scan] Last Rank (%d) prefix_sum = %lld\n", rank, prefix_sum);
        printf("[Scan] Expected Total          = %lld\n", expected_total);
        printf("[Scan] Correct?                = %s\n", prefix_sum == expected_total ? "YES" : "NO");
        printf("[Scan] Time                    = %.4f sec\n", elapsed);
    }

    if (rank == 0) free(array);
    free(local_chunk);
    MPI_Finalize();
    return 0;
}
