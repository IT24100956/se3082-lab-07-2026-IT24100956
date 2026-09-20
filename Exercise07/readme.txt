
1. COMPARISON TABLE:
--------------------------------------------------------------------------------

| Program Name     | Collectives Used             | Memory Allocation Per Process   | Root Needs Manual Sum Loop? | Where Final Result is Available       | Communication Complexity (Data Dist. / Collection) |
|------------------|------------------------------|---------------------------------|-----------------------------|---------------------------------------|-----------------------------------------------------|
| 1. sum_bcast     | MPI_Bcast                    | Every process allocates FULL N  | Yes (manual MPI_Recv loop)  | Root only                             | O(N log P) / O(P) linear point-to-point             |
| 2. sum_scatter   | MPI_Scatter                  | Only root allocates N;          | Yes (manual MPI_Recv loop)  | Root only                             | O(N) linear scatter / O(P) linear point-to-point    |
|                  |                              | others allocate N/P             |                             |                                       |                                                     |
| 3. sum_gather    | MPI_Scatter, MPI_Gather      | Only root allocates N + P;      | Yes (loops over all_sums)   | Root only                             | O(N) / O(P) tree-based gather                       |
|                  |                              | others allocate N/P             |                             |                                       |                                                     |
| 4. sum_reduce    | MPI_Scatter, MPI_Reduce      | Only root allocates N;          | No (computed by MPI_Reduce) | Root only                             | O(N) / O(log P) tree-based reduction                |
|                  |                              | others allocate N/P             |                             |                                       |                                                     |
| 5. sum_allreduce | MPI_Scatter, MPI_Allreduce   | Only root allocates N;          | No (computed by Allreduce)  | All processes                         | O(N) / O(log P) recursive-doubling / butterfly      |
|                  |                              | others allocate N/P             |                             |                                       |                                                     |
| 6. sum_scan      | MPI_Scatter, MPI_Scan        | Only root allocates N;          | No (computed by MPI_Scan)   | Different per rank (prefix running sum| O(N) / O(log P) parallel prefix tree                |
|                  |                              | others allocate N/P             |                             | up to each rank; total on last rank)  |                                                     |


2. EXECUTION TIME BENCHMARKS (2, 4, AND 8 PROCESSES):
--------------------------------------------------------------------------------
Benchmark tests were performed across all 6 implementations using 1,000,000 integers.
Execution times (in seconds) averaged across multiple runs:

+------------------+---------------------+---------------------+---------------------+
| Implementation   | np = 2              | np = 4              | np = 8              |
+------------------+---------------------+---------------------+---------------------+
| 1. sum_bcast     | 0.0062 s            | 0.0051 s            | 0.0078 s            |
| 2. sum_scatter   | 0.0039 s            | 0.0028 s            | 0.0034 s            |
| 3. sum_gather    | 0.0031 s            | 0.0022 s            | 0.0025 s            |
| 4. sum_reduce    | 0.0026 s            | 0.0018 s            | 0.0019 s            |
| 5. sum_allreduce | 0.0027 s            | 0.0019 s            | 0.0021 s            |
| 6. sum_scan      | 0.0028 s            | 0.0019 s            | 0.0022 s            |
+------------------+---------------------+---------------------+---------------------+



