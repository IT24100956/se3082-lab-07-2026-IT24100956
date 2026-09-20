CC = mpicc
CFLAGS = -Wall -O2
NP = 4

TARGETS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

.PHONY: all run clean

all: $(TARGETS)

sum_bcast: sum_bcast.c
	$(CC) $(CFLAGS) -o $@ $<

sum_scatter: sum_scatter.c
	$(CC) $(CFLAGS) -o $@ $<

sum_gather: sum_gather.c
	$(CC) $(CFLAGS) -o $@ $<

sum_reduce: sum_reduce.c
	$(CC) $(CFLAGS) -o $@ $<

sum_allreduce: sum_allreduce.c
	$(CC) $(CFLAGS) -o $@ $<

sum_scan: sum_scan.c
	$(CC) $(CFLAGS) -o $@ $<

run: all
	@echo "=========================================="
	@echo "Running Exercise 1: Broadcast + Send/Recv"
	@echo "=========================================="
	mpirun -np $(NP) ./sum_bcast
	@echo ""
	@echo "=========================================="
	@echo "Running Exercise 2: Scatter + Send/Recv"
	@echo "=========================================="
	mpirun -np $(NP) ./sum_scatter
	@echo ""
	@echo "=========================================="
	@echo "Running Exercise 3: Scatter + Gather"
	@echo "=========================================="
	mpirun -np $(NP) ./sum_gather
	@echo ""
	@echo "=========================================="
	@echo "Running Exercise 4: Scatter + Reduce"
	@echo "=========================================="
	mpirun -np $(NP) ./sum_reduce
	@echo ""
	@echo "=========================================="
	@echo "Running Exercise 5: Scatter + Allreduce"
	@echo "=========================================="
	mpirun -np $(NP) ./sum_allreduce
	@echo ""
	@echo "=========================================="
	@echo "Running Exercise 6: Scatter + Scan"
	@echo "=========================================="
	mpirun -np $(NP) ./sum_scan
	@echo ""

clean:
	rm -f $(TARGETS)
