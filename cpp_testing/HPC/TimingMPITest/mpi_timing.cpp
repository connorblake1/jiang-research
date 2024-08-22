#include <unistd.h>
#include <stdio.h>
#include <mpi.h>
int main(int argc, char *argv[]) {
	int mype;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	printf("Processor Rank: %d", mype);
	double t1, t2;
	t1 = MPI_Wtime();
	sleep(10);
	t2 = MPI_Wtime();
	if (mype == 0) {
		printf("Elapsed: %f \n", t2-t1);
	}
	MPI_Finalize();

}
