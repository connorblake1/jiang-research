#include <unistd.h>
#include <stdio.h>
#include <time.h>
int main(int argc, char *argv[]) {
	struct timespec tstart, tstop, tresult;
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	sleep(10);
	clock_gettime(CLOCK_MONOTONIC, &tstop);
	tresult.tv_sec = tstop.tv_sec - tstart.tv_sec;
	tresult.tv_nsec = tstop.tv_nsec - tstart.tv_nsec;
	printf("Elapsed: %f secs \n", (double)tresult.tv_sec + (double)tresult.tv_nsec*1.0e-9);
}
