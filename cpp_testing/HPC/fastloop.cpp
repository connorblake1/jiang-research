#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
int main(int argc, char *argv[]) {
	int imax = 10000;
	int jmax = 10000;
	struct timespec tstart, tmid, tstop, tresultA, tresultB;
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	double **x = (double **)malloc(jmax*sizeof(double *));
	x[0] = (double *)malloc(jmax*imax*sizeof(double));
	for (int j = 1; j < jmax; j++) {
		x[j] = x[j-1] + imax;
	}
	clock_gettime(CLOCK_MONOTONIC, &tmid);
	for (int j = 0; j < jmax; j++) {
		for (int i = 0; i < imax; i++) {
			x[j][i] = j*i;
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &tstop);
	free(x[0]);
	free(x);
	tresultA.tv_sec = tmid.tv_sec - tstart.tv_sec;
	tresultA.tv_nsec = tmid.tv_nsec - tstart.tv_nsec;
	tresultB.tv_sec = tstop.tv_sec - tmid.tv_sec;
	tresultB.tv_nsec = tstop.tv_nsec - tmid.tv_nsec;
	printf("Allocation Time: %f secs \n", (double)tresultA.tv_sec + (double)tresultA.tv_nsec*1.0e-9);
	printf("Execution Time: %f secs \n", (double)tresultB.tv_sec + (double)tresultB.tv_nsec*1.0e-9);
}
