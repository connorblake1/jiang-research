Running:
	gcc -o timing timing.cpp
	mpic++ -o mpi_timing mpi_timing.cpp
	cd build
		ctest
