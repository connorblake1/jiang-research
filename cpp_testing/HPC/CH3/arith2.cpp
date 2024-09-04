#include <iostream>
int main(int argc, char *argv[]) {
	int n = 1000;
	int *mem = (int *) malloc(8*n);
	for (int i = 0; i < n; i+= 10) {
		mem[i] = i*i;
	}
	std::cout << "Done (2)" << std::endl;
}
