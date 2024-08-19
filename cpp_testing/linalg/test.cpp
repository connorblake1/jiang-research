#include <iostream>
#include <eigen-3.4.0/Eigen/Dense>
#include <eigen-3.4.0/Eigen/Eigenvalues>
#include <vector>
#include <chrono>
using namespace Eigen;
typedef Matrix<double, 10, 10> Matrixe1;
typedef Matrix<double, 100, 100> Matrixe2;
typedef Matrix<double, Dynamic, Dynamic> Matrixe3;
//typedef Matrix<double, 10000, 10000> Matrixe4;

int main() {  
  const int N = 100;
  auto start = std::chrono::high_resolution_clock::now();
  Matrixe3 matrix = Matrixe3::Zero(N,N);

  for(int i = 0; i < N; ++i) {
      matrix(i, i) = -2;
      if(i > 0) {
          matrix(i, i - 1) = 1;
      }       if(i < N - 1) {
          matrix(i, i + 1) = 1;
      }
  }
  auto mid = std::chrono::high_resolution_clock::now();
  Eigen::EigenSolver<Matrixe3> eigen_solver(matrix);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> allocationTime = mid - start;
  std::chrono::duration<double> duration = end-start;
  std::cout << "\nN:" << N << " mid:" << allocationTime.count() << " " << duration.count();
  std::cout << std::endl;
  return 0;
}
