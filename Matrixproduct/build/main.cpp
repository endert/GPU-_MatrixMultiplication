#include <iostream>
#include <string>
#include "Matrix.h"


int main()
{
	int i = 3, j = 3, k = 3;

	Matrix<float> A(i, j);
	Matrix<float> B(j, k);
	
	A.fillMatrix();
	A.printMatrix();

	B.fillMatrix();
	B.printMatrix();

	Matrix<float> C(A.multiplication(B));
	C.printMatrix();

	return 0;
}