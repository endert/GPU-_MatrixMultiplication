#pragma once

template <class T>
class Matrix
{
public:
	Matrix<T>(int rows, int cols);
	Matrix(const Matrix<T>& rhs);
	//Matrix& operator=(const Matrix& rhs);
	~Matrix();
	void printMatrix();
	void fillMatrix();
	Matrix<T> multiplication(Matrix<T> A);
	int getTotalSize();
	int getRows();
	int getCols();
	int* m_ptValues;

private:
	int m_rows;
	int m_cols;
	int m_totalSize;
	
};



#include <String>
#include <iostream>
#include <time.h>
#include <stdlib.h>

#define DEBUG
template <class T>
Matrix<T>::Matrix<T>(int rows, int cols) : m_rows(rows), m_cols(cols)
{
	m_totalSize = m_rows*m_cols;
	m_ptValues = new int[m_totalSize]();
}

template <class T>
Matrix<T>::Matrix(const Matrix<T>& rhs) : m_rows(rhs.m_rows), m_cols(rhs.m_cols) {
	m_totalSize = rhs.m_totalSize;
	m_ptValues = new int[rhs.m_totalSize]();
	std::memcpy(m_ptValues, rhs.m_ptValues, rhs.m_totalSize * sizeof(int));
}

/*Matrix& Matrix::operator=(const Matrix& rhs) {
if (&rhs == this) {
return *this;
}
if (m_totalSize == rhs.m_totalSize) {
std::memcpy(m_ptValues, rhs.m_ptValues, rhs.m_totalSize * sizeof(int));
}
else {
delete[] m_ptValues;
m_ptValues = new int[rhs.m_totalSize]();
std::memcpy(m_ptValues, rhs.m_ptValues, rhs.m_totalSize * sizeof(int));
}
m_rows = rhs.m_rows;
m_cols = rhs.m_cols;
m_totalSize = rhs.m_totalSize;

return *this;
}*/

template <class T>
Matrix<T>::~Matrix()
{
	delete[] m_ptValues;
}

template <class T>
void Matrix<T>::printMatrix()
{
#ifdef DEBUG
	std::cout << "Cols: " << m_cols << std::endl;
	std::cout << "Rows: " << m_rows << std::endl;
#endif

	for (int i = 0; i < m_rows; ++i)
	{
		for (int j = 0; j < m_cols; ++j)
		{
			std::cout << m_ptValues[i*m_cols + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

template <class T>
void Matrix<T>::fillMatrix()
{
	srand(clock());
	T r;
	for (int i = 0; i < m_totalSize; ++i)
	{
		r = rand() % 10;
		m_ptValues[i] = r;
	}
}

template <class T>
Matrix<T> Matrix<T>::multiplication(Matrix<T> B)
{
	if (m_cols != B.m_rows)
	{
		throw std::length_error("False Matrix size! Can't mulitply.");
	}

	Matrix<T> C(m_rows, B.m_cols);
	for (int i = 0; i < C.m_cols; ++i)
	{
		for (int j = 0; j < C.m_rows; ++j)
		{
			for (int k = 0; k < m_cols; ++k)
			{
				C.m_ptValues[i*C.m_rows + j] += m_ptValues[i*m_rows + k] * B.m_ptValues[k*B.m_rows + j];
			}
		}
	}

	return C;
}

template <class T>
int Matrix<T>::getTotalSize()
{
	return m_totalSize;
}

template <class T>
int Matrix<T>::getRows()
{
	return m_rows;
}

template <class T>
int Matrix<T>::getCols()
{
	return m_cols;
}
