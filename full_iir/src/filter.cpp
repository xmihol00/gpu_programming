#include "filter.h"
#include "babymath.h"

#include <assert.h>

template<typename T>
void GenericFilter<T>::generateStateSpace(std::vector<T>& oA, std::vector<T>& oB, std::vector<T>& oC, T& D) const
{
	// state space formulation with u[.]..input, y[.]..output, x..state vector
	// x_{n+1} = A * x_n + B * u[n]
	// y[n] = C * x_n + D * u[n]

	assert(_a[0] == 1);

	size_t M = _size - 1;
	Mat<T> A(M, M, 0);
	for (size_t i = 1; i < M; ++i)
		A(i - 1, i) = 1;
	for (size_t i = 0; i < M; ++i)
		A(M - 1, i) = -_a[M - i];

	Vec<T> B(M, 0);
	B(M - 1) = 1;


	Vec<T> C(M);
	for (size_t i = 0; i < M; ++i)
		C(i) = _b[M - i] - _a[M - i] * _b[0];

	D = _b[0];
	std::swap(oA, A.data);
	std::swap(oB, B.data);
	std::swap(oC, C.data);	
}

template<typename T>
std::vector<T> GenericFilter<T>::generateStateSpaceMatrix(size_t outputs) const
{
	// rely on the state space formulation to create a single matrix that moves the state 
	// forward by #outputs and 


	// get the state space first
	size_t M = _size - 1;
	Mat<T> A(M, M);
	Vec<T> B(M), C(M);
	T D;
	generateStateSpace(A.data, B.data, C.data, D);

	// example for M = 2 and n = 4 parallel outputs 

	// An -> A**n .. state update matrix part = A**n
	// AxB = A**xB -> influence of the input to the state  (from 0 to n-1 with A0B = B, A1B = AB)
	// CAx = CA**x -> influence of the state to the output (from 0 to n-1 with CA0 = C, CA1 = CA)
	// CAxB = CA**xB -> influence of the previous input to the output (from 0 to n-1 with CA0B = CB, CA1B = CAB)


	//[ xi+o_0 ]   [ An  An  A3B  A2B  A1B  A0B ]   [ xi_0  ]
	//[ xi+o_1 ]   [ An  An  A3B  A2B  A1B  A0B ]   [ xi_1  ] 
	//[ y_i    ] = [ CA0 CA0  D    0    0    0  ] * [ u_i   ]
	//[ y_i+1  ]   [ CA1 CA1 CA0B  D    0    0  ]   [ u_i+1 ]
	//[ y_i+2  ]   [ CA2 CA2 CA1B CA0B  D    0  ]   [ u_i+2 ]
	//[ y_i+3  ]   [ CA3 CA3 CA2B CA1B CA0B  D  ]   [ u_i+3 ]

	
	Mat<T> outMat = Mat<T>::identidy(_size + outputs - 1);

	// D  (rest will be overwritten)
	outMat = outMat * D;

	//An
	Mat<T> An = pow(A, outputs);
	for (size_t m = 0; m < M; m++)
		for (size_t n = 0; n < M; n++)
			outMat(m, n) = An(m, n);


	for (size_t x = 0; x < outputs; ++x)
	{
		// AxB
		auto Ax = pow(A, x);
		auto AxB = Ax * B;
		for (size_t m = 0; m < M; m++)
			outMat(m, M + outputs - 1 - x) = AxB(m);

		// CAx
		auto CAx = transpose(Ax) * C;
		for (size_t m = 0; m < M; m++)
			outMat(M + x, m) = CAx(m);

		// CAxB
		T CAxB = dot(CAx, B);
		for (size_t i = 0; i < outputs - x - 1; i++)
			outMat(M + x + 1 + i, M + i) = CAxB;
	}

	return outMat.data;
}

template void GenericFilter<float>::generateStateSpace(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, float& D) const;
template void GenericFilter<double>::generateStateSpace(std::vector<double>& A, std::vector<double>& B, std::vector<double>& C, double& D) const;
template std::vector<float> GenericFilter<float>::generateStateSpaceMatrix(size_t outputs) const;
template std::vector<double> GenericFilter<double>::generateStateSpaceMatrix(size_t outputs) const;