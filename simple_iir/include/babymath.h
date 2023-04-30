#include <vector>
#include <cassert>

template<typename T>
struct Vec
{
	const size_t Length;

	std::vector<T> data;
	Vec(size_t length, T d = 0) : Length(length)
	{
		data.resize(length);
		for (auto& m : data)
			m = d;
	}
	Vec(const std::vector<T>& data) : Length(data.size()), data(data)
	{
	}
	Vec(const Vec& other) = default;
	Vec(Vec&& other) = default;

	Vec& operator= (const Vec& other)
	{
		assert(Length == other.Length);
		data = other.data;
		return *this;
	}
	Vec& operator= (Vec&& other)
	{
		assert(Length == other.Length);
		std::swap(data, other.data);
		return *this;
	}

	T& operator[](size_t i) { return data[i]; }
	T& operator()(size_t i) { return data[i]; }
	const T& operator[](size_t i) const { return data[i]; }
	const T& operator()(size_t i) const { return data[i]; }

	Vec operator+(const Vec& other) const
	{
		assert(Length == other.Length);
		Vec res{Length};
		for (size_t i = 0; i < Length; ++i)
			res[i] = data[i] + other[i];
		return res;
	}
	Vec operator*(T v) const
	{
		Vec res{ data };
		for (size_t i = 0; i < Length; ++i)
			res[i] *= v;
		return res;
	}
	friend T dot(const Vec& a, const Vec& b)
	{
		assert(a.Length == b.Length);
		T res = 0;
		for (size_t i = 0; i < a.Length; ++i)
			res += a[i] * b[i];
		return res;
	}
};

template<typename T>
struct Mat
{
	const size_t Rows;
	const size_t Cols;
	std::vector<T> data;

	Mat(size_t rows, size_t cols, T d = 0) : Rows(rows), Cols(cols)
	{
		data.resize(rows * cols);
		for (auto& m : data)
			m = d;
	}

	Mat(size_t rows, size_t cols, const std::vector<T>& data) : Rows(rows), Cols(cols), data(data)
	{
		assert(data.size() == Rows * Cols);
	}

	Mat(const Mat& other) = default;
	Mat(Mat&& other) = default;


	Mat& operator= (const Mat& other)
	{
		assert(Rows == other.Rows);
		assert(Cols == other.Cols);
		data = other.data;
		return *this;
	}
	Mat& operator= (Mat&& other)
	{
		assert(Rows == other.Rows);
		assert(Cols == other.Cols);
		std::swap(data, other.data);
		return *this;
	}

	static Mat identidy(size_t size) {
		Mat d{ size, size };
		for (size_t i = 0; i < size; ++i)
			d(i, i) = 1;
		return d;
	}


	T& operator()(size_t row, size_t col)
	{
		return data[row * Cols + col];
	}

	const T& operator()(size_t row, size_t col) const
	{
		return data[row * Cols + col];
	}

	Mat<T> operator *(T v) const
	{
		Mat<T> res{ Rows, Cols, data };
		for (size_t m = 0; m < Rows; ++m)
			for (size_t n = 0; n < Cols; ++n)
				res(m, n) *= v;
		return res;
	}

	Vec<T> operator *(const Vec<T>& v) const
	{
		Vec<T> res{ Rows };
		for (size_t m = 0; m < Rows; ++m)
		{
			for (size_t i = 0; i < Cols; ++i)
				res[m] += data[Cols * m + i] * v[i];
		}
		return res;
	}



	Mat<T> operator *(const Mat<T>& other) const
	{
		assert(Cols == other.Rows);
		Mat<T> res(Rows, other.Cols);
		for (size_t m = 0; m < Rows; ++m)
			for (size_t n = 0; n < other.Cols; ++n)
			{
				for (size_t i = 0; i < Cols; ++i)
					res(m, n) += data[Cols * m + i] * other(i, n);
			}
		return res;
	}

	friend Mat transpose(const Mat& mat)
	{
		Mat res{ mat.Cols, mat.Rows };
		for (size_t m = 0; m < mat.Rows; ++m)
			for (size_t n = 0; n < mat.Cols; ++n)
				res(n, m) = mat(m, n);
		return res;
	}
};


template<typename T>
const Mat<T> pow(const Mat<T>& A, size_t pow)
{
	assert(A.Rows == A.Cols);
	size_t M = A.Rows;
	auto res = Mat<T>::identidy(M);

	Mat<T> powA = A;
	for (size_t e = 1; e <= pow; e *= 2)
	{
		if ((pow & e) != 0)
			res = res * powA;
		powA = powA * powA;
	}
	return res;
}