#pragma once

#include <memory>
#include <initializer_list>
#include <vector>
#include <algorithm>

template<typename T>
struct GenericFilter
{
	std::unique_ptr<T[]> _a;
	std::unique_ptr<T[]> _b;
	int _size;

	GenericFilter() = default;
	~GenericFilter() = default;
	template<typename FA, typename FB>
	GenericFilter(FA&& fa, FB&& fb, int size) :
		_a{ std::make_unique<T[]>(size) },
		_b{ std::make_unique<T[]>(size) },
		_size(size)
	{
		for (int i = 0; i < _size; ++i)
		{
			_a[i] = fa(i);
			_b[i] = fb(i);
		}
	}
	GenericFilter(const GenericFilter& other) : 
		_a{ std::make_unique<T[]>(other._size) },
		_b{ std::make_unique<T[]>(other._size) },
		_size(other._size)
	{
		for (int i = 0; i < _size; ++i)
		{
			_a[i] = other._a[i];
			_b[i] = other._b[i];
		}
	}
	GenericFilter(GenericFilter&&) = default;
	GenericFilter& operator= (const GenericFilter& other)
	{
		_size = other._size;
		_a = std::make_unique<T[]>(_size);
		_b = std::make_unique<T[]>(_size);
		for (int i = 0; i < _size; ++i)
		{
			_a[i] = other._a[i];
			_b[i] = other._b[i];
		}
	}
	GenericFilter& operator= (GenericFilter&&) = default;

	template<typename T2>
	GenericFilter(const GenericFilter<T2>& other) : 
		_a{ std::make_unique<T[]>(other._size) }, 
		_b{ std::make_unique<T[]>(other._size) }, 
		_size{ other._size }
	{
		for (int i = 0; i < _size; ++i)
		{
			_a[i] = other._a[i];
			_b[i] = other._b[i];
		}
	}

	bool operator== (const GenericFilter& other) const
	{
		if (_size != other._size)
			return false;

		if(!std::equal(_a.get(), _a.get() + _size, other._a.get()))
			return false;
		
		return std::equal(_b.get(), _b.get() + _size, other._b.get());
	}

	void setup(const std::initializer_list<T>& a, const std::initializer_list<T> b, int size)
	{
		static_assert(a.size() == size && b.size() == size, "invalid filter coefficient sizes");
		_a = std::make_unique<T[]>(a);
		_b = std::make_unique<T[]>(b);
		_size = size;
	}


	void generateStateSpace(std::vector<T>& A, std::vector<T>& B, std::vector<T>& C, T& D) const;
	std::vector<T> generateStateSpaceMatrix(size_t outputs) const;
};


using Filter = GenericFilter<float>;
