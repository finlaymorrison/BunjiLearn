#pragma once

#include <vector>
#include <cstddef>
#include <utility>
#include <tuple>

namespace bunji
{

/* predeclarations of view classes for use in iterators */
template<typename Ty, int N>
class TensorView;
template<typename Ty, int N>
class ConstTensorView;

/*
 *
 */
template<class Tensor>
class TensorIterator
{
public:
    using ValueType = typename Tensor::ValueType;
    static constexpr int DIM = Tensor::DIM;

private:
    ValueType *data;
    const std::size_t *offsets;

public:
    TensorIterator() = default;
    TensorIterator(ValueType *data, const std::size_t *offsets) :
        data(data), offsets(offsets)
    {}

    TensorIterator<Tensor> &operator++ ()
    {
        data += offsets[DIM-1];
        return *this;
    }
    TensorIterator<Tensor> operator++ (int)
    {
        TensorIterator<Tensor> iterator = *this;
        ++(*this);
        return iterator;
    }

    TensorIterator<Tensor> &operator-- ()
    {
        data -= offsets[DIM-1];
        return *this;
    }
    TensorIterator<Tensor> operator-- (int)
    {
        TensorIterator<Tensor> iterator = *this;
        --(*this);
        return iterator;
    }

    decltype(auto) operator* ()
    {
        if constexpr(DIM == 1)
        {
            return *data;
        }
        else
        {
            return TensorView<ValueType, DIM-1>(data, offsets);
        }
    }

    bool operator== (const TensorIterator<Tensor> &other) const
    {
        return other.data == data;
    }
    bool operator!= (const TensorIterator<Tensor> &other) const
    {
        return !(*this == other);
    }

};

/*
 *
 */
template<class Tensor>
class ConstTensorIterator
{
public:
    using ValueType = typename Tensor::ValueType;
    static constexpr int DIM = Tensor::DIM;

private:
    const ValueType *data;
    const std::size_t *offsets;

public:
    ConstTensorIterator() = default;
    ConstTensorIterator(const ValueType *data, const std::size_t *offsets) :
        data(data), offsets(offsets)
    {}

    ConstTensorIterator<Tensor> operator++ ()
    {
        data += offsets[DIM-1];
        return *this;
    }
    ConstTensorIterator<Tensor> operator++ (int)
    {
        ConstTensorIterator<Tensor> iterator = *this;
        ++(*this);
        return iterator;
    }

    ConstTensorIterator<Tensor> operator-- ()
    {
        data -= offsets[DIM-1];
        return *this;
    }
    ConstTensorIterator<Tensor> operator-- (int)
    {
        ConstTensorIterator<Tensor> iterator = *this;
        --(*this);
        return iterator;
    }

    auto operator* () const
    {
        if constexpr(DIM == 1)
        {
            return *data;
        }
        else
        {
            return ConstTensorView<ValueType, DIM-1>(data, offsets);
        }
    }

    bool operator== (const ConstTensorIterator<Tensor> &other) const
    {
        return other.data == data;
    }
    bool operator!= (const ConstTensorIterator<Tensor> &other) const
    {
        return !(*this == other);
    }
};

/*
 *
 */
template<typename Ty, int N>
class ConstTensorView
{
public:
    using ValueType = Ty;
    static constexpr int DIM = N;
    using const_iterator = ConstTensorIterator<TensorView<Ty, N>>;

private:
    const ValueType *data;
    const std::size_t *offsets;

public:
    ConstTensorView() = delete;
    ConstTensorView(const ValueType *data, const std::size_t *offsets) :
        data(data), offsets(offsets)
    {}

    auto operator[] (std::size_t index) const
    {
        if constexpr (DIM == 1)
        {
            return data[index];
        }
        else
        {
            return ConstTensorView<ValueType, DIM-1>(&data[index*offsets[DIM-1]], offsets);
        }
    }

    std::size_t size(std::size_t axis=0) const
    {
        return offsets[DIM-axis] / offsets[DIM-axis-1];
    }

    const_iterator begin() const
    {
        return const_iterator{&data[0], &offsets[0]};
    }
    const_iterator end() const
    {
        return const_iterator{&data[offsets[DIM]], &offsets[0]};
    }

    auto shape() const
    {
        auto make_tuple = [this]<typename I, I... indices>(std::index_sequence<indices...>)
        {
            return std::make_tuple((this->offsets[N - indices] / this->offsets[N - indices - 1])...);
        };
        return make_tuple(std::make_index_sequence<N>());
    }
};

/*
 *
 */
template<typename Ty, int N>
class TensorView
{
public:
    using ValueType = Ty;
    static constexpr int DIM = N;
    using iterator = TensorIterator<TensorView<Ty, N>>;
    using const_iterator = ConstTensorIterator<TensorView<Ty, N>>;

private:
    ValueType *data;
    const std::size_t *offsets;

public:
    TensorView() = delete;
    TensorView(ValueType *data, const std::size_t *offsets) :
        data(data), offsets(offsets)
    {}

    decltype(auto) operator[] (std::size_t index)
    {
        if constexpr (DIM == 1)
        {
            return data[index];
        }
        else
        {
            return TensorView<ValueType, DIM-1>(&data[index*offsets[DIM-1]], offsets);
        }
    }
    auto operator[] (std::size_t index) const
    {
        if constexpr (DIM == 1)
        {
            return data[index];
        }
        else
        {
            return ConstTensorView<ValueType, DIM-1>(&data[index*offsets[DIM-1]], offsets);
        }
    }

    std::size_t size(std::size_t axis=0) const
    {
        return offsets[DIM-axis] / offsets[DIM-axis-1];
    }

    iterator begin()
    {
        return iterator{&data[0], &offsets[0]};
    }
    iterator end()
    {
        return iterator{&data[offsets[DIM]], &offsets[0]};
    }
    const_iterator begin() const
    {
        return const_iterator{&data[0], &offsets[0]};
    }
    const_iterator end() const
    {
        return const_iterator{&data[offsets[DIM]], &offsets[0]};
    }

    auto shape() const
    {
        auto make_tuple = [this]<typename I, I... indices>(std::index_sequence<indices...>)
        {
            return std::make_tuple((this->offsets[N - indices] / this->offsets[N - indices - 1])...);
        };
        return make_tuple(std::make_index_sequence<N>());
    }
};

/*
 * A class representing a multi-dimesional array as a single continuous
 * memory buffer provided by a `std::vector` to optimise cache usage.
 *
 * The default constructor simply allocates memory for defining the strides;
 * however no buffer for data storage is allocated. Another constructor is
 * provided which takes in a vector of dimensions which allocates memory for
 * the buffer with the `resize` member function.
 *
 * // 10 x 50 x 5 tensor of floats.
 * Tensor<float, 3> my_float_tensor({10, 50, 5});
 *
 * The tensor can be indexed with `operator[]`, which will return `ValueType`
 * if the tensor has only a single dimension, and will return an object of
 * type `TensorView` or `ConstTensorView` whose dimension is one less
 * than the tensor otherwise. Both const and non-const iterators are provided 
 * for this class which similarly return different types depending on the
 * Tensor's dimension.
 *
 * Tensor<long, 2> matrix({8, 3});
 * for (std::size_t i = 0; i < 8; ++i)
 * {
 *     for (std::size_t j = 0; j < 3; ++j)
 *     {
 *         matrix[i][j] = 4;
 *     }
 * }
 *
 * When iterating over a Tensor in an auto-for loop there is no need to state
 * that the range delcaration is a reference type besides when the range 
 * expression is a tensor of dimension 1, as when returning a `TensorView`
 * or `ConstTensorView` no memory is allocated.
 *
 * Tensor<int, 2> matrix({10, 5});
 * for (auto vector : matrix) // no memory allocated as TensorView is returned
 * {
 *     for (int &value : vector)
 *     {
 *         // Reference used as range declaration for base type of Tensor.
 *         value = 42;
 *     }
 * }
 *
 * A member function `shape` is also provided which returns the size of each
 * dimension of the tensor as a tuple.
 *
 * Tensor<long double, 4> tensor({40, 20, 10, 100});
 * const auto &[x, y, z, v] = tensor.shape(); // x=40, y=20, z=10, v=100
 */
template<typename Ty, int N>
class Tensor
{
public:
    using ValueType = Ty;
    static constexpr int DIM = N;
    using iterator = TensorIterator<TensorView<Ty, N>>;
    using const_iterator = ConstTensorIterator<TensorView<Ty, N>>;

private:
    std::vector<ValueType> data;
    std::vector<std::size_t> offsets;

public:
    Tensor() :
        offsets(DIM + 1)
    {}
    Tensor(const std::vector<std::size_t> &dims) :
        offsets(DIM + 1)
    {
        resize(&dims.data()[0]);
    }

    decltype(auto) operator[] (std::size_t index)
    {
        if constexpr (DIM == 1)
        {
            return data[index];
        }
        else
        {
            return TensorView<ValueType, DIM-1>(&data[index*offsets[DIM-1]], &offsets[0]);
        }
    }
    auto operator[] (size_t index) const
    {
        if constexpr (DIM == 1)
        {
            return data[index];
        }
        else
        {
            return ConstTensorView<ValueType, DIM-1>(&data[index*offsets[DIM-1]], &offsets[0]);
        }
    }

    void resize(const size_t *dims)
    {
        offsets[0] = 1;

        for (size_t i = 1; i <= DIM; ++i)
        {
            offsets[i] = offsets[i - 1] * dims[DIM - i];
        }
        
        data.resize(offsets[DIM], ValueType());
    }

    std::size_t size(int axis=0) const
    {
        return offsets[DIM-axis] / offsets[DIM-axis-1];
    }

    iterator begin()
    {
        return iterator{&data[0], &offsets[0]};
    }
    iterator end()
    {
        return iterator{&data[offsets[DIM]], &offsets[0]};
    }
    const_iterator begin() const
    {
        return const_iterator{&data[0], &offsets[0]};
    }
    const_iterator end() const
    {
        return const_iterator{&data[offsets[DIM]], &offsets[0]};
    }

    auto shape() const
    {
        auto make_tuple = [this]<typename I, I... indices>(std::index_sequence<indices...>)
        {
            return std::make_tuple((this->offsets[N - indices] / this->offsets[N - indices - 1])...);
        };
        return make_tuple(std::make_index_sequence<N>());
    }
};

} // namespace bunji