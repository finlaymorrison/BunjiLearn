#pragma once

#include <vector>
#include <cstddef>

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
class TensorView
{
public:
    using ValueType = Ty;
    static constexpr int DIM = N;
    using iterator = TensorIterator<TensorView<Ty, N>>;

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

    const_iterator cbegin() const
    {
        return const_iterator{&data[0], &offsets[0]};
    }
    const_iterator cend() const
    {
        return const_iterator{&data[offsets[DIM]], &offsets[0]};
    }
};

/*
 *
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
        
        data.resize(offsets[DIM]);
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
    const_iterator cbegin() const
    {
        return const_iterator{&data[0], &offsets[0]};
    }
    const_iterator cend() const
    {
        return const_iterator{&data[offsets[DIM]], &offsets[0]};
    }
};

} // namespace bunji