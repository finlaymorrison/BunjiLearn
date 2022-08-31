#pragma once

#include <vector>
#include <cstddef>

namespace bunji
{

template<typename Ty, int DIM>
class Tensor
{
private:
    std::vector<Ty> data;
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

    decltype(auto) operator[] (std::size_t i)
    {
        if constexpr (DIM == 1)
        {
            return data[i];
        }
        else
        {
            return typename Tensor<Ty, DIM-1>::View(&data[i*offsets[DIM-1]], &offsets.data()[0]);
        }
    }
    auto operator[] (size_t i) const
    {
        if constexpr (DIM == 1)
        {
            return data[i];
        }
        else
        {
            return typename Tensor<Ty, DIM-1>::ConstView(&data[i*offsets[DIM-1]], &offsets.data()[0]);
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



    class View
    {
    private:
        Ty *data;
        std::size_t *offsets;

        View() = delete;
        View(Ty *data, std::size_t *offsets) :
            data(data), offsets(offsets)
        {}
    
    public:
        friend class Tensor<Ty, DIM+1>;

        decltype(auto) operator[] (std::size_t i)
        {
            if constexpr (DIM == 1)
            {
                return data[i];
            }
            else
            {
                return typename Tensor<Ty, DIM-1>::View(&data[i*offsets[DIM-1]], offsets);
            }
        }

        std::size_t size(std::size_t axis=0) const
        {
            return offsets[DIM-axis] / offsets[DIM-axis-1];
        }
    };



    class ConstView
    {
    private:
        const Ty *data;
        const std::size_t *offsets;

        ConstView() = delete;
        ConstView(const Ty *data, const std::size_t *offsets) :
            data(data), offsets(offsets)
        {}

    public:
        friend class Tensor<Ty, DIM+1>;

        auto operator[] (std::size_t i) const
        {
            if constexpr (DIM == 1)
            {
                return data[i];
            }
            else
            {
                return typename Tensor<Ty, DIM-1>::ConstView(&data[i*offsets[DIM-1]], offsets);
            }
        }

        std::size_t size(std::size_t axis=0) const
        {
            return offsets[DIM-axis] / offsets[DIM-axis-1];
        }
    };
};

} // namespace bunji