#include "tensor.hpp"
#include "log.hpp"

#include <gtest/gtest.h>

#include <array>
#include <numeric>

template<std::size_t N>
bool next(std::array<std::size_t, N> &indices, const std::array<std::size_t, N> &dimensions)
{
    for (int i = N-1; i >= 0; --i)
    {
        if (++indices[i] != dimensions[i])
        {
            return true;
        }
        indices[i] = 0;
    }
    return false;
}

template<std::size_t N, std::size_t I=0, typename Ty>
auto &tensor_get(Ty &tensor, const std::array<std::size_t, N> &indices)
{
    if constexpr(I == N-1)
    {
        return tensor[indices[I]];
    }
    else
    {
        auto indexed = tensor[indices[I]];
        return tensor_get<N, I + 1>(indexed, indices);
    }
}

template<typename... Dimensions>
void test_manual(Dimensions ...dimensions)
{
    constexpr std::size_t N = sizeof...(Dimensions);
    
    std::array<std::size_t, N> indices = {};
    std::array<std::size_t, N> dimensions_arr = {dimensions ...};
    bunji::Tensor<int, N> tensor({dimensions ...});
    
    do
    {
        auto &value = tensor_get<N>(tensor, indices);
        EXPECT_EQ(value, 0);
        value = std::accumulate(indices.begin(), indices.end(), 1, std::multiplies<int>());
    }
    while (next<N>(indices, dimensions_arr));
    
    do
    {
        auto value = tensor_get<N>(tensor, indices);
        auto expected_value = std::accumulate(indices.begin(), indices.end(), 1, std::multiplies<int>());
        EXPECT_EQ(value, expected_value);
    }
    while (next<N>(indices, dimensions_arr));
}

template<std::size_t N, typename Ty>
void init_check(Ty &value, int &sum)
{
    if constexpr(N == 0)
    {
        EXPECT_EQ(value, 0);
        value = sum++;
    }
    else
    {
        if constexpr(N == 1)
        {
            for (auto &x : value)
            {
                init_check<N-1>(x, sum);
            }
        }
        else
        {
            for (auto x : value)
            {
                init_check<N-1>(x, sum);
            }
        }
    }
}

template<std::size_t N, typename Ty>
void read_check(Ty value, int &sum)
{
    if constexpr(N == 0)
    {
        EXPECT_EQ(value, sum++);
    }
    else
    {
        for (auto x : value)
        {
            read_check<N-1>(x, sum);
        }
    }
}

template<typename... Dimensions>
void test_automatic(Dimensions ...dimensions)
{
    constexpr std::size_t N = sizeof...(Dimensions);
    
    bunji::Tensor<int, N> tensor({dimensions ...});

    for (auto x : tensor)
    {
        int sum = 0;
        init_check<N-1>(x, sum);   
    }

    for (auto x : tensor)
    {
        int sum = 0;
        read_check<N-1>(x, sum);   
    }
}

template<std::size_t Dimensions, class Callable>
void nd_for_loop(std::size_t begin, std::size_t end, Callable &&c)
{
    for(size_t i = begin; i != end; ++i)
    {
        if constexpr(Dimensions == 1)
        {
            c(i);
        }
        else
        {
            auto bind_argument = [i, &c](auto... args)
            {
                c(i, args...);
            };
            nd_for_loop<Dimensions-1>(begin, end, bind_argument);
        }
    }
}

TEST(tensor, tensor_manual)
{
    nd_for_loop<1>(1, 8500, test_manual<std::size_t>);
    nd_for_loop<2>(1, 100, test_manual<std::size_t, std::size_t>);
    nd_for_loop<3>(1, 24, test_manual<std::size_t, std::size_t, std::size_t>);
    nd_for_loop<4>(1, 12, test_manual<std::size_t, std::size_t, std::size_t, std::size_t>);
}


TEST(tensor, tensor_auto)
{
    nd_for_loop<1>(0, 8500, test_automatic<std::size_t>);
    nd_for_loop<2>(0, 100, test_automatic<std::size_t, std::size_t>);
    nd_for_loop<3>(0, 24, test_automatic<std::size_t, std::size_t, std::size_t>);
    nd_for_loop<4>(0, 12, test_automatic<std::size_t, std::size_t, std::size_t, std::size_t>);
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}