#include "tensor.hpp"

#include <gtest/gtest.h>

void test_manual_1d(const std::size_t x)
{
    bunji::Tensor<int, 1> tensor({x});

    for (std::size_t i = 0; i < x; ++i)
    {
        EXPECT_EQ(tensor[i], 0);
        tensor[i] = i+1;
    }

    for (std::size_t i = 0; i < x; ++i)
    {
        EXPECT_EQ(tensor[i], i+1);
    }
}

void test_auto_1d(const std::size_t x)
{
    bunji::Tensor<int, 1> tensor({x});

    int index = 0;
    for (int &i : tensor)
    {
        EXPECT_EQ(i, 0);
        i = ++index;
    }

    index = 0;
    for (const int i : tensor)
    {
        EXPECT_EQ(i, ++index);
    }
}

void test_manual_2d(const std::size_t x, const std::size_t y)
{
    bunji::Tensor<int, 2> tensor({x, y});

    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            EXPECT_EQ(tensor[i][j], 0);
            tensor[i][j] = (i+1) * (j+1);
        }
    }

    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            EXPECT_EQ(tensor[i][j], (i+1) * (j+1));
        }
    }
}

void test_auto_2d(const std::size_t x, const std::size_t y)
{
    bunji::Tensor<int, 2> tensor({x, y});

    int index = 0;
    for (auto v_0 : tensor)
    {
        for (int & i : v_0)
        {
            EXPECT_EQ(i, 0);
            i = ++index;
        }
    }

    index = 0;
    for (auto v_0 : tensor)
    {
        for (const int i : v_0)
        {
            EXPECT_EQ(i, ++index);
        }
    }
}

void test_manual_3d(const std::size_t x, const std::size_t y, const std::size_t z)
{
    bunji::Tensor<int, 3> tensor({x, y, z});

    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            for (std::size_t k = 0; k < z; ++k)
            {
                EXPECT_EQ(tensor[i][j][k], 0);
                tensor[i][j][k] = (i+1) * (j+1) * (k+1);
            }
        }
    }

    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            for (std::size_t k = 0; k < z; ++k)
            {
                EXPECT_EQ(tensor[i][j][k], (i+1) * (j+1) * (k+1));
            }
        }
    }
}

void test_auto_3d(const std::size_t x, const std::size_t y, const std::size_t z)
{
    bunji::Tensor<int, 3> tensor({x, y, z});

    int index = 0;
    for (auto v_0 : tensor)
    {
        for (auto v_1 : v_0)
        {
            for (int & i : v_1)
            {
                EXPECT_EQ(i, 0);
                i = ++index;
            }
        }
    }

    index = 0;
    for (auto v_0 : tensor)
    {
        for (auto v_1 : v_0)
        {
            for (const int i : v_1)
            {
                EXPECT_EQ(i, ++index);
            }
        }
    }
}

void test_manual_4d(const std::size_t x, const std::size_t y, const std::size_t z, const std::size_t v)
{
    bunji::Tensor<int, 4> tensor({x, y, z, v});

    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            for (std::size_t k = 0; k < z; ++k)
            {
                for (std::size_t l = 0; l < v; ++l)
                {
                    EXPECT_EQ(tensor[i][j][k][l], 0);
                    tensor[i][j][k][l] = (i+1) * (j+1) * (k+1) * (l+1);
                }
            }
        }
    }

    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            for (std::size_t k = 0; k < z; ++k)
            {
                for (std::size_t l = 0; l < v; ++l)
                {
                    EXPECT_EQ(tensor[i][j][k][l], (i+1) * (j+1) * (k+1) * (l+1));
                }
            }
        }
    }
}

void test_auto_4d(const std::size_t x, const std::size_t y, const std::size_t z, const std::size_t v)
{
    bunji::Tensor<int, 4> tensor({x, y, z, v});

    int index = 0;
    for (auto v_0 : tensor)
    {
        for (auto v_1 : v_0)
        {
            for (auto v_2 : v_1)
            {
                for (int & i : v_2)
                {
                    EXPECT_EQ(i, 0);
                    i = ++index;
                }
            }
        }
    }

    index = 0;
    for (auto v_0 : tensor)
    {
        for (auto v_1 : v_0)
        {
            for (auto v_2 : v_1)
            {
                for (const int i : v_2)
                {
                    EXPECT_EQ(i, ++index);
                }
            }
        }
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
    nd_for_loop<1>(0, 8500, test_manual_1d);
    nd_for_loop<2>(0, 100, test_manual_2d);
    nd_for_loop<3>(0, 24, test_manual_3d);
    nd_for_loop<4>(0, 12, test_manual_4d);
}

TEST(tensor, tensor_auto)
{
    nd_for_loop<1>(0, 8500, test_auto_1d);
    nd_for_loop<2>(0, 100, test_auto_2d);
    nd_for_loop<3>(0, 24, test_auto_3d);
    nd_for_loop<4>(0, 12, test_auto_4d);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}