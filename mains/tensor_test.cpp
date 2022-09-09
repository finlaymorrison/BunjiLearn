#include "tensor.hpp"

#include <gtest/gtest.h>

#include <vector>
#include <random>

TEST(tensor, tensor_1d_manual)
{
    constexpr std::size_t size = 173;
    constexpr std::size_t seed = 1895;

    bunji::Tensor<double, 1> tensor({size});
    std::vector<double> vector(size);

    for (std::size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(tensor[i], vector[i]);
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution dist(-10.0, 10.0);
    for (std::size_t i = 0; i < size; ++i)
    {
        double value = dist(gen);
        tensor[i] = value;
        vector[i] = value;
    }

    for (std::size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(tensor[i], vector[i]);
    }
}

TEST(tensor, tensor_1d_iterator)
{
    constexpr std::size_t size = 147;
    constexpr std::size_t seed = 7426;

    bunji::Tensor<double, 1> tensor({size});
    std::vector<double> vector(size);

    auto tensor_it = tensor.begin();
    auto vector_it = vector.begin();

    while (tensor_it != tensor.end() || vector_it != vector.end())
    {
        EXPECT_EQ(*tensor_it, *vector_it);
        ++tensor_it;
        ++vector_it;
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution dist(-10.0,10.0);

    tensor_it = tensor.begin();
    vector_it = vector.begin();
    while (tensor_it != tensor.end() || vector_it != vector.end())
    {
        double value = dist(gen);
        *tensor_it = value;
        *vector_it = value;
        ++tensor_it;
        ++vector_it;
    }

    tensor_it = tensor.begin();
    vector_it = vector.begin();
    while (tensor_it != tensor.end() || vector_it != vector.end())
    {
        EXPECT_EQ(*tensor_it, *vector_it);
        ++tensor_it;
        ++vector_it;
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}