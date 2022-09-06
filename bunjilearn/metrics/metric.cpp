#include "metric.hpp"
#include "log.hpp"

#include <algorithm>

namespace bunji
{
    
Accuracy::Accuracy() :
    correct(0)
{}

/*
 * It doesnt really makes sense to use accuracy on an input which is more
 * than 1 dimension, other than potentially trying to estimate many random
 * variables, and so each vector in the deepest axis will be considered
 * as a separate random variable. If all labels are predicted correctly,
 * it will be counted as one correct answer.
 */
void Accuracy::update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    const auto &[x, y, z] = output.shape();
    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            std::size_t max_pred_index{0}, max_expt_index{0};
            double max_pred_val = -1.0;
            double max_expt_val = -1.0;
            for (std::size_t k = 0; k < z; ++k)
            {
                if (output[i][j][k] > max_pred_val)
                {
                    max_pred_val = output[i][j][k];
                    max_pred_index = k;
                }
                if (expected_output[i][j][k] > max_expt_val)
                {
                    max_expt_val = expected_output[i][j][k];
                    max_expt_index = k;
                }
            }
    
            if (max_pred_index != max_expt_index)
            {
                return;
            }
        }
    }
    
    ++correct;
}

double Accuracy::evaluate(int example_count)
{
    double accuracy = static_cast<double>(correct) / example_count;
    correct = 0;
    return accuracy;
}

std::string Accuracy::get_name()
{
    return std::string("accuracy");
}

} // namespace bunji
