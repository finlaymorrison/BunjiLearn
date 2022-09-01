#include "metric.hpp"

#include <algorithm>

namespace bunji
{
    
Accuracy::Accuracy() :
    correct(0)
{}

void Accuracy::update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    int max_pred_index = -1;
    double max_pred_val = -1.0;
    int max_expt_index = -1;
    double max_expt_val = -1.0;
    for (std::size_t i = 0; i < output[0][0].size(); ++i)
    {
        if (output[0][0][i] > max_pred_val)
        {
            max_pred_val = output[0][0][i];
            max_pred_index = i;
        }
        if (expected_output[0][0][i] > max_expt_val)
        {
            max_expt_val = output[0][0][i];
            max_expt_index = i;
        }
    }
    
    if (max_pred_index == max_expt_index)
    {
        ++correct;
    }
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
