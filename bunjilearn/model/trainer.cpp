#include "trainer.hpp"
#include "log.hpp"

namespace bunji
{

Trainer::Trainer(Network *network, Dataset *dataset, Loss *loss, const std::vector<Metric*> &metrics, double learn_rate) :
    network(network), dataset(dataset), loss(loss), metrics(metrics), learn_rate(learn_rate)
{}

void Trainer::train_example(const Tensor<double, 3> &input, const Tensor<double, 3> &expected_output)
{
    Tensor output = network->forward_pass(input);
    loss->update(output, expected_output);
    for (Metric *metric : metrics)
    {
        metric->update(output, expected_output);
    }
    Tensor output_derivatives = loss->derivative(output, expected_output);
    Tensor input_derivatives = network->backward_pass(input, output_derivatives);
}

std::vector<double> Trainer::train_pass()
{
    for (int i = 0; i < dataset->train_len(); ++i)
    {
        std::pair<Tensor<double, 3>, Tensor<double, 3>> example = dataset->train(i);
        train_example(example.first, example.second);
    }
    network->apply_gradients(learn_rate / dataset->train_len());

    std::vector<double> metric_vals(metrics.size());

    for (int i = 0; i < metrics.size(); ++i)
    {
        metric_vals[i] = metrics[i]->evaluate(dataset->train_len());
    }

    return metric_vals;
}

void Trainer::fit(int epochs)
{
    for (int i = 0; i < epochs; ++i)
    {
        std::vector<double> metric_vals = train_pass();
        BUNJI_PRINT("Epoch: {}", i);
        for (int i = 0; i < metrics.size(); ++i)
        {
            BUNJI_PRINT("\t{}:{}", metrics[i]->get_name(), metric_vals[i]);
        }
        BUNJI_PRINT("\n");
    }
}

} // namespace bunji
