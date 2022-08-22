#include "trainer.hpp"

Trainer::Trainer(Network *network, Dataset *dataset, Loss *loss, double learn_rate) :
    network(network), dataset(dataset), loss(loss), learn_rate(learn_rate)
{}

void Trainer::train_example(const Tensor &input, const Tensor &expected_output)
{
    Tensor output = network->forward_pass(input);
    Tensor output_derivatives = loss->derivative(output, expected_output);
    Tensor input_derivatives = network->backward_pass(input, output_derivatives);
}

void Trainer::train_pass()
{
    for (int i = 0; i < dataset->train_len(); ++i)
    {
        std::pair<Tensor, Tensor> example = dataset->train(i);
        train_example(example.first, example.second);
    }
    network->apply_gradients(learn_rate);
}

void Trainer::fit(int epochs)
{
    for (int i = 0; i < epochs; ++i)
    {
        train_pass();
    }
}