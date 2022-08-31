#include "trainer.hpp"

#include <iostream>

Trainer::Trainer(Network *network, Dataset *dataset, Loss *loss, double learn_rate) :
    network(network), dataset(dataset), loss(loss), learn_rate(learn_rate)
{}

void Trainer::train_example(const Tensor<double, 3> &input, const Tensor<double, 3> &expected_output)
{
    Tensor output = network->forward_pass(input);
    loss->update(output, expected_output);
    Tensor output_derivatives = loss->derivative(output, expected_output);
    Tensor input_derivatives = network->backward_pass(input, output_derivatives);
}

double Trainer::train_pass()
{
    for (int i = 0; i < dataset->train_len(); ++i)
    {
        std::pair<Tensor<double, 3>, Tensor<double, 3>> example = dataset->train(i);
        train_example(example.first, example.second);
    }
    network->apply_gradients(learn_rate / dataset->train_len());
    double total_loss = loss->evaluate();

    return total_loss / dataset->train_len();
}

void Trainer::fit(int epochs)
{
    for (int i = 0; i < epochs; ++i)
    {
        double network_loss = train_pass();
        std::cout << "Epoch: " << i << " Loss: " << network_loss << std::endl;
    }
}