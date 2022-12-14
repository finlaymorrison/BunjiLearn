#include "dense.hpp"
#include "dataset.hpp"
#include "activation.hpp"
#include "network.hpp"
#include "trainer.hpp"
#include "loss.hpp"
#include "metric.hpp"
#include "flatten.hpp"
#include "log.hpp"
#include "tensor.hpp"
#include "dropout.hpp"
#include "convolution.hpp"
#include "pool.hpp"

#include <iostream>

int main(__attribute__((unused)) int argc, __attribute__((unused)) char **argv)
{
    bunji::Dataset dataset("scripts/dataset.json");

    bunji::Network network;

    bunji::Convolution c0(16, 5, 5, 1, 1);
    bunji::Sigmoid a0;
    bunji::MaxPool p0(2, 2);
    bunji::Flatten f0;
    bunji::Dense d0(256);
    bunji::Sigmoid a2;
    bunji::Dropout r0(0.35);
    bunji::Dense d1(10);
    bunji::Softmax a3;

    network.add_layer(&c0);
    network.add_layer(&a0);
    network.add_layer(&p0);
    
    network.add_layer(&f0);
    
    network.add_layer(&d0);
    network.add_layer(&a2);
    
    network.add_layer(&r0);
    
    network.add_layer(&d1);
    network.add_layer(&a3);
    
    network.build(std::make_tuple(28, 28, 1));
    
    bunji::Crossentropy loss;

    bunji::Crossentropy loss_metric;
    bunji::Accuracy acc_metric;
    
    std::vector<bunji::Metric*> metrics = {&loss_metric, &acc_metric};

    bunji::Trainer network_trainer(&network, &dataset, &loss, metrics, 0.1);

    network_trainer.fit(100, 32);
    
    return 0;
}
