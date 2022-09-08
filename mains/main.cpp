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
#include "config.h"

#include <iostream>

int main(__attribute__((unused)) int argc, __attribute__((unused)) char **argv)
{
    BUNJI_INF("bunjilearn version {}.{}", BUNJILEARN_VERSION_MAJOR, BUNJILEARN_VERSION_MINOR);

    bunji::Dataset dataset("../scripts/dataset.json");

    bunji::Network network;

    bunji::Flatten f0;
    bunji::Dense d0(256);
    bunji::Sigmoid a0;
    bunji::Dropout r0(0.5);
    bunji::Dense d1(10);
    bunji::Softmax a1;

    network.add_layer(&f0);
    network.add_layer(&d0);
    network.add_layer(&a0);
    network.add_layer(&r0);
    network.add_layer(&d1);
    network.add_layer(&a1);
    network.build(std::make_tuple(28, 28, 1));

    bunji::Crossentropy loss;

    bunji::Crossentropy loss_metric;
    bunji::Accuracy acc_metric;
    
    std::vector<bunji::Metric*> metrics = {&loss_metric, &acc_metric};

    bunji::Trainer network_trainer(&network, &dataset, &loss, metrics, 5);

    network_trainer.fit(100, 32);
    
    return 0;
}
