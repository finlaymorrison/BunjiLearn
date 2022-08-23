#include "dense.hpp"
#include "dataset.hpp"
#include "activation.hpp"
#include "network.hpp"
#include "trainer.hpp"
#include "loss.hpp"
#include "flatten.hpp"

#include "config.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include <iostream>

int main(int argc, char **argv)
{
    std::cout << "ml_library version " << ml_library_VERSION_MAJOR << "." << ml_library_VERSION_MINOR << '\n' << std::endl;

    Dataset dataset("../scripts/dataset.json");

    Network network;

    Flatten f0(1, 28, 28);
    Dense d0(784, 128);
    Sigmoid a0;
    Dense d1(128, 10);
    Softmax a1;

    network.add_layer(&f0);
    network.add_layer(&d0);
    network.add_layer(&a0);
    network.add_layer(&d1);
    network.add_layer(&a1);

    SquaredError loss;

    Trainer network_trainer(&network, &dataset, &loss, 0.005);

    network_trainer.fit(1000);
    
    return 0;
}
