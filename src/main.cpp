#include "dense.hpp"
#include "dataset.hpp"
#include "config.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include <iostream>

int main(int argc, char **argv)
{
    std::cout << "ml_library version " << ml_library_VERSION_MAJOR << "." << ml_library_VERSION_MINOR << '\n' << std::endl;

    Dataset dataset("../scripts/dataset.json");

    
    
    return 0;
}