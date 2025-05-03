#include <torch/torch.h>
#include <iostream>
#include <xtorch/xtorch.h>

int main() {
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << endl;
    std::cout << "Hello World" << std::endl;
    std::string  r = "./test/test1/ali";
    auto temp = xt::temp::TestDataset();
    auto dataset = xt::data::datasets::MNIST("/home/kami/Documents/datasets/", DataMode::TRAIN, true);

    return 0;
}

