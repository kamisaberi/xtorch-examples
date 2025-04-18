#include <torch/torch.h>
//#include <torch/data/datasets/mnist.h>
//#include <vector>
//#include <fstream>
#include <iostream>
//#include <string>
//#include <filesystem>
//#include <curl/curl.h>
#include <xTorch/xtorch.h>
#include <torch/torch.h>


int main() {
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << endl;
    std::cout << "Hello World" << std::endl;
    std::string  r = "./test/test1/ali";
    auto temp = xt::temp::TestDataset();
    // auto dataset = xt::data::datasets::MNIST("/home/kami/Documents/temp/", DataMode::TRAIN, true);

    // torch::ext::data::datasets::UCF101 u1(r);
    // std::cout << "End\n";
//    torch::ext::data::datasets::CIFAR100 cifar100("/home/kami/Documents/temp/", true , true);


    return 0;
}

