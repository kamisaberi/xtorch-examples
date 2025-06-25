#include <iostream>
#include <torch/torch.h>
#include <torch/data/datasets.h> // For torch::data::Dataset
#include <torch/data/example.h>  // For torch::data::Example
#include <vector>
#include <string>
#include <algorithm> // For std::shuffle, std::iota, std::min, std::max
#include <future>    // Not directly used in V2, but good to keep in mind for other patterns
#include <optional>   // For torch::optional and std::optional
#include <chrono>     // For timing in main
#include <xtorch/xtorch.h>

using namespace std;

class SimpleModule : xt::Module
{
public:
    SimpleModule(): fc1(10, 20), fc2(20, 10), fc3(10, 5), flatten(5, 1)
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("flatten", flatten);
    }

    auto forward(std::initializer_list<std::any> tensors) -> std::any override
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor input = tensor_vec[0];

        input = input.to(torch::kFloat32);
        input = fc1->forward(input);
        input = fc2->forward(input);
        input = fc3->forward(input);
        input = flatten->forward(input);
        return input;
    }

    torch::Tensor forward(torch::Tensor input)
    {
        std::initializer_list<std::any> tensors = {input};
        return any_cast<torch::Tensor>(this->forward(tensors));
    }

private:
    torch::nn::Linear fc1 = {nullptr};
    torch::nn::Linear fc2 = {nullptr};
    torch::nn::Linear fc3 = {nullptr};
    torch::nn::Flatten flatten = {nullptr};
};


int main()
{
    std::cout.precision(10);

    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}));
    transform_list.push_back(
        std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5}, std::vector<float>{0.5}));

    auto compose = std::make_unique<xt::transforms::Compose>(transform_list);

    auto dataset = xt::datasets::MNIST(
        "/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false, std::move(compose));


    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 64, true, 2, /*prefetch_factor=*/2);

    xt::models::LeNet5 model(10);
    model.to(torch::Device(torch::kCPU));
    model.train();

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    auto logger = std::make_shared<xt::LoggingCallback>("[MyTrain]", /*log_every_N_batches=*/20, /*log_time=*/true);
    xt::Trainer trainer;
    trainer.set_max_epochs(10).set_optimizer(optimizer)
           .set_loss_fn([](auto output, auto target)
           {
               return torch::nll_loss(output, target);
           })
           .add_callback(logger);

    trainer.fit(model, data_loader, &data_loader, torch::Device(torch::kCPU));


    return 0;
}
