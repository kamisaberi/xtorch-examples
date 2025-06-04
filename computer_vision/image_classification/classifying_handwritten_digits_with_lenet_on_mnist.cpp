#include <iostream>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <algorithm> // For std::shuffle, std::iota, std::min, std::max
#include <future>    // Not directly used in V2, but good to keep in mind for other patterns
#include <optional>   // For torch::optional and std::optional
#include <chrono>     // For timing in main
#include <xtorch/xtorch.h>

using namespace std;



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


    int num_epochs = 2;
    auto datum = dataset.get(0);
    cout << datum.data.sizes() << endl;
    cout << dataset.size().value() << endl;


    // return 0;
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
