#include <xtorch/xtorch.h>
#include <torch/torch.h>
#include <iostream>

using namespace std;

int main() {
    try {
        // Hyperparameters
        const int nz = 100; // Size of latent vector
        const int ngf = 64; // Size of feature maps in generator
        const int ndf = 64; // Size of feature maps in discriminator
        const int nc = 1; // Number of channels (1 for MNIST)
        const int num_epochs = 5;
        const int batch_size = 64;
        const double lr = 0.0002;
        const double beta1 = 0.5;
        const std::string dataroot = "./data";

        // Device
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Initialize models
        xt::models::DCGAN::Generator netG(nz, ngf, nc);
        xt::models::DCGAN::Discriminator netD(nc, ndf);
        netG.to(device);
        netD.to(device);

        // Optimizers (fixed: optimD uses netD.parameters())
        torch::optim::Adam optimG(netG.parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.999}));
        torch::optim::Adam optimD(netD.parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.999}));

        // Loss function
        torch::nn::BCELoss criterion;

        // Transforms (adjusted normalization for MNIST to [-1, 1])
        std::vector<std::shared_ptr<xt::Module>> transform_list;
        transform_list.push_back(
            std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5}, std::vector<float>{0.5}));

        auto compose = std::make_unique<xt::transforms::Compose>(transform_list);
        auto dataset = xt::datasets::MNIST("/home/kami/Documents/datasets/",
                                           xt::datasets::DataMode::TRAIN, false,
                                           std::move(compose));

        xt::dataloaders::ExtendedDataLoader data_loader(dataset, batch_size, true, 2, /*prefetch_factor=*/2);

        // Training loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            for (auto& batch : data_loader) {
                // Train Discriminator
                netD.zero_grad();
                auto real_data = batch.first;
                auto batch_size = real_data.size(0);

                // Real images
                auto real_label = torch::full({batch_size}, 1.0, torch::TensorOptions().device(device).dtype(torch::kFloat));
                auto output = torch::sigmoid(netD.forward(real_data)).view(-1); // Added sigmoid for BCELoss
                auto errD_real = criterion(output, real_label);
                errD_real.backward();

                // Fake images
                auto noise = torch::randn({batch_size, nz, 1, 1}, torch::TensorOptions().device(device));
                auto fake_data = netG.forward(noise);
                auto fake_label = torch::full({batch_size}, 0.0, torch::TensorOptions().device(device).dtype(torch::kFloat));
                output = torch::sigmoid(netD.forward(fake_data.detach())).view(-1); // Added sigmoid
                auto errD_fake = criterion(output, fake_label);
                errD_fake.backward();
                auto errD = errD_real + errD_fake;
                optimD.step();

                // Train Generator
                netG.zero_grad();
                output = torch::sigmoid(netD.forward(fake_data)).view(-1); // Added sigmoid
                auto errG = criterion(output, real_label); // Generator wants discriminator to think fakes are real
                errG.backward();
                optimG.step();

                // Print progress
                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
                          << "] D_Loss: " << errD.item<float>()
                          << " G_Loss: " << errG.item<float>() << std::endl;
            }

            // Save generated images (optional, requires OpenCV or other image handling)
            if (epoch % 2 == 0) {
                auto noise = torch::randn({16, nz, 1, 1}, torch::TensorOptions().device(device));
                auto fake_images = netG.forward(noise);
                // Implement image saving logic here if needed (e.g., using OpenCV)
            }
        }

        // Save models
        // torch::save(netG, "generator.pt");
        // torch::save(netD, "discriminator.pt");
        // std::cout << "Models saved to generator.pt and discriminator.pt" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}