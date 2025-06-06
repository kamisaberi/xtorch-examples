#include <xtorch/xtorch.h>
#include <torch/torch.h>
#include <iostream>


using namespace std;


int main()
{
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
    auto device = torch::kCUDA;
    if (!torch::cuda::is_available())
    {
        std::cerr << "CUDA is not available. Using CPU." << std::endl;
        device = torch::kCPU;
    }

    // Initialize models
    xt::models::DCGAN::Generator netG(nz, ngf, nc);
    xt::models::DCGAN::Discriminator netD(nc, ndf);
    netG.to(device);
    netD.to(device);


    // Optimizers
    torch::optim::Adam optimG(netG.parameters(), torch::optim::AdamOptions(1e-3).betas({beta1, 0.999}));
    torch::optim::Adam optimD(netG.parameters(), torch::optim::AdamOptions(1e-3).betas({beta1, 0.999}));

    auto criterion = torch::nn::BCELoss();

    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(
        std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{127.5}, std::vector<float>{127.5}));

    auto compose = std::make_unique<xt::transforms::Compose>(transform_list);
    auto dataset = xt::datasets::MNIST("/home/kami/Documents/datasets/",
                                       xt::datasets::DataMode::TRAIN, false,
                                       std::move(compose));

    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 64, true, 2, /*prefetch_factor=*/2);

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        for (auto& batch : *data_loader)
        {
            // Train Discriminator
            netD.zero_grad();
            auto real_data = batch.data.to(device);
            auto batch_size = real_data.size(0);

            // Real images
            auto real_label = torch::full({batch_size}, 1.0, device);
            auto output = netD.forward(real_data).view(-1);
            auto errD_real = criterion(output, real_label);
            errD_real.backward();

            // Fake images
            auto noise = torch::randn({batch_size, nz, 1, 1}, device);
            auto fake_data = netG.forward(noise);
            auto fake_label = torch::full({batch_size}, 0.0, device);
            output = netD.forward(fake_data.detach()).view(-1);
            auto errD_fake = criterion(output, fake_label);
            errD_fake.backward();
            auto errD = errD_real + errD_fake;
            optimD.step();

            // Train Generator
            netG.zero_grad();
            output = netD.forward(fake_data).view(-1);
            auto errG = criterion(output, real_label); // Generator wants discriminator to think fakes are real
            errG.backward();
            optimG.step();

            // Print progress
            std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
                << "] D_Loss: " << errD.item<float>()
                << " G_Loss: " << errG.item<float>() << std::endl;
        }

        // Save generated images (optional, requires OpenCV or other image handling)
        if (epoch % 2 == 0)
        {
            auto noise = torch::randn({16, nz, 1, 1}, device);
            auto fake_images = netG.forward(noise);
            // Implement image saving logic here if needed
        }
    }
}
