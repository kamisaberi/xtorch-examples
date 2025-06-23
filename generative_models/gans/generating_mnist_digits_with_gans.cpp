#include <torch/torch.h>
#include <torch/data/datasets/mnist.h> // For MNIST dataset
#include <torch/data/transforms/tensor.h> // For ToTensor
#include <torch/data/transforms/normalize.h> // For Normalize
#include <iostream>
#include <vector>
#include <iomanip> // For std::fixed and std::setprecision

// --- Configuration ---
const int64_t IMG_H = 28;
const int64_t IMG_W = 28;
const int64_t IMG_CHANNELS = 1;
const int64_t FLATTENED_DIM = IMG_H * IMG_W * IMG_CHANNELS;
const int64_t LATENT_DIM = 100;       // Dimensionality of the input noise for Generator
const int64_t HIDDEN_DIM_G = 256;     // Hidden layer size for Generator
const int64_t HIDDEN_DIM_D = 128;     // Hidden layer size for Discriminator
const int64_t BATCH_SIZE = 128;
const int64_t NUM_EPOCHS = 50;        // GANs often need many epochs
const double LR_G = 2e-4;             // Learning rate for Generator
const double LR_D = 2e-4;             // Learning rate for Discriminator
const double BETA1 = 0.5;             // Beta1 for Adam optimizer (common for GANs)
const int64_t LOG_INTERVAL = 100;     // Log every N batches
const std::string MNIST_DATA_PATH = "./mnist_data"; // Path to download/store MNIST

// --- Generator Model (MLP) ---
struct GeneratorImpl : torch::nn::Module {
    torch::nn::Sequential net{nullptr};

    GeneratorImpl(int64_t latent_dim = LATENT_DIM, int64_t hidden_dim = HIDDEN_DIM_G, int64_t output_dim = FLATTENED_DIM) {
        net = torch::nn::Sequential(
            torch::nn::Linear(latent_dim, hidden_dim),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
            torch::nn::Linear(hidden_dim, hidden_dim * 2),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
            torch::nn::Linear(hidden_dim * 2, output_dim),
            torch::nn::Tanh() // Output pixels in [-1, 1] (MNIST will be normalized to this range)
        );
        register_module("net", net);
    }

    torch::Tensor forward(torch::Tensor x) {
        return net->forward(x);
    }
};
TORCH_MODULE(Generator);

// --- Discriminator Model (MLP) ---
struct DiscriminatorImpl : torch::nn::Module {
    torch::nn::Sequential net{nullptr};

    DiscriminatorImpl(int64_t input_dim = FLATTENED_DIM, int64_t hidden_dim = HIDDEN_DIM_D) {
        net = torch::nn::Sequential(
            torch::nn::Linear(input_dim, hidden_dim * 2),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
            // torch::nn::Dropout(0.3), // Optional dropout for regularization
            torch::nn::Linear(hidden_dim * 2, hidden_dim),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
            // torch::nn::Dropout(0.3),
            torch::nn::Linear(hidden_dim, 1),
            torch::nn::Sigmoid() // Output a probability (real or fake)
        );
        register_module("net", net);
    }

    torch::Tensor forward(torch::Tensor x) {
        return net->forward(x);
    }
};
TORCH_MODULE(Discriminator);


// Function to save a batch of generated images (requires OpenCV or similar)
// This is a placeholder; you'd need to implement it with an image library.
void save_generated_images(const torch::Tensor& images, int64_t epoch, int64_t batch_idx, int num_to_save = 16) {
    if (images.size(0) == 0) return;
    std::cout << "[INFO] Placeholder for saving " << std::min(num_to_save, (int)images.size(0))
              << " images for epoch " << epoch << ", batch " << batch_idx << std::endl;
    // Example:
    // for (int i = 0; i < std::min(num_to_save, (int)images.size(0)); ++i) {
    //     torch::Tensor single_image = images[i].cpu().clone(); // [1, H, W] or [H, W]
    //     // Rescale from [-1, 1] to [0, 255]
    //     single_image = ((single_image + 1.0) / 2.0) * 255.0;
    //     single_image = single_image.to(torch::kByte);
    //     // ... convert to cv::Mat and use cv::imwrite ...
    //     // std::string filename = "generated_epoch_" + std::to_string(epoch) + "_batch_" + std::to_string(batch_idx) + "_img_" + std::to_string(i) + ".png";
    //     // cv::imwrite(filename, image_mat);
    // }
}


int main() {
    std::cout << "GAN for MNIST Digit Generation (LibTorch C++)" << std::endl;
    torch::manual_seed(1);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    // --- Data Loading (MNIST) ---
    // MNIST images are 28x28 grayscale. We'll normalize them to [-1, 1].
    // Mean and std for MNIST [0,1] are approx 0.1307 and 0.3081.
    // To map to [-1, 1], we can use: (x - 0.5) / 0.5 = 2x - 1.
    // So, if input is [0,1], new_mean = (0.1307 - 0.5) / 0.5 = -0.7386
    // new_std = 0.3081 / 0.5 = 0.6162
    // A simpler way: normalize to [0,1] then transform using x' = (x - 0.5) * 2
    // For LibTorch's Normalize, it's (input - mean) / std.
    // If we want output in [-1, 1], we need target_mean=0, target_std=1 (after (x-0.5)*2 mapping).
    // Let input be X ~ [0,1]. Y = (X - 0.5) / 0.5. E[Y] = (E[X]-0.5)/0.5. Var[Y] = Var[X]/(0.5)^2
    // So if X ~ N(0.1307, 0.3081^2), then Y has mean (0.1307-0.5)/0.5 = -0.7386, std 0.3081/0.5 = 0.6162
    // It's often easier to just scale to [-1,1] directly: image = (image / 255.0 - 0.5) * 2.0
    // Or use Normalize({0.5}, {0.5}) on [0,1] data.
    auto train_dataset = torch::data::datasets::MNIST(
        MNIST_DATA_PATH,
        torch::data::datasets::MNIST::Mode::kTrain)
        .map(torch::data::transforms::Normalize<>(0.5, 0.5)) // Normalizes (image - 0.5) / 0.5. Input is [0,1] from ToTensor.
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));
    std::cout << "MNIST DataLoader created." << std::endl;

    // --- Models ---
    Generator generator(LATENT_DIM, HIDDEN_DIM_G, FLATTENED_DIM);
    Discriminator discriminator(FLATTENED_DIM, HIDDEN_DIM_D);
    generator->to(device);
    discriminator->to(device);
    std::cout << "Generator and Discriminator models created." << std::endl;

    // --- Optimizers ---
    // Adam optimizer is common for GANs. Beta1=0.5 can help stabilize.
    torch::optim::Adam optimizer_G(generator->parameters(), torch::optim::AdamOptions(LR_G).beta1(BETA1));
    torch::optim::Adam optimizer_D(discriminator->parameters(), torch::optim::AdamOptions(LR_D).beta1(BETA1));
    std::cout << "Optimizers created." << std::endl;

    // --- Loss Function ---
    // Binary Cross Entropy for GANs
    torch::nn::BCELoss criterion;
    std::cout << "Loss function (BCELoss) created." << std::endl;

    // --- Training Loop ---
    std::cout << "\nStarting Training..." << std::endl;
    std::cout << std::fixed << std::setprecision(4); // For printing losses

    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        generator->train();
        discriminator->train();
        double epoch_loss_G = 0.0;
        double epoch_loss_D = 0.0;
        size_t batch_idx = 0;

        for (auto& batch : *train_loader) {
            // Data comes as [B, 1, H, W], target is class labels (not used here)
            torch::Tensor real_images = batch.data.to(device);
            int64_t current_batch_size = real_images.size(0);
            real_images = real_images.view({current_batch_size, -1}); // Flatten images [B, FLATTENED_DIM]

            // Create labels for real and fake images
            torch::Tensor real_labels = torch::ones({current_batch_size, 1}, device);
            torch::Tensor fake_labels = torch::zeros({current_batch_size, 1}, device);

            // ---------------------
            //  Train Discriminator
            // ---------------------
            optimizer_D.zero_grad();

            // Loss for real images
            torch::Tensor output_real = discriminator->forward(real_images);
            torch::Tensor loss_D_real = criterion(output_real, real_labels);

            // Generate fake images
            torch::Tensor noise = torch::randn({current_batch_size, LATENT_DIM}, device);
            torch::Tensor fake_images = generator->forward(noise);

            // Loss for fake images (detach generator output so gradients don't flow to G)
            torch::Tensor output_fake = discriminator->forward(fake_images.detach());
            torch::Tensor loss_D_fake = criterion(output_fake, fake_labels);

            // Total discriminator loss
            torch::Tensor loss_D = (loss_D_real + loss_D_fake) / 2.0;
            loss_D.backward();
            optimizer_D.step();
            epoch_loss_D += loss_D.item<double>();

            // -----------------
            //  Train Generator
            // -----------------
            optimizer_G.zero_grad();

            // We want discriminator to classify fake images as real
            // (i.e., generator wants D(G(z)) to be close to 1)
            // Need to re-generate fake_images or use the ones from before,
            // but this time without detaching, so gradients flow back.
            // For efficiency, can reuse fake_images if not detached. Here we'll re-run D.
            output_fake = discriminator->forward(fake_images); // Pass through D again (fake_images NOT detached)
            torch::Tensor loss_G = criterion(output_fake, real_labels); // Generator wants D to output 'real' for its fakes

            loss_G.backward();
            optimizer_G.step();
            epoch_loss_G += loss_G.item<double>();

            if (batch_idx % LOG_INTERVAL == 0) {
                std::cout << "Epoch: " << epoch << "/" << NUM_EPOCHS
                          << " | Batch: " << batch_idx << "/" << (train_dataset.size().value() / BATCH_SIZE)
                          << " | Loss_D: " << loss_D.item<double>()
                          << " | Loss_G: " << loss_G.item<double>()
                          << " | D(real): " << output_real.mean().item<double>()
                          << " | D(fake)_before_G_update: " << discriminator->forward(fake_images.detach()).mean().item<double>() // D output for G's fakes
                          << " | D(fake)_after_G_update: " << output_fake.mean().item<double>() // D output for G's fakes after G update
                          << std::endl;
            }
            batch_idx++;
        }
        std::cout << "--------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Epoch: " << epoch << " Avg Loss_D: " << (epoch_loss_D / batch_idx)
                  << " | Avg Loss_G: " << (epoch_loss_G / batch_idx) << std::endl;
        std::cout << "--------------------------------------------------------------------------------------" << std::endl;

        // Save some generated images periodically
        if (epoch % 5 == 0 || epoch == NUM_EPOCHS) {
            generator->eval(); // Set generator to eval mode for consistent generation
            torch::NoGradGuard no_grad;
            torch::Tensor fixed_noise = torch::randn({16, LATENT_DIM}, device);
            torch::Tensor samples = generator->forward(fixed_noise); // [16, FLATTENED_DIM]
            samples = samples.view({16, IMG_CHANNELS, IMG_H, IMG_W}); // Reshape to image format
            save_generated_images(samples.cpu(), epoch, 0); // Pass CPU tensor
            generator->train(); // Back to train mode
        }
    }

    std::cout << "Training finished." << std::endl;

    // --- Save Models (Example) ---
    // try {
    //     torch::save(generator, "generator_mnist_gan.pt");
    //     torch::save(discriminator, "discriminator_mnist_gan.pt");
    //     std::cout << "Models saved." << std::endl;
    // } catch (const c10::Error& e) {
    //     std::cerr << "Error saving models: " << e.what() << std::endl;
    // }

    return 0;
}