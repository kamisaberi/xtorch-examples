#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random> // For dummy data generation

// --- Configuration ---
const int64_t IMG_SIZE = 32;          // Input image size (e.g., 32x32)
const int64_t BATCH_SIZE = 16;
const int64_t NUM_EPOCHS = 20;        // More epochs usually needed for good results
const double LEARNING_RATE = 1e-3;
const int64_t LOG_INTERVAL = 10;
const double NOISE_FACTOR = 0.3;      // Amount of noise to add
const int64_t LATENT_DIM = 64;        // Example latent dimension (not explicitly used in pure CNN AE like this, but conceptual)

// --- Denoising Autoencoder Model ---
struct DenoisingAutoencoderImpl : torch::nn::Module {
    torch::nn::Sequential encoder{nullptr};
    torch::nn::Sequential decoder{nullptr};

    DenoisingAutoencoderImpl() {
        // Encoder: Compresses the image
        // Input: [B, 1, IMG_SIZE, IMG_SIZE] (e.g., 32x32)
        encoder = torch::nn::Sequential(
            // Layer 1: 32x32 -> 16x16
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(2).padding(1)), // (32-3+2*1)/2 + 1 = 16
            torch::nn::ReLU(),
            // Layer 2: 16x16 -> 8x8
            torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(2).padding(1)),// (16-3+2*1)/2 + 1 = 8
            torch::nn::ReLU(),
            // Layer 3: 8x8 -> 4x4 (bottleneck features)
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)) // (8-3+2*1)/2 + 1 = 4
            // Latent representation will be [B, 64, 4, 4]
        );
        register_module("encoder", encoder);

        // Decoder: Reconstructs the image from latent representation
        // Input: [B, 64, 4, 4]
        decoder = torch::nn::Sequential(
            // Layer 1: 4x4 -> 8x8
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 3).stride(2).padding(1).output_padding(1)),
            torch::nn::ReLU(),
            // Layer 2: 8x8 -> 16x16
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(32, 16, 3).stride(2).padding(1).output_padding(1)),
            torch::nn::ReLU(),
            // Layer 3: 16x16 -> 32x32
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(16, 1, 3).stride(2).padding(1).output_padding(1)),
            torch::nn::Sigmoid() // Output pixels between 0 and 1
        );
        register_module("decoder", decoder);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = encoder->forward(x);
        x = decoder->forward(x);
        return x;
    }
};
TORCH_MODULE(DenoisingAutoencoder);

// --- Dummy Dataset for Denoising ---
// Generates clean images and adds noise to them.
class DummyImageDataset : public torch::data::datasets::Dataset<DummyImageDataset> {
public:
    size_t dataset_size_;
    double noise_factor_;

    DummyImageDataset(size_t size = 1000, double noise_factor = NOISE_FACTOR)
        : dataset_size_(size), noise_factor_(noise_factor) {}

    // Returns a pair: {noisy_image, clean_image}
    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override {
        // Generate a simple "clean" image (e.g., random, or could be patterns)
        // For simplicity, let's make a random image and normalize it to [0,1]
        torch::Tensor clean_image = torch::rand({1, IMG_SIZE, IMG_SIZE}); // Values in [0, 1)

        // Add Gaussian noise
        torch::Tensor noise = torch::randn_like(clean_image) * noise_factor_;
        torch::Tensor noisy_image = clean_image + noise;

        // Clip values to be in [0, 1] range after adding noise
        noisy_image = torch::clamp(noisy_image, 0.0, 1.0);
        clean_image = torch::clamp(clean_image, 0.0, 1.0); // Should already be, but good practice

        return {noisy_image, clean_image};
    }

    torch::optional<size_t> size() const override {
        return dataset_size_;
    }
};


int main() {
    std::cout << "Denoising Autoencoder for Image Restoration (LibTorch C++)" << std::endl;

    torch::manual_seed(1); // For reproducibility

    // --- Device ---
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // --- Model ---
    DenoisingAutoencoder model;
    model->to(device);
    std::cout << "Model created successfully." << std::endl;

    // --- DataLoaders ---
    // For a real dataset, you'd load images here (e.g., MNIST, CelebA)
    // and apply transformations + noise.
    auto train_dataset = DummyImageDataset(1024, NOISE_FACTOR) // Larger dummy dataset
                             .map(torch::data::transforms::Stack<>()); // Default collate stacks samples

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2) // .workers(0) on Windows if issues
    );
    std::cout << "DataLoader created." << std::endl;

    // --- Optimizer ---
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    std::cout << "Optimizer created." << std::endl;

    // --- Loss Function ---
    // Mean Squared Error is common for image reconstruction
    torch::nn::MSELoss criterion;
    std::cout << "Loss function (MSELoss) created." << std::endl;

    // --- Training Loop ---
    std::cout << "\nStarting Training..." << std::endl;
    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        model->train(); // Set model to training mode
        size_t batch_idx = 0;
        double epoch_loss = 0.0;
        int batch_count_for_loss_avg = 0;

        for (auto& batch : *train_loader) {
            optimizer.zero_grad();

            torch::Tensor noisy_images = batch.data.to(device);
            torch::Tensor clean_images = batch.target.to(device); // Target is the clean image

            // Forward pass: Model reconstructs from noisy input
            torch::Tensor reconstructed_images = model->forward(noisy_images);

            // Compute loss against the clean image
            torch::Tensor loss = criterion(reconstructed_images, clean_images);

            // Backward pass and optimize
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
            batch_count_for_loss_avg++;

            if (batch_idx % LOG_INTERVAL == 0) {
                std::cout << "Epoch: " << epoch << "/" << NUM_EPOCHS
                          << " | Batch: " << batch_idx << "/" << ( (1024 / BATCH_SIZE) ) // Update dataset size if it changes
                          << " | Loss: " << loss.item<double>() << std::endl;
            }
            batch_idx++;
        }
        double avg_epoch_loss = (batch_count_for_loss_avg > 0) ? (epoch_loss / batch_count_for_loss_avg) : 0.0;
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "Epoch: " << epoch << " Average Loss: " << avg_epoch_loss << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;

        // Optional: Save a few example images every N epochs to see progress
        // This would require a library like OpenCV or saving raw tensor data.
        // For example, if you had OpenCV:
        // if (epoch % 5 == 0 && batch_count_for_loss_avg > 0) {
        //     torch::Tensor sample_noisy = train_dataset.get(0).data.to(device).unsqueeze(0);
        //     torch::Tensor sample_clean = train_dataset.get(0).target.to(device).unsqueeze(0);
        //     model->eval();
        //     torch::Tensor sample_reconstructed = model->forward(sample_noisy);
        //     model->train();
        //     // Convert tensors to cv::Mat and save using cv::imwrite
        //     // (requires converting [0,1] float to [0,255] uchar, handling channels, etc.)
        //     std::cout << "Saved sample images for epoch " << epoch << std::endl;
        // }
    }

    std::cout << "Training finished." << std::endl;

    // --- Save Model (Example) ---
    // try {
    //     torch::save(model, "denoising_autoencoder_model.pt");
    //     std::cout << "Model saved to denoising_autoencoder_model.pt" << std::endl;
    // } catch (const c10::Error& e) {
    //     std::cerr << "Error saving model: " << e.what() << std::endl;
    // }

    // --- Example Denoising (Inference) ---
    // model->eval(); // Set to evaluation mode
    // auto test_sample = DummyImageDataset(1, NOISE_FACTOR + 0.1).get(0); // Get a new noisy sample
    // torch::Tensor noisy_input = test_sample.data.to(device).unsqueeze(0); // Add batch dimension
    // torch::Tensor clean_target = test_sample.target.to(device).unsqueeze(0);
    //
    // torch::NoGradGuard no_grad; // Disable gradient calculations for inference
    // torch::Tensor denoised_output = model->forward(noisy_input);
    //
    // torch::Tensor inference_loss = criterion(denoised_output, clean_target);
    // std::cout << "\nExample Denoising:" << std::endl;
    // std::cout << "Inference MSE Loss on a sample: " << inference_loss.item<double>() << std::endl;
    // You could save noisy_input, denoised_output, and clean_target to visualize them.

    return 0;
}