#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random> // For dummy data generation

// --- Configuration ---
const int64_t IMG_H = 28;             // Image height
const int64_t IMG_W = 28;             // Image width
const int64_t INPUT_DIM = IMG_H * IMG_W; // Flattened image size
const int64_t HIDDEN_DIM = 256;       // Hidden layer size
const int64_t LATENT_DIM = 20;        // Dimensionality of the latent space
const int64_t BATCH_SIZE = 64;
const int64_t NUM_EPOCHS = 30;
const double LEARNING_RATE = 1e-3;
const int64_t LOG_INTERVAL = 10;
const double KLD_WEIGHT = 0.00025; // Weight for the KL divergence term (can be tuned)


// --- Variational Autoencoder Model ---
struct VAEImpl : torch::nn::Module {
    torch::nn::Linear fc_enc1{nullptr}, fc_enc2{nullptr};
    torch::nn::Linear fc_mu{nullptr}, fc_log_var{nullptr}; // To output mean and log_var of latent distribution
    torch::nn::Linear fc_dec1{nullptr}, fc_dec2{nullptr}, fc_dec_out{nullptr};

    VAEImpl(int64_t input_dim = INPUT_DIM, int64_t hidden_dim = HIDDEN_DIM, int64_t latent_dim = LATENT_DIM) {
        // Encoder
        fc_enc1 = register_module("fc_enc1", torch::nn::Linear(input_dim, hidden_dim));
        fc_enc2 = register_module("fc_enc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc_mu = register_module("fc_mu", torch::nn::Linear(hidden_dim, latent_dim));
        fc_log_var = register_module("fc_log_var", torch::nn::Linear(hidden_dim, latent_dim));

        // Decoder
        fc_dec1 = register_module("fc_dec1", torch::nn::Linear(latent_dim, hidden_dim));
        fc_dec2 = register_module("fc_dec2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc_dec_out = register_module("fc_dec_out", torch::nn::Linear(hidden_dim, input_dim));
    }

    // Encoder part: returns mu and log_var
    std::tuple<torch::Tensor, torch::Tensor> encode(torch::Tensor x) {
        x = torch::relu(fc_enc1(x));
        x = torch::relu(fc_enc2(x));
        torch::Tensor mu = fc_mu(x);
        torch::Tensor log_var = fc_log_var(x); // log_var to ensure variance is positive
        return {mu, log_var};
    }

    // Reparameterization trick: z = mu + std * epsilon
    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor log_var) {
        if (!this->is_training()) { // During inference, just use the mean
            return mu;
        }
        torch::Tensor std = torch::exp(0.5 * log_var);
        torch::Tensor epsilon = torch::randn_like(std); // Sample from N(0, I)
        return mu + std * epsilon;
    }

    // Decoder part
    torch::Tensor decode(torch::Tensor z) {
        z = torch::relu(fc_dec1(z));
        z = torch::relu(fc_dec2(z));
        // Use sigmoid for output if input pixels are normalized to [0,1]
        // (common for BCE loss with image data)
        return torch::sigmoid(fc_dec_out(z));
    }

    // Full forward pass
    // Returns: reconstructed_x, mu, log_var
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        auto [mu, log_var] = encode(x.view({-1, INPUT_DIM})); // Flatten input
        torch::Tensor z = reparameterize(mu, log_var);
        torch::Tensor reconstructed_x = decode(z);
        return {reconstructed_x, mu, log_var};
    }

    // For generation: sample from latent space and decode
    torch::Tensor generate(torch::Tensor z) {
        return decode(z);
    }
};
TORCH_MODULE(VAE);

// --- VAE Loss Function ---
// Reconstruction loss + KL divergence
// reconstruction_loss = MSE or BCE
// kld_loss = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
torch::Tensor vae_loss_function(
    const torch::Tensor& reconstructed_x,
    const torch::Tensor& x, // Original input (flattened)
    const torch::Tensor& mu,
    const torch::Tensor& log_var,
    double kld_weight = KLD_WEIGHT
) {
    // Reconstruction Loss (e.g., Binary Cross Entropy for [0,1] pixels)
    // BCE expects input and target to have the same shape.
    torch::Tensor recon_loss = torch::binary_cross_entropy(
        reconstructed_x, x.view({-1, INPUT_DIM}), torch::Reduction::Sum);
    // Or MSE: torch::mse_loss(reconstructed_x, x.view({-1, INPUT_DIM}), torch::Reduction::Sum);

    // KL Divergence (between learned N(mu, var) and N(0, I))
    // KLD = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp())
    torch::Tensor kld_loss = -0.5 * torch::sum(1 + log_var - mu.pow(2) - log_var.exp());

    return recon_loss + kld_weight * kld_loss;
}


// --- Dummy Dataset for VAE ---
// Generates simple "flattened images"
class DummyFlatImageDataset : public torch::data::datasets::Dataset<DummyFlatImageDataset> {
public:
    size_t dataset_size_;

    DummyFlatImageDataset(size_t size = 1000) : dataset_size_(size) {}

    // Returns a single flattened "image" (which is also the target)
    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override {
        // Generate a simple "image" (e.g., random pattern)
        // Ensure values are roughly in [0,1] if using BCE loss and sigmoid output
        torch::Tensor image = torch::rand({1, IMG_H, IMG_W}); // [0,1)

        // For VAE, input and target are the same (the clean image)
        return {image, image}; // Data and Target are the same
    }

    torch::optional<size_t> size() const override {
        return dataset_size_;
    }
};


int main() {
    std::cout << "Variational Autoencoder (VAE) for Latent Space Exploration (LibTorch C++)" << std::endl;

    torch::manual_seed(1);

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);
    std::cout << "Using device: " << device << std::endl;

    VAE model(INPUT_DIM, HIDDEN_DIM, LATENT_DIM);
    model->to(device);
    std::cout << "VAE model created." << std::endl;

    auto train_dataset = DummyFlatImageDataset(5000) // Larger dummy dataset
                             .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2)
    );
    std::cout << "DataLoader created." << std::endl;

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    std::cout << "Optimizer created." << std::endl;

    std::cout << "\nStarting Training..." << std::endl;
    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        model->train();
        double epoch_loss = 0.0;
        size_t batch_idx = 0;

        for (auto& batch : *train_loader) {
            optimizer.zero_grad();

            // Data and target are the same for autoencoders
            torch::Tensor data = batch.data.to(device); // [B, 1, H, W]
            // Target is also batch.data, will be flattened in loss function

            auto [reconstructed_x, mu, log_var] = model->forward(data);

            torch::Tensor loss = vae_loss_function(reconstructed_x, data, mu, log_var, KLD_WEIGHT);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();

            if (batch_idx % LOG_INTERVAL == 0) {
                std::cout << "Epoch: " << epoch << "/" << NUM_EPOCHS
                          << " | Batch: " << batch_idx << "/" << ( (5000 / BATCH_SIZE) )
                          << " | Loss: " << loss.item<double>() / data.size(0) // Per-sample loss
                          << std::endl;
            }
            batch_idx++;
        }
        double avg_epoch_loss = (batch_idx > 0) ? (epoch_loss / (batch_idx * BATCH_SIZE)) : 0.0; // Per-sample average
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "Epoch: " << epoch << " Average Loss: " << avg_epoch_loss << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
    }

    std::cout << "Training finished." << std::endl;

    // --- Latent Space Exploration (Generation) ---
    model->eval(); // Set to evaluation mode
    std::cout << "\nGenerating a few samples from latent space..." << std::endl;
    int num_samples_to_generate = 5;
    { // Scope for NoGradGuard
        torch::NoGradGuard no_grad; // Disable gradients for generation

        // Sample random points from N(0, I) in the latent space
        torch::Tensor z_samples = torch::randn({num_samples_to_generate, LATENT_DIM}, device);

        torch::Tensor generated_images_flat = model->generate(z_samples); // [num_samples, INPUT_DIM]

        // Reshape to image format (conceptual, as we don't visualize here)
        torch::Tensor generated_images = generated_images_flat.view({num_samples_to_generate, 1, IMG_H, IMG_W});
        std::cout << "Generated " << generated_images.size(0) << " images of shape " << generated_images.sizes() << std::endl;

        // In a real application, you'd save or display these generated_images.
        // For example, to print the first generated image (flattened, first 10 pixels):
        if (generated_images.size(0) > 0) {
            std::cout << "First 10 pixels of the first generated sample (flattened): \n"
                      << generated_images_flat[0].slice(0, 0, 10) << std::endl;
        }
    }

    // --- Save Model (Example) ---
    // try {
    //     torch::save(model, "vae_model.pt");
    //     std::cout << "Model saved to vae_model.pt" << std::endl;
    // } catch (const c10::Error& e) {
    //     std::cerr << "Error saving model: " << e.what() << std::endl;
    // }

    return 0;
}