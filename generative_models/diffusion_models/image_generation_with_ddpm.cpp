#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath> // For M_PI, sin, cos, pow

// --- Configuration ---
const int64_t IMG_SIZE = 32;          // Input image size (e.g., 32x32)
const int64_t IMG_CHANNELS = 1;       // Grayscale images
const int64_t BATCH_SIZE = 8;        // Keep small for example due to U-Net size
const int64_t NUM_EPOCHS = 50;       // DDPMs need many epochs
const double LEARNING_RATE = 2e-4;   // Common LR for DDPMs
const int64_t LOG_INTERVAL = 10;

const int T_TIMESTEPS = 100;         // Number of diffusion timesteps (small for example)
const double BETA_START = 1e-4;
const double BETA_END = 0.02;

// --- Diffusion Schedule Helpers ---
// Linearly spaced beta values
torch::Tensor linear_beta_schedule(int timesteps, double start = BETA_START, double end = BETA_END) {
    return torch::linspace(start, end, timesteps);
}

// Helper to extract specific values from a tensor for a batch of timesteps t
// result_shape should be like (batch_size, 1, 1, 1) for broadcasting with images
torch::Tensor extract(const torch::Tensor& a, const torch::Tensor& t, const torch::IntArrayRef& result_shape) {
    int64_t batch_size = t.size(0);
    torch::Tensor out = torch::gather(a, 0, t.to(a.device()).to(torch::kLong)); // t needs to be long for gather
    return out.reshape(result_shape);
}

// --- Timestep Embedding ---
// From "Attention Is All You Need"
struct SinusoidalPositionalEmbeddingImpl : torch::nn::Module {
    int64_t dim;
    torch::Tensor pe; // Positional encoding tensor

    SinusoidalPositionalEmbeddingImpl(int64_t embedding_dim, int64_t max_len = T_TIMESTEPS + 1)
        : dim(embedding_dim) {
        pe = torch::zeros({max_len, dim});
        torch::Tensor position = torch::arange(0, max_len, torch::kFloat).unsqueeze(1);
        torch::Tensor div_term = torch::exp(torch::arange(0, dim, 2, torch::kFloat) * (-std::log(10000.0) / dim));

        pe.index_put_({"...", torch::indexing::Slice(0, torch::indexing::None, 2)}, torch::sin(position * div_term));
        pe.index_put_({"...", torch::indexing::Slice(1, torch::indexing::None, 2)}, torch::cos(position * div_term));
        pe = register_buffer("pe", pe); // Register as buffer, not parameter
    }

    torch::Tensor forward(torch::Tensor t) { // t is a tensor of timesteps [B]
        // Ensure t is long and on the same device as pe
        return pe.index_select(0, t.to(pe.device()).to(torch::kLong)); // [B, dim]
    }
};
TORCH_MODULE(SinusoidalPositionalEmbedding);

// --- Simplified U-Net Components ---
// Basic Conv Block: Conv -> BatchNorm (optional) -> Activation
struct ConvBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr}; // Optional, GroupNorm is often better for DDPMs
    torch::nn::SiLU act{nullptr}; // SiLU (Swish) is common in DDPMs

    ConvBlockImpl(int64_t in_channels, int64_t out_channels, int kernel_size = 3, int padding = 1, bool use_bn = true) {
        conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(padding)));
        if (use_bn) {
            bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
        }
        act = register_module("act", torch::nn::SiLU());
    }
    torch::Tensor forward(torch::Tensor x) {
        x = conv(x);
        if (bn) x = bn(x);
        return act(x);
    }
};
TORCH_MODULE(ConvBlock);

// Downsampling Block
struct DownBlockImpl : torch::nn::Module {
    ConvBlock conv1{nullptr}, conv2{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    torch::nn::Linear time_mlp{nullptr}; // To project time embedding

    DownBlockImpl(int64_t in_ch, int64_t out_ch, int64_t time_emb_dim) {
        conv1 = register_module("conv1", ConvBlock(in_ch, out_ch));
        conv2 = register_module("conv2", ConvBlock(out_ch, out_ch)); // Second conv keeps out_ch
        pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
        time_mlp = register_module("time_mlp", torch::nn::Linear(time_emb_dim, out_ch)); // Project time_emb to match feature channels
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor time_emb) {
        x = conv1(x);
        // Add time embedding (broadcast across H, W)
        torch::Tensor time_projection = time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1); // [B, out_ch, 1, 1]
        x = x + time_projection;
        x = conv2(x);
        torch::Tensor skip_connection = x; // Save for skip connection
        x = pool(x);
        return {x, skip_connection};
    }
};
TORCH_MODULE(DownBlock);

// Upsampling Block
struct UpBlockImpl : torch::nn::Module {
    torch::nn::ConvTranspose2d upsample{nullptr};
    ConvBlock conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear time_mlp{nullptr};

    UpBlockImpl(int64_t in_ch, int64_t out_ch, int64_t time_emb_dim) {
        // in_ch for upsample will be from the previous up_block or bottleneck
        // after concatenation with skip connection, channels for conv1 will be in_ch (from prev_up) + skip_ch
        upsample = register_module("upsample", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(in_ch, out_ch, 2).stride(2))); // Kernel 2, Stride 2 for 2x upsampling

        conv1 = register_module("conv1", ConvBlock(out_ch + out_ch, out_ch)); // out_ch from upsample + out_ch from skip
        conv2 = register_module("conv2", ConvBlock(out_ch, out_ch));
        time_mlp = register_module("time_mlp", torch::nn::Linear(time_emb_dim, out_ch));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor skip_conn, torch::Tensor time_emb) {
        x = upsample(x); // x: [B, out_ch, H, W]
        x = torch::cat({x, skip_conn}, 1); // skip_conn should have out_ch channels
        x = conv1(x);
        torch::Tensor time_projection = time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1);
        x = x + time_projection;
        x = conv2(x);
        return x;
    }
};
TORCH_MODULE(UpBlock);


// --- Simplified U-Net for Noise Prediction (`epsilon_theta`) ---
struct UNetModelImpl : torch::nn::Module {
    SinusoidalPositionalEmbedding time_embedder{nullptr};
    ConvBlock initial_conv{nullptr};
    DownBlock down1{nullptr}, down2{nullptr};
    ConvBlock bottleneck_conv1{nullptr}, bottleneck_conv2{nullptr};
    UpBlock up1{nullptr}, up2{nullptr};
    torch::nn::Conv2d final_conv{nullptr};

    int64_t time_emb_dim_;

    UNetModelImpl(int64_t img_channels = IMG_CHANNELS, int64_t base_channels = 32, int64_t time_emb_dim = 128)
        : time_emb_dim_(time_emb_dim) {
        time_embedder = register_module("time_embedder", SinusoidalPositionalEmbedding(time_emb_dim_));

        initial_conv = register_module("initial_conv", ConvBlock(img_channels, base_channels)); // 1 -> 32

        down1 = register_module("down1", DownBlock(base_channels, base_channels * 2, time_emb_dim_));         // 32 -> 64
        down2 = register_module("down2", DownBlock(base_channels * 2, base_channels * 4, time_emb_dim_));     // 64 -> 128

        bottleneck_conv1 = register_module("bottleneck_conv1", ConvBlock(base_channels * 4, base_channels * 4)); // 128 -> 128
        bottleneck_conv2 = register_module("bottleneck_conv2", ConvBlock(base_channels * 4, base_channels * 4)); // 128 -> 128

        // UpBlock in_ch is output of previous upsample, out_ch is the target channel before skip concat
        up1 = register_module("up1", UpBlock(base_channels * 4, base_channels * 2, time_emb_dim_));       // 128 -> 64 (skip from down2 is 64)
        up2 = register_module("up2", UpBlock(base_channels * 2, base_channels, time_emb_dim_));         // 64 -> 32  (skip from down1 is 32)

        final_conv = register_module("final_conv", torch::nn::Conv2d(base_channels, img_channels, 1)); // 32 -> 1
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor t) { // x: [B, C, H, W], t: [B]
        torch::Tensor time_emb = time_embedder(t); // [B, time_emb_dim]

        x = initial_conv(x); // [B, base_ch, H, W]

        auto [d1_out, skip1] = down1(x, time_emb);      // d1_out: [B, base*2, H/2, W/2], skip1: [B, base*2, H, W]
        auto [d2_out, skip2] = down2(d1_out, time_emb); // d2_out: [B, base*4, H/4, W/4], skip2: [B, base*4, H/2, W/2]

        torch::Tensor bn = bottleneck_conv1(d2_out);
        bn = bottleneck_conv2(bn); // [B, base*4, H/4, W/4]

        torch::Tensor u1_out = up1(bn, skip2, time_emb);      // [B, base*2, H/2, W/2]
        torch::Tensor u2_out = up2(u1_out, skip1, time_emb);  // [B, base, H, W]

        return final_conv(u2_out); // [B, img_channels, H, W]
    }
};
TORCH_MODULE(UNetModel);


// --- Dummy Dataset ---
class DummyImageDataset : public torch::data::datasets::Dataset<DummyImageDataset> {
public:
    size_t dataset_size_;
    DummyImageDataset(size_t size = 1000) : dataset_size_(size) {}

    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override {
        // Random image, values in [-1, 1] (common for DDPMs, if not, adjust U-Net output activation)
        torch::Tensor image = torch::rand({IMG_CHANNELS, IMG_SIZE, IMG_SIZE}) * 2.0 - 1.0;
        return {image, image}; // Data and target are the same (original image)
    }
    torch::optional<size_t> size() const override { return dataset_size_; }
};


int main() {
    std::cout << "DDPM Image Generation (Conceptual - LibTorch C++)" << std::endl;
    torch::manual_seed(1);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    // --- Diffusion Schedule Precomputation ---
    torch::Tensor betas = linear_beta_schedule(T_TIMESTEPS).to(device);
    torch::Tensor alphas = 1.0 - betas;
    torch::Tensor alphas_cumprod = torch::cumprod(alphas, 0);
    torch::Tensor alphas_cumprod_prev = torch::cat({torch::tensor({1.0}, device), alphas_cumprod.slice(0, 0, -1)}, 0);

    torch::Tensor sqrt_alphas_cumprod = torch::sqrt(alphas_cumprod);
    torch::Tensor sqrt_one_minus_alphas_cumprod = torch::sqrt(1.0 - alphas_cumprod);
    torch::Tensor posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod); // beta_tilde_t

    // --- Model & Optimizer ---
    UNetModel model(IMG_CHANNELS, 32, 128); // base_channels=32, time_emb_dim=128
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    std::cout << "U-Net model created." << std::endl;

    // --- Dataloader ---
    auto train_dataset = DummyImageDataset(128).map(torch::data::transforms::Stack<>()); // Small dataset for demo
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(BATCH_SIZE));
    std::cout << "DataLoader created." << std::endl;

    // --- Training Loop ---
    std::cout << "\nStarting Training..." << std::endl;
    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        model->train();
        double epoch_loss = 0.0;
        size_t batch_idx = 0;

        for (auto& batch : *train_loader) {
            optimizer.zero_grad();
            torch::Tensor x_start = batch.data.to(device); // [B, C, H, W] Clean images

            // Sample timesteps t for the batch
            torch::Tensor t = torch::randint(0, T_TIMESTEPS, {x_start.size(0)}, device).to(torch::kLong);

            // Sample noise epsilon
            torch::Tensor noise = torch::randn_like(x_start);

            // Create noisy images x_t using q_sample: x_t = sqrt(alpha_bar_t)x_0 + sqrt(1-alpha_bar_t)epsilon
            torch::Tensor sqrt_alpha_bar_t = extract(sqrt_alphas_cumprod, t, {x_start.size(0), 1, 1, 1});
            torch::Tensor sqrt_one_minus_alpha_bar_t = extract(sqrt_one_minus_alphas_cumprod, t, {x_start.size(0), 1, 1, 1});
            torch::Tensor x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise;

            // Predict noise using U-Net
            torch::Tensor predicted_noise = model->forward(x_t, t);

            // Loss: MSE between actual noise and predicted noise
            torch::Tensor loss = torch::mse_loss(predicted_noise, noise);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
            if (batch_idx % LOG_INTERVAL == 0) {
                std::cout << "Epoch: " << epoch << "/" << NUM_EPOCHS
                          << " | Batch: " << batch_idx << "/" << ( (128 / BATCH_SIZE) )
                          << " | Loss: " << loss.item<double>() << std::endl;
            }
            batch_idx++;
        }
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "Epoch: " << epoch << " Average Loss: " << (epoch_loss / batch_idx) << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
    }
    std::cout << "Training finished." << std::endl;

    // --- Sampling (Image Generation) ---
    std::cout << "\nStarting Sampling..." << std::endl;
    model->eval();
    int num_images_to_sample = 4;
    torch::Tensor sampled_images; // Will hold the final images
    {
        torch::NoGradGuard no_grad;
        torch::Tensor img = torch::randn({num_images_to_sample, IMG_CHANNELS, IMG_SIZE, IMG_SIZE}, device); // Start with x_T from N(0,I)

        for (int i = T_TIMESTEPS - 1; i >= 0; --i) {
            torch::Tensor t_batch = torch::full({num_images_to_sample}, i, device).to(torch::kLong);

            torch::Tensor pred_noise = model->forward(img, t_batch);

            torch::Tensor alpha_t_val = extract(alphas, t_batch, {num_images_to_sample, 1, 1, 1});
            torch::Tensor alpha_bar_t_val = extract(alphas_cumprod, t_batch, {num_images_to_sample, 1, 1, 1});
            torch::Tensor beta_t_val = extract(betas, t_batch, {num_images_to_sample, 1, 1, 1});
            torch::Tensor sqrt_one_minus_alpha_bar_t_val = extract(sqrt_one_minus_alphas_cumprod, t_batch, {num_images_to_sample, 1, 1, 1});
            torch::Tensor posterior_var_t_val = extract(posterior_variance, t_batch, {num_images_to_sample, 1, 1, 1});

            // Equation for mu_theta(x_t, t)
            torch::Tensor model_mean = (1.0 / torch::sqrt(alpha_t_val)) *
                                       (img - (beta_t_val / sqrt_one_minus_alpha_bar_t_val) * pred_noise);

            if (i == 0) {
                img = model_mean; // No noise at the last step
            } else {
                torch::Tensor noise_z = torch::randn_like(img);
                img = model_mean + torch::sqrt(posterior_var_t_val) * noise_z;
            }
            if (i % (T_TIMESTEPS/5) == 0 || i < 5) std::cout << "Sampling timestep " << i << std::endl;
        }
        sampled_images = img;
    }
    std::cout << "Sampling finished. Sampled images tensor shape: " << sampled_images.sizes() << std::endl;
    // Here you would save or display the `sampled_images`. For example, print a slice:
    if (sampled_images.defined() && sampled_images.numel() > 0) {
         std::cout << "First few values of the first sampled image:\n"
                   << sampled_images[0].flatten().slice(0,0,10) << std::endl;
    }

    // torch::save(model, "ddpm_model.pt");
    return 0;
}