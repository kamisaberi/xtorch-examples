#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm> // For std::shuffle, std::remove, std::sort, std::unique
#include <cmath>     // For std::sqrt, std::pow, std::sin, std::cos
#include <random>

// --- Helper: Character Tokenizer ---
struct CharTokenizer {
    std::map<char, int> char_to_int;
    std::map<int, char> int_to_char;
    int vocab_size = 0;

    CharTokenizer(const std::string& text) {
        std::string sorted_text = text;
        std::sort(sorted_text.begin(), sorted_text.end());
        sorted_text.erase(std::unique(sorted_text.begin(), sorted_text.end()), sorted_text.end());

        for (char c : sorted_text) {
            if (char_to_int.find(c) == char_to_int.end()) {
                char_to_int[c] = vocab_size;
                int_to_char[vocab_size] = c;
                vocab_size++;
            }
        }
        std::cout << "Vocabulary size: " << vocab_size << std::endl;
        // for(auto const& [key, val] : char_to_int) {
        //     std::cout << key << ":" << val << " ";
        // }
        // std::cout << std::endl;
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> encoded;
        for (char c : text) {
            encoded.push_back(char_to_int[c]);
        }
        return encoded;
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string decoded;
        for (int token : tokens) {
            decoded += int_to_char[token];
        }
        return decoded;
    }
};

// --- Positional Encoding ---
struct PositionalEncodingImpl : torch::nn::Module {
    torch::Tensor pe;

    PositionalEncodingImpl(int64_t d_model, int64_t max_len = 5000, double dropout_p = 0.1) {
        pe = torch::zeros({max_len, d_model});
        torch::Tensor position = torch::arange(0, max_len, torch::kFloat).unsqueeze(1);
        torch::Tensor div_term = torch::exp(torch::arange(0, d_model, 2, torch::kFloat) * (-std::log(10000.0) / d_model));

        pe.index_put_({"...", torch::indexing::Slice(0, torch::indexing::None, 2)}, torch::sin(position * div_term));
        pe.index_put_({"...", torch::indexing::Slice(1, torch::indexing::None, 2)}, torch::cos(position * div_term));
        pe = pe.unsqueeze(0); // Add batch dimension: [1, max_len, d_model]
        register_buffer("pe", pe); // Not a parameter, but part of state
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [batch_size, seq_len, d_model]
        // self.pe: [1, max_len, d_model]
        // We need self.pe[:, :seq_len, :]
        return x + pe.index({torch::indexing::Slice(), torch::indexing::Slice(0, x.size(1)), torch::indexing::Slice()});
    }
};
TORCH_MODULE(PositionalEncoding);

// --- Multi-Head Attention ---
struct MultiHeadAttentionImpl : torch::nn::Module {
    int64_t d_model, num_heads, d_k;
    torch::nn::Linear W_q{nullptr}, W_k{nullptr}, W_v{nullptr}, W_o{nullptr};

    MultiHeadAttentionImpl(int64_t d_model, int64_t num_heads)
        : d_model(d_model), num_heads(num_heads) {
        assert(d_model % num_heads == 0);
        d_k = d_model / num_heads;

        W_q = register_module("W_q", torch::nn::Linear(d_model, d_model));
        W_k = register_module("W_k", torch::nn::Linear(d_model, d_model));
        W_v = register_module("W_v", torch::nn::Linear(d_model, d_model));
        W_o = register_module("W_o", torch::nn::Linear(d_model, d_model));
    }

    // x: [batch_size, seq_len, d_model]
    // mask: [batch_size, seq_len, seq_len] (optional, for causal attention)
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {}) {
        int64_t batch_size = x.size(0);
        int64_t seq_len = x.size(1);

        torch::Tensor q = W_q(x); // [batch_size, seq_len, d_model]
        torch::Tensor k = W_k(x);
        torch::Tensor v = W_v(x);

        // Split into multiple heads
        // [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k]
        q = q.view({batch_size, seq_len, num_heads, d_k}).transpose(1, 2);
        k = k.view({batch_size, seq_len, num_heads, d_k}).transpose(1, 2);
        v = v.view({batch_size, seq_len, num_heads, d_k}).transpose(1, 2);

        // Scaled Dot-Product Attention
        // scores: [batch_size, num_heads, seq_len, seq_len]
        torch::Tensor scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(d_k));

        if (mask.defined()) {
            // mask should be broadcastable. For causal mask, it's typically [1, seq_len, seq_len] or [seq_len, seq_len]
            // We want mask to be 0 where attention is allowed, -inf where it's masked.
            // If mask is boolean (true where to mask), use masked_fill_
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9); // unsqueeze for num_heads if mask is [batch, sl, sl]
        }

        torch::Tensor attn_weights = torch::softmax(scores, /*dim=*/-1);
        // context: [batch_size, num_heads, seq_len, d_k]
        torch::Tensor context = torch::matmul(attn_weights, v);

        // Concatenate heads and project
        // context: [batch_size, seq_len, num_heads, d_k] -> [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view({batch_size, seq_len, d_model});

        return W_o(context);
    }
};
TORCH_MODULE(MultiHeadAttention);

// --- Feed Forward Network ---
struct FeedForwardImpl : torch::nn::Module {
    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
    torch::nn::Dropout dropout{nullptr};

    FeedForwardImpl(int64_t d_model, int64_t d_ff, double dropout_p = 0.1) {
        linear1 = register_module("linear1", torch::nn::Linear(d_model, d_ff));
        linear2 = register_module("linear2", torch::nn::Linear(d_ff, d_model));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(linear1(x));
        x = dropout(x);
        x = linear2(x);
        return x;
    }
};
TORCH_MODULE(FeedForward);

// --- Transformer Decoder Block ---
struct TransformerBlockImpl : torch::nn::Module {
    MultiHeadAttention self_attn{nullptr};
    FeedForward ff{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};

    TransformerBlockImpl(int64_t d_model, int64_t num_heads, int64_t d_ff, double dropout_p = 0.1) {
        self_attn = register_module("self_attn", MultiHeadAttention(d_model, num_heads));
        ff = register_module("ff", FeedForward(d_model, d_ff, dropout_p));
        norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        dropout1 = register_module("dropout1", torch::nn::Dropout(dropout_p));
        dropout2 = register_module("dropout2", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask) {
        // Self-attention sublayer
        torch::Tensor attn_output = self_attn(x, mask);
        x = norm1(x + dropout1(attn_output)); // Add & Norm

        // Feed-forward sublayer
        torch::Tensor ff_output = ff(x);
        x = norm2(x + dropout2(ff_output)); // Add & Norm
        return x;
    }
};
TORCH_MODULE(TransformerBlock);

// --- GPT-like Model ---
struct GPTLikeModelImpl : torch::nn::Module {
    torch::nn::Embedding token_embedding{nullptr};
    PositionalEncoding pos_encoding{nullptr};
    torch::nn::ModuleList transformer_blocks;
    torch::nn::Linear final_linear{nullptr};
    int64_t block_size_; // context window size

    GPTLikeModelImpl(int64_t vocab_size, int64_t d_model, int64_t num_heads,
                       int64_t num_layers, int64_t d_ff, int64_t block_size, double dropout_p = 0.1)
                       : block_size_(block_size) {
        token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, d_model));
        pos_encoding = register_module("pos_encoding", PositionalEncoding(d_model, block_size, dropout_p)); // max_len = block_size

        for (int i = 0; i < num_layers; ++i) {
            transformer_blocks->push_back(TransformerBlock(d_model, num_heads, d_ff, dropout_p));
        }
        register_module("transformer_blocks", transformer_blocks);

        final_linear = register_module("final_linear", torch::nn::Linear(d_model, vocab_size));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask) {
        // x: [batch_size, seq_len] (token IDs)
        x = token_embedding(x); // [batch_size, seq_len, d_model]
        x = pos_encoding(x);    // Add positional encodings

        for (auto& block : transformer_blocks) {
            x = block->as<TransformerBlock>()->forward(x, mask);
        }

        torch::Tensor logits = final_linear(x); // [batch_size, seq_len, vocab_size]
        return logits;
    }

    // Causal mask for autoregressive generation
    static torch::Tensor create_causal_mask(int64_t seq_len, torch::Device device) {
        torch::Tensor mask = torch::triu(torch::ones({seq_len, seq_len}, device), 1);
        return mask.to(torch::kBool); // True where to mask (upper triangle)
    }

    torch::Tensor generate(torch::Tensor idx, int64_t max_new_tokens, torch::Device device) {
        this->eval(); // Set to evaluation mode
        torch::NoGradGuard no_grad;

        for (int i = 0; i < max_new_tokens; ++i) {
            // Crop idx to the last block_size tokens if it's too long
            torch::Tensor idx_cond = idx;
            if (idx.size(1) > block_size_) {
                idx_cond = idx.index({torch::indexing::Slice(), torch::indexing::Slice(idx.size(1) - block_size_, torch::indexing::None)});
            }

            int64_t current_seq_len = idx_cond.size(1);
            torch::Tensor mask = create_causal_mask(current_seq_len, device);

            torch::Tensor logits = this->forward(idx_cond, mask); // [batch, current_seq_len, vocab_size]

            // Focus only on the last time step's logits
            logits = logits.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}); // [batch, vocab_size]

            // Apply softmax to get probabilities
            torch::Tensor probs = torch::softmax(logits, /*dim=*/-1);

            // Sample from the distribution (or take argmax for greedy)
            torch::Tensor idx_next = torch::multinomial(probs, 1); // [batch, 1]
            // torch::Tensor idx_next = torch::argmax(probs, /*dim=*/-1, /*keepdim=*/true); // Greedy

            // Append sampled token to the running sequence
            idx = torch::cat({idx, idx_next}, /*dim=*/1);
        }
        return idx;
    }
};
TORCH_MODULE(GPTLikeModel);

// --- Data Loader ---
std::tuple<torch::Tensor, torch::Tensor> get_batch(const std::vector<int>& data, int64_t block_size, int64_t batch_size, torch::Device device) {
    std::vector<torch::Tensor> x_list, y_list;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < batch_size; ++i) {
        std::uniform_int_distribution<> distrib(0, data.size() - block_size - 1);
        int64_t start_idx = distrib(gen);

        std::vector<int64_t> x_chunk(data.begin() + start_idx, data.begin() + start_idx + block_size);
        std::vector<int64_t> y_chunk(data.begin() + start_idx + 1, data.begin() + start_idx + block_size + 1);

        x_list.push_back(torch::tensor(x_chunk, torch::kLong));
        y_list.push_back(torch::tensor(y_chunk, torch::kLong));
    }
    return {torch::stack(x_list).to(device), torch::stack(y_list).to(device)};
}


int main() {
    torch::manual_seed(1337);

    // --- Hyperparameters ---
    std::string text_data = "Hello, LibTorch! This is a small test for a GPT-like model. "
                            "We will see if it can learn to generate some text. "
                            "The quick brown fox jumps over the lazy dog. "
                            "LibTorch is a C++ library for PyTorch. Training neural networks can be fun.";
    int64_t batch_size = 16;       // How many independent sequences to process in parallel?
    int64_t block_size = 32;       // What is the maximum context length for predictions?
    int64_t d_model = 64;          // Embedding dimension
    int64_t num_heads = 4;
    int64_t num_layers = 3;
    int64_t d_ff = 4 * d_model;    // Feed-forward hidden dimension
    double dropout_p = 0.1;
    int64_t max_iters = 5000;
    int64_t eval_interval = 200;
    double learning_rate = 1e-3;

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
    }

    // --- Tokenizer and Data ---
    CharTokenizer tokenizer(text_data);
    std::vector<int> data = tokenizer.encode(text_data);
    int64_t vocab_size = tokenizer.vocab_size;

    // --- Model ---
    GPTLikeModel model(vocab_size, d_model, num_heads, num_layers, d_ff, block_size, dropout_p);
    model->to(device);
    std::cout << "Model created with " << model->parameters().size() << " parameters." << std::endl;

    // --- Optimizer ---
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // --- Training Loop ---
    std::cout << "\nStarting training..." << std::endl;
    for (int iter = 0; iter < max_iters; ++iter) {
        model->train();
        auto [xb, yb] = get_batch(data, block_size, batch_size, device);
        // xb: [batch_size, block_size], yb: [batch_size, block_size]

        torch::Tensor mask = GPTLikeModelImpl::create_causal_mask(block_size, device);
        torch::Tensor logits = model->forward(xb, mask);
        // logits: [batch_size, block_size, vocab_size]

        // Reshape for cross_entropy_loss
        // Expected logits: [N, C], Expected target: [N]
        // N = batch_size * block_size, C = vocab_size
        logits = logits.view({-1, vocab_size});
        yb = yb.view({-1});

        torch::Tensor loss = torch::cross_entropy_loss(logits, yb);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (iter % eval_interval == 0 || iter == max_iters - 1) {
            std::cout << "Iter " << iter << " | Loss: " << loss.item<float>() << std::endl;

            // Generate some text
            model->eval();
            torch::NoGradGuard no_grad;
            torch::Tensor start_tokens = torch::tensor(tokenizer.encode("H"), torch::kLong).unsqueeze(0).to(device); // Start with 'H'
            torch::Tensor generated_tokens_tensor = model->generate(start_tokens, 50, device);
            std::vector<int> generated_tokens_vec;
            for(int i=0; i<generated_tokens_tensor.size(1); ++i) {
                generated_tokens_vec.push_back(generated_tokens_tensor[0][i].item<int>());
            }
            std::cout << "Generated: " << tokenizer.decode(generated_tokens_vec) << std::endl;
        }
    }

    std::cout << "\nTraining finished." << std::endl;

    // --- Final Generation ---
    std::cout << "\n--- Final Generation Example ---" << std::endl;
    model->eval();
    torch::NoGradGuard no_grad;
    torch::Tensor start_tokens = torch::tensor(tokenizer.encode("Lib"), torch::kLong).unsqueeze(0).to(device);
    torch::Tensor generated_tokens_tensor = model->generate(start_tokens, 100, device);
    std::vector<int> generated_tokens_vec;
    for(int i=0; i<generated_tokens_tensor.size(1); ++i) {
        generated_tokens_vec.push_back(generated_tokens_tensor[0][i].item<int>());
    }
    std::cout << "Prompt: Lib" << std::endl;
    std::cout << "Generated: " << tokenizer.decode(generated_tokens_vec) << std::endl;

    return 0;
}