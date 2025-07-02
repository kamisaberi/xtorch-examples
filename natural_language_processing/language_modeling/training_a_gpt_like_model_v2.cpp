#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm> // For std::transform, std::min
#include <sstream>   // For basic tokenization

// --- Configuration (Miniature GPT) ---
const int64_t VOCAB_SIZE = 100;       // Small vocabulary for demo
const int64_t EMBED_DIM = 64;         // Embedding dimension
const int64_t NUM_HEADS = 4;          // Number of attention heads
const int64_t NUM_LAYERS = 2;         // Number of Transformer decoder layers
const int64_t BLOCK_SIZE = 32;        // Max sequence length (context window)
const int64_t FFN_HIDDEN_DIM = EMBED_DIM * 4; // Hidden dim for FFN in Transformer block
const double DROPOUT_PROB = 0.1;

const int64_t BATCH_SIZE = 8;
const int64_t NUM_EPOCHS = 50;        // Needs many more for real tasks
const double LEARNING_RATE = 3e-4;
const int64_t LOG_INTERVAL = 5;

// --- Simplified Tokenizer (Character Level for Demo) ---
struct SimpleCharTokenizer {
    std::map<char, int64_t> char_to_idx;
    std::map<int64_t, char> idx_to_char;
    int64_t vocab_size_actual = 0;
    int64_t pad_idx = 0; // Assuming PAD is the first token

    SimpleCharTokenizer(const std::vector<std::string>& texts, int64_t max_vocab_size) {
        std::set<char> unique_chars;
        for (const auto& text : texts) {
            for (char c : text) {
                unique_chars.insert(c);
            }
        }

        // Add PAD token first
        char_to_idx['\0'] = pad_idx; // Using null char for PAD conceptually
        idx_to_char[pad_idx] = '\0';
        vocab_size_actual = 1;

        for (char c : unique_chars) {
            if (vocab_size_actual >= max_vocab_size) break;
            if (char_to_idx.find(c) == char_to_idx.end()) {
                char_to_idx[c] = vocab_size_actual;
                idx_to_char[vocab_size_actual] = c;
                vocab_size_actual++;
            }
        }
        std::cout << "Tokenizer: Actual vocab size: " << vocab_size_actual << std::endl;
    }

    std::vector<int64_t> encode(const std::string& text, int64_t max_len) const {
        std::vector<int64_t> encoded;
        for (size_t i = 0; i < text.length() && encoded.size() < max_len; ++i) {
            if (char_to_idx.count(text[i])) {
                encoded.push_back(char_to_idx.at(text[i]));
            } else {
                // Handle unknown characters if necessary (e.g., map to a special UNK token)
                // For this simple char tokenizer, we assume all chars in training data are known
            }
        }
        // Padding
        while (encoded.size() < max_len) {
            encoded.push_back(pad_idx);
        }
        return encoded;
    }

    std::string decode(const std::vector<int64_t>& ids) const {
        std::string decoded_text;
        for (int64_t id : ids) {
            if (id == pad_idx && !decoded_text.empty()) continue; // Skip further PADs after some content
            if (idx_to_char.count(id)) {
                 if (idx_to_char.at(id) == '\0' && !decoded_text.empty()) continue; // Skip PAD char
                 decoded_text += idx_to_char.at(id);
            }
        }
        return decoded_text;
    }
};


// --- GPT Components ---
// Masked Multi-Head Self-Attention
struct CausalSelfAttentionImpl : torch::nn::Module {
    torch::nn::Linear c_attn{nullptr}; // Combined query, key, value projections
    torch::nn::Linear c_proj{nullptr}; // Output projection
    torch::nn::Dropout resid_dropout{nullptr};
    torch::nn::Dropout attn_dropout{nullptr};
    int64_t n_head;
    int64_t n_embed;

    CausalSelfAttentionImpl(int64_t embed_dim, int64_t num_heads, double dropout_p)
        : n_embed(embed_dim), n_head(num_heads) {
        assert(embed_dim % num_heads == 0);
        c_attn = register_module("c_attn", torch::nn::Linear(embed_dim, embed_dim * 3));
        c_proj = register_module("c_proj", torch::nn::Linear(embed_dim, embed_dim));
        attn_dropout = register_module("attn_dropout", torch::nn::Dropout(dropout_p));
        resid_dropout = register_module("resid_dropout", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(torch::Tensor x) {
        int64_t B = x.size(0); // Batch size
        int64_t T = x.size(1); // Sequence length
        int64_t E = x.size(2); // Embedding dimension (n_embed)

        // Calculate query, key, values for all heads in batch
        torch::Tensor qkv = c_attn(x); // [B, T, 3*E]
        auto qkv_chunks = qkv.chunk(3, /*dim=*/2);
        torch::Tensor q = qkv_chunks[0]; // [B, T, E]
        torch::Tensor k = qkv_chunks[1]; // [B, T, E]
        torch::Tensor v = qkv_chunks[2]; // [B, T, E]

        // Reshape and transpose for multi-head attention
        // q, k, v: [B, n_head, T, head_size] where head_size = E / n_head
        q = q.view({B, T, n_head, E / n_head}).transpose(1, 2);
        k = k.view({B, T, n_head, E / n_head}).transpose(1, 2);
        v = v.view({B, T, n_head, E / n_head}).transpose(1, 2);

        // Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        torch::Tensor att = torch::matmul(q, k.transpose(-2, -1)) * (1.0 / std::sqrt(static_cast<float>(k.size(-1))));

        // Causal mask (upper triangular part)
        torch::Tensor mask = torch::triu(torch::ones({T, T}, x.options()), /*diagonal=*/1).to(torch::kBool);
        att = att.masked_fill(mask, -std::numeric_limits<float>::infinity());

        att = torch::softmax(att, /*dim=*/-1);
        att = attn_dropout(att);

        torch::Tensor y = torch::matmul(att, v); // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view({B, T, E}); // Re-assemble all head outputs side by side

        // Output projection
        y = resid_dropout(c_proj(y));
        return y;
    }
};
TORCH_MODULE(CausalSelfAttention);

// Transformer Decoder Block
struct GPTBlockImpl : torch::nn::Module {
    CausalSelfAttention attn{nullptr};
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    torch::nn::Sequential mlp{nullptr};

    GPTBlockImpl(int64_t embed_dim, int64_t num_heads, int64_t ffn_hidden_dim, double dropout_p) {
        ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
        attn = register_module("attn", CausalSelfAttention(embed_dim, num_heads, dropout_p));
        ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
        mlp = register_module("mlp", torch::nn::Sequential(
            torch::nn::Linear(embed_dim, ffn_hidden_dim),
            torch::nn::GELU(), // Or ReLU
            torch::nn::Linear(ffn_hidden_dim, embed_dim),
            torch::nn::Dropout(dropout_p)
        ));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x + attn(ln1(x)); // Residual connection after attention
        x = x + mlp(ln2(x));  // Residual connection after MLP
        return x;
    }
};
TORCH_MODULE(GPTBlock);

// --- Simplified GPT Model ---
struct SimpleGPTImpl : torch::nn::Module {
    torch::nn::Embedding token_embeddings{nullptr};
    torch::nn::Embedding position_embeddings{nullptr}; // Learned positional embeddings
    torch::nn::Sequential blocks{nullptr};
    torch::nn::LayerNorm ln_f{nullptr};    // Final LayerNorm
    torch::nn::Linear lm_head{nullptr}; // Output head to predict next token

    int64_t block_size;

    SimpleGPTImpl(int64_t vocab_sz, int64_t embed_dim, int64_t blk_size,
                  int64_t num_layers, int64_t num_heads, int64_t ffn_hidden_dim, double dropout_p)
        : block_size(blk_size) {

        token_embeddings = register_module("token_embeddings", torch::nn::Embedding(vocab_sz, embed_dim));
        position_embeddings = register_module("position_embeddings", torch::nn::Embedding(blk_size, embed_dim));

        blocks = torch::nn::Sequential();
        for (int i = 0; i < num_layers; ++i) {
            blocks->push_back(GPTBlock(embed_dim, num_heads, ffn_hidden_dim, dropout_p));
        }
        register_module("blocks", blocks);

        ln_f = register_module("ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
        lm_head = register_module("lm_head", torch::nn::Linear(embed_dim, vocab_sz, /*bias=*/false)); // GPT-2 often no bias on lm_head
    }

    torch::Tensor forward(torch::Tensor idx) { // idx: [B, T] tensor of token indices
        int64_t B = idx.size(0);
        int64_t T = idx.size(1);
        assert(T <= block_size && "Cannot forward sequence longer than block_size");

        // Token embeddings
        torch::Tensor tok_emb = token_embeddings(idx); // [B, T, E]

        // Positional embeddings
        torch::Tensor pos = torch::arange(0, T, torch::kLong, idx.device()).unsqueeze(0); // [1, T]
        torch::Tensor pos_emb = position_embeddings(pos); // [1, T, E]

        torch::Tensor x = tok_emb + pos_emb; // [B, T, E]
        x = blocks->forward(x);
        x = ln_f(x);
        torch::Tensor logits = lm_head(x); // [B, T, VocabSize]
        return logits;
    }

    // For generation
    torch::Tensor generate(torch::Tensor idx_start, int64_t max_new_tokens, const SimpleCharTokenizer& tokenizer) {
        this->eval(); // Set to eval mode
        torch::NoGradGuard no_grad;

        torch::Tensor idx = idx_start; // [B, current_T]
        for (int i = 0; i < max_new_tokens; ++i) {
            // Crop idx to the last block_size tokens if it grows too long
            torch::Tensor idx_cond = idx.size(1) <= block_size ? idx : idx.slice(/*dim=*/1, /*start=*/idx.size(1) - block_size);

            torch::Tensor logits = this->forward(idx_cond); // [B, T_cond, VocabSize]
            // Focus only on the logits for the last time step
            logits = logits.slice(/*dim=*/1, /*start=*/-1, /*end=*/torch::indexing::None); // [B, 1, VocabSize]

            // Apply softmax to get probabilities
            torch::Tensor probs = torch::softmax(logits, /*dim=*/-1); // [B, 1, VocabSize]

            // Greedy sampling: pick the token with the highest probability
            torch::Tensor idx_next = torch::argmax(probs, /*dim=*/-1); // [B, 1]

            // Append sampled index to the running sequence
            idx = torch::cat({idx, idx_next}, /*dim=*/1); // [B, current_T + 1]

            // If all sequences in batch ended with PAD or some EOS token, could break early
            // For char level, this is less common unless '\n' is an EOS.
        }
        return idx;
    }
};
TORCH_MODULE(SimpleGPT);


// --- Dummy Dataset (Character Level) ---
struct TextDataset : torch::data::datasets::Dataset<TextDataset> {
    std::vector<std::vector<int64_t>> encoded_texts;
    int64_t block_size;

    TextDataset(const std::vector<std::string>& texts, const SimpleCharTokenizer& tokenizer, int64_t blk_size)
        : block_size(blk_size) {
        for (const auto& text : texts) {
            // Encode the whole text, then we'll chunk it
            std::vector<int64_t> full_encoded = tokenizer.encode(text, text.length() + 1); // +1 just in case
            // Remove padding added by tokenizer.encode if we are chunking
            full_encoded.erase(std::remove(full_encoded.begin(), full_encoded.end(), tokenizer.pad_idx), full_encoded.end());

            if (full_encoded.size() > block_size) { // Only use texts longer than block_size for simplicity
                encoded_texts.push_back(full_encoded);
            }
        }
        std::cout << "Dataset: Usable texts (longer than block_size): " << encoded_texts.size() << std::endl;
    }

    // Returns {input_sequence, target_sequence}
    // input:  idx[0...T-1]
    // target: idx[1...T] (next token prediction)
    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override {
        // For a real dataset, 'index' would map to a specific chunk of text.
        // Here, we'll pick a random text and a random chunk from it.
        std::mt19937 rng(std::random_device{}()); // New RNG for each get for randomness
        if (encoded_texts.empty()) { // Should not happen if constructor checks
             return {torch::zeros({block_size}, torch::kLong), torch::zeros({block_size}, torch::kLong)};
        }
        size_t text_idx = std::uniform_int_distribution<size_t>(0, encoded_texts.size() - 1)(rng);
        const auto& chosen_text_encoded = encoded_texts[text_idx];

        size_t start_pos = 0;
        if (chosen_text_encoded.size() > block_size + 1) { // +1 because we need target
             start_pos = std::uniform_int_distribution<size_t>(0, chosen_text_encoded.size() - (block_size + 1))(rng);
        } else { // Should not happen if filtered in constructor
            std::cerr << "Warning: Text too short in get(), this shouldn't happen." << std::endl;
             return {torch::zeros({block_size}, torch::kLong), torch::zeros({block_size}, torch::kLong)};
        }

        std::vector<int64_t> input_chunk_vec, target_chunk_vec;
        input_chunk_vec.reserve(block_size);
        target_chunk_vec.reserve(block_size);

        for (size_t i = 0; i < block_size; ++i) {
            input_chunk_vec.push_back(chosen_text_encoded[start_pos + i]);
            target_chunk_vec.push_back(chosen_text_encoded[start_pos + i + 1]);
        }

        return {torch::tensor(input_chunk_vec, torch::kLong), torch::tensor(target_chunk_vec, torch::kLong)};
    }

    torch::optional<size_t> size() const override {
        // This is tricky for chunked text.
        // A common approach is to set a large number of "virtual" samples per epoch.
        // Or calculate total number of possible chunks.
        // For this demo, let's just make it a fixed large number or based on num texts * avg_chunks.
        if (encoded_texts.empty()) return 0;
        return encoded_texts.size() * 5; // Arbitrary: 5 chunks per text on average
    }
};


int main() {
    std::cout << "Training a GPT-like Model (Conceptual - LibTorch C++)" << std::endl;
    torch::manual_seed(1337); // Common seed from Karpathy's nanoGPT
    std::cout << std::fixed << std::setprecision(4);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    // --- Dummy Data & Tokenizer ---
    std::vector<std::string> dummy_texts = {
        "this is the first sentence for training our toy gpt model.",
        "another sentence is here to provide more data examples.",
        "language models learn patterns from text sequences.",
        "the quick brown fox jumps over the lazy dog.",
        "a stitch in time saves nine examples for later use.",
        "make hay while the sun shines brightly in the sky today."
        // Add much more and varied text for any real learning
    };
    SimpleCharTokenizer tokenizer(dummy_texts, VOCAB_SIZE);

    // --- Model ---
    SimpleGPT model(tokenizer.vocab_size_actual, EMBED_DIM, BLOCK_SIZE,
                    NUM_LAYERS, NUM_HEADS, FFN_HIDDEN_DIM, DROPOUT_PROB);
    model->to(device);
    std::cout << "SimpleGPT model created. Parameters: "
              << std::accumulate(model->parameters().begin(), model->parameters().end(), 0L,
                                 [](long sum, const torch::Tensor& t){ return sum + t.numel(); })
              << std::endl;

    // --- DataLoader ---
    // Make sure dataset_size is large enough for BATCH_SIZE if using RandomSampler.
    // The DummyTextDataset size() is a bit arbitrary.
    auto train_dataset = TextDataset(dummy_texts, tokenizer, BLOCK_SIZE)
                            .map(torch::data::transforms::Stack<>()); // Stacks samples into batch

    // Check if dataset is effectively empty
    if (!train_dataset.size().has_value() || train_dataset.size().value() == 0) {
        std::cerr << "Dataset is empty or too small. Ensure texts are longer than block_size (" << BLOCK_SIZE << ")" << std::endl;
        return 1;
    }
     if (train_dataset.size().value() < BATCH_SIZE) {
        std::cerr << "Warning: Dataset size (" << train_dataset.size().value()
                  << ") is smaller than batch size (" << BATCH_SIZE
                  << "). This might cause issues with DataLoader or training." << std::endl;
        // Consider reducing batch size or adding more data. For now, proceed.
    }


    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(0) // workers > 0 can be problematic on Windows
    );
    std::cout << "DataLoader created." << std::endl;

    // --- Optimizer & Loss ---
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(LEARNING_RATE));
    // CrossEntropyLoss expects logits [B, C, d1, d2...] and targets [B, d1, d2...]
    // Our logits are [B, T, VocabSize], targets are [B, T]
    // We need to reshape logits to [B*T, VocabSize] and targets to [B*T]
    torch::nn::CrossEntropyLoss criterion(torch::nn::CrossEntropyLossOptions().ignore_index(tokenizer.pad_idx));
    std::cout << "Optimizer and Loss created." << std::endl;

    // --- Training Loop ---
    std::cout << "\nStarting Training..." << std::endl;
    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        model->train();
        double epoch_loss = 0.0;
        size_t batch_idx_count = 0;

        try {
            for (auto& batch : *train_loader) {
                optimizer.zero_grad();

                torch::Tensor inputs = batch.data.to(device);    // [B, BLOCK_SIZE]
                torch::Tensor targets = batch.target.to(device); // [B, BLOCK_SIZE]

                torch::Tensor logits = model->forward(inputs); // [B, BLOCK_SIZE, VocabSize]

                // Reshape for CrossEntropyLoss
                // logits: [B*BLOCK_SIZE, VocabSize]
                // targets: [B*BLOCK_SIZE]
                torch::Tensor loss = criterion(
                    logits.view({-1, logits.size(-1)}),
                    targets.view(-1)
                );

                loss.backward();
                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0); // Gradient clipping
                optimizer.step();

                epoch_loss += loss.item<double>();
                batch_idx_count++;

                if (batch_idx_count % LOG_INTERVAL == 0) {
                    std::cout << "Epoch: " << epoch << "/" << NUM_EPOCHS
                            << " | Batch: " << batch_idx_count //<< "/" << (train_dataset.size().value() / BATCH_SIZE)
                            << " | Loss: " << loss.item<double>() << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception in training loop: " << e.what() << std::endl;
            // This can happen if dataset size is too small for batching with RandomSampler
            // or other dataloader issues.
            if (batch_idx_count == 0) { // No batches processed, likely a dataloader issue
                 std::cerr << "No batches processed, check dataset size and dataloader configuration." << std::endl;
                 break; // Exit epoch loop
            }
        }

        if (batch_idx_count > 0) {
            std::cout << "-------------------------------------------------------" << std::endl;
            std::cout << "Epoch: " << epoch << " Average Loss: " << (epoch_loss / batch_idx_count) << std::endl;
            std::cout << "-------------------------------------------------------" << std::endl;
        } else {
             std::cout << "-------------------------------------------------------" << std::endl;
             std::cout << "Epoch: " << epoch << " No batches processed." << std::endl;
             std::cout << "-------------------------------------------------------" << std::endl;
             // Potentially break training if no data is being processed
             if (epoch > 1) { // Allow first epoch to try, then break
                 std::cerr << "Stopping training due to no data processed in multiple epochs." << std::endl;
                 break;
             }
        }


        // Generate some text periodically
        if (epoch % (NUM_EPOCHS / 5) == 0 || epoch == NUM_EPOCHS) {
            std::cout << "\nGenerating text at epoch " << epoch << ":" << std::endl;
            std::string start_prompt_str = "th"; // Example starting characters
            std::vector<int64_t> start_ids_vec = tokenizer.encode(start_prompt_str, start_prompt_str.length());
            start_ids_vec.erase(std::remove(start_ids_vec.begin(), start_ids_vec.end(), tokenizer.pad_idx), start_ids_vec.end()); // Remove any PADs

            if (start_ids_vec.empty() && !start_prompt_str.empty()) { // If all chars were unknown
                std::cout << "Warning: Start prompt characters are not in vocab. Using PAD as start." << std::endl;
                start_ids_vec.push_back(tokenizer.pad_idx);
            }
             if (start_ids_vec.empty() && start_prompt_str.empty()){
                 std::cout << "Warning: Start prompt is empty. Using PAD as start." << std::endl;
                start_ids_vec.push_back(tokenizer.pad_idx);
            }


            torch::Tensor start_idx_tensor = torch::tensor(start_ids_vec, torch::kLong).unsqueeze(0).to(device); // [1, T_start]

            torch::Tensor generated_sequence = model->generate(start_idx_tensor, 50, tokenizer); // Generate 50 new tokens
            std::cout << "\"" << tokenizer.decode(generated_sequence.squeeze(0).cpu().to(torch::kLong).accessor<int64_t,1>().data(),
                                                generated_sequence.size(1))
                      << "\"" << std::endl; // Basic way to get vector from tensor
        }
    }
    std::cout << "Training finished." << std::endl;

    // torch::save(model, "simple_gpt_model.pt");
    return 0;
}