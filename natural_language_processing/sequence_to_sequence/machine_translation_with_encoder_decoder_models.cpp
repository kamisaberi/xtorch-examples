#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm> // For std::transform, std::min, std::max
#include <set>       // For building vocab

// --- Configuration (Miniature Transformer) ---
const int64_t SRC_VOCAB_SIZE = 100;
const int64_t TGT_VOCAB_SIZE = 100;
const int64_t EMBED_DIM = 64;
const int64_t NUM_HEADS = 4;
const int64_t NUM_ENCODER_LAYERS = 2;
const int64_t NUM_DECODER_LAYERS = 2;
const int64_t MAX_SEQ_LEN = 32;       // Max sequence length for both src and tgt
const int64_t FFN_HIDDEN_DIM = EMBED_DIM * 4;
const double DROPOUT_PROB = 0.1;

const int64_t BATCH_SIZE = 8;
const int64_t NUM_EPOCHS = 100;
const double LEARNING_RATE = 1e-3; // Might need adjustment
const int64_t LOG_INTERVAL = 5;

// Special token indices (assuming they are managed by tokenizer)
const int64_t PAD_IDX = 0;
const int64_t SOS_IDX = 1; // Start of Sentence
const int64_t EOS_IDX = 2; // End of Sentence
const int64_t UNK_IDX = 3; // Unknown token


// --- Simplified Tokenizer (Character Level for Demo) ---
struct SimpleCharTokenizer {
    std::map<char, int64_t> char_to_idx;
    std::map<int64_t, char> idx_to_char;
    int64_t vocab_size_actual = 0;
    int64_t pad_idx_val = PAD_IDX;
    int64_t sos_idx_val = SOS_IDX;
    int64_t eos_idx_val = EOS_IDX;
    int64_t unk_idx_val = UNK_IDX;

    SimpleCharTokenizer(const std::vector<std::string>& texts, int64_t max_vocab_size_config,
                        bool is_target_tokenizer = false) {

        // Initialize special tokens first
        char_to_idx['\1'] = pad_idx_val; idx_to_char[pad_idx_val] = '\1'; // Placeholder for PAD
        vocab_size_actual++;
        if (is_target_tokenizer) { // SOS/EOS mostly for target
            char_to_idx['\2'] = sos_idx_val; idx_to_char[sos_idx_val] = '\2'; // Placeholder for SOS
            vocab_size_actual++;
            char_to_idx['\3'] = eos_idx_val; idx_to_char[eos_idx_val] = '\3'; // Placeholder for EOS
            vocab_size_actual++;
        }
        char_to_idx['\4'] = unk_idx_val; idx_to_char[unk_idx_val] = '\4'; // Placeholder for UNK
        vocab_size_actual++;


        std::set<char> unique_chars;
        for (const auto& text : texts) {
            for (char c : text) {
                unique_chars.insert(c);
            }
        }

        for (char c : unique_chars) {
            if (vocab_size_actual >= max_vocab_size_config) break;
            if (char_to_idx.find(c) == char_to_idx.end()) { // Avoid re-adding special placeholders
                char_to_idx[c] = vocab_size_actual;
                idx_to_char[vocab_size_actual] = c;
                vocab_size_actual++;
            }
        }
        std::cout << "Tokenizer: Actual vocab size: " << vocab_size_actual << std::endl;
    }

    std::vector<int64_t> encode(const std::string& text, int64_t max_len, bool add_sos_eos = false) const {
        std::vector<int64_t> encoded;
        if (add_sos_eos) {
            encoded.push_back(sos_idx_val);
        }
        for (size_t i = 0; i < text.length(); ++i) {
            if (encoded.size() >= (add_sos_eos ? max_len -1 : max_len)) break; // -1 for EOS if adding
            if (char_to_idx.count(text[i])) {
                encoded.push_back(char_to_idx.at(text[i]));
            } else {
                encoded.push_back(unk_idx_val);
            }
        }
        if (add_sos_eos) {
            if (encoded.size() < max_len) encoded.push_back(eos_idx_val);
            else encoded[max_len-1] = eos_idx_val; // Ensure EOS fits
        }
        // Padding
        while (encoded.size() < max_len) {
            encoded.push_back(pad_idx_val);
        }
        return encoded;
    }

    std::string decode(const std::vector<int64_t>& ids, bool is_target_decode = false) const {
        std::string decoded_text;
        for (int64_t id : ids) {
            if (id == pad_idx_val && !decoded_text.empty() && is_target_decode) continue;
            if (is_target_decode && (id == sos_idx_val)) continue;
            if (is_target_decode && (id == eos_idx_val)) break; // Stop at EOS for target

            if (idx_to_char.count(id)) {
                 char c = idx_to_char.at(id);
                 if (c >= '\1' && c <= '\4') { // Skip printing placeholder chars for special tokens
                    if (decoded_text.empty() && c == '\1') {} // Allow initial PAD for source if needed
                    else continue;
                 }
                 decoded_text += c;
            }
        }
        return decoded_text;
    }
};

// --- Transformer Components (Simplified from PyTorch's nn.Transformer) ---
// Note: PyTorch nn.TransformerEncoderLayer and nn.TransformerDecoderLayer are good references.
// This is a much more condensed version for brevity.

// Positional Encoding
struct PositionalEncodingImpl : torch::nn::Module {
    torch::Tensor pe; // Positional encoding tensor [1, max_len, embed_dim]

    PositionalEncodingImpl(int64_t embed_dim, int64_t max_len = MAX_SEQ_LEN, double dropout_p = DROPOUT_PROB) {
        pe = torch::zeros({max_len, embed_dim});
        torch::Tensor position = torch::arange(0, max_len, torch::kFloat).unsqueeze(1);
        torch::Tensor div_term = torch::exp(torch::arange(0, embed_dim, 2, torch::kFloat) * (-std::log(10000.0) / embed_dim));

        pe.index_put_({"...", torch::indexing::Slice(0, torch::indexing::None, 2)}, torch::sin(position * div_term));
        pe.index_put_({"...", torch::indexing::Slice(1, torch::indexing::None, 2)}, torch::cos(position * div_term));
        pe = pe.unsqueeze(0); // Add batch dimension: [1, max_len, embed_dim]
        register_buffer("pe_buffer", pe); // Register as buffer
    }

    torch::Tensor forward(torch::Tensor x) { // x: [B, T, E]
        // Add positional encoding to token embeddings
        // pe might be longer than T, so slice it.
        return x + pe.index({"...", torch::indexing::Slice(0, x.size(1)), torch::indexing::Slice()}).detach();
    }
};
TORCH_MODULE(PositionalEncoding);


// Transformer Encoder Layer
struct TransformerEncoderLayerImpl : torch::nn::Module {
    torch::nn::MultiheadAttention self_attn{nullptr};
    torch::nn::Linear ff1{nullptr}, ff2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr}, dropout_ff{nullptr};

    TransformerEncoderLayerImpl(int64_t embed_dim, int64_t num_heads, int64_t ffn_hidden_dim, double dropout_p) {
        self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads).dropout(dropout_p)));
        ff1 = register_module("ff1", torch::nn::Linear(embed_dim, ffn_hidden_dim));
        ff2 = register_module("ff2", torch::nn::Linear(ffn_hidden_dim, embed_dim));
        norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
        norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
        dropout1 = register_module("dropout1", torch::nn::Dropout(dropout_p));
        dropout2 = register_module("dropout2", torch::nn::Dropout(dropout_p));
        dropout_ff = register_module("dropout_ff", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(torch::Tensor src, const torch::optional<torch::Tensor>& src_mask = {}, const torch::optional<torch::Tensor>& src_key_padding_mask = {}) {
        // Self-attention block
        // MHA expects (Seq, Batch, Embed)
        torch::Tensor src2 = std::get<0>(self_attn(src.transpose(0,1), src.transpose(0,1), src.transpose(0,1), src_key_padding_mask, /*need_weights=*/false, src_mask));
        src = src + dropout1(src2.transpose(0,1)); // Back to (Batch, Seq, Embed)
        src = norm1(src);

        // Feed-forward block
        src2 = ff2(dropout_ff(torch::relu(ff1(src))));
        src = src + dropout2(src2);
        src = norm2(src);
        return src;
    }
};
TORCH_MODULE(TransformerEncoderLayer);


// Transformer Decoder Layer
struct TransformerDecoderLayerImpl : torch::nn::Module {
    torch::nn::MultiheadAttention self_attn{nullptr}, cross_attn{nullptr};
    torch::nn::Linear ff1{nullptr}, ff2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
    torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr}, dropout3{nullptr}, dropout_ff{nullptr};

    TransformerDecoderLayerImpl(int64_t embed_dim, int64_t num_heads, int64_t ffn_hidden_dim, double dropout_p) {
        self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads).dropout(dropout_p)));
        cross_attn = register_module("cross_attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(embed_dim, num_heads).dropout(dropout_p)));
        ff1 = register_module("ff1", torch::nn::Linear(embed_dim, ffn_hidden_dim));
        ff2 = register_module("ff2", torch::nn::Linear(ffn_hidden_dim, embed_dim));
        norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
        norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
        norm3 = register_module("norm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
        dropout1 = register_module("dropout1", torch::nn::Dropout(dropout_p));
        dropout2 = register_module("dropout2", torch::nn::Dropout(dropout_p));
        dropout3 = register_module("dropout3", torch::nn::Dropout(dropout_p));
        dropout_ff = register_module("dropout_ff", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(torch::Tensor tgt, const torch::Tensor& memory,
                          const torch::optional<torch::Tensor>& tgt_mask = {},
                          const torch::optional<torch::Tensor>& memory_mask = {},
                          const torch::optional<torch::Tensor>& tgt_key_padding_mask = {},
                          const torch::optional<torch::Tensor>& memory_key_padding_mask = {}) {
        // Masked Self-attention block for target
        // MHA expects (Seq, Batch, Embed)
        torch::Tensor tgt2 = std::get<0>(self_attn(tgt.transpose(0,1), tgt.transpose(0,1), tgt.transpose(0,1), tgt_key_padding_mask, /*need_weights=*/false, tgt_mask));
        tgt = tgt + dropout1(tgt2.transpose(0,1)); // Back to (Batch, Seq, Embed)
        tgt = norm1(tgt);

        // Cross-attention block (target attends to encoder memory)
        tgt2 = std::get<0>(cross_attn(tgt.transpose(0,1), memory.transpose(0,1), memory.transpose(0,1), memory_key_padding_mask, /*need_weights=*/false, memory_mask));
        tgt = tgt + dropout2(tgt2.transpose(0,1));
        tgt = norm2(tgt);

        // Feed-forward block
        tgt2 = ff2(dropout_ff(torch::relu(ff1(tgt))));
        tgt = tgt + dropout3(tgt2);
        tgt = norm3(tgt);
        return tgt;
    }
};
TORCH_MODULE(TransformerDecoderLayer);


// --- Full Transformer Model ---
struct TransformerModelImpl : torch::nn::Module {
    torch::nn::Embedding src_token_emb{nullptr}, tgt_token_emb{nullptr};
    PositionalEncoding pos_encoder{nullptr}; // Can share or have separate for src/tgt
    torch::nn::ModuleList encoder_layers{nullptr};
    torch::nn::ModuleList decoder_layers{nullptr};
    torch::nn::Linear final_linear{nullptr}; // Output layer
    int64_t embed_dim_;

    TransformerModelImpl(int64_t src_vocab_sz, int64_t tgt_vocab_sz, int64_t embed_dim,
                         int64_t n_heads, int64_t num_enc_layers, int64_t num_dec_layers,
                         int64_t ffn_hid_dim, double dropout_p) : embed_dim_(embed_dim) {

        src_token_emb = register_module("src_token_emb", torch::nn::Embedding(src_vocab_sz, embed_dim));
        tgt_token_emb = register_module("tgt_token_emb", torch::nn::Embedding(tgt_vocab_sz, embed_dim));
        pos_encoder = register_module("pos_encoder", PositionalEncoding(embed_dim, MAX_SEQ_LEN, dropout_p));

        for (int i = 0; i < num_enc_layers; ++i) {
            encoder_layers->push_back(TransformerEncoderLayer(embed_dim, n_heads, ffn_hid_dim, dropout_p));
        }
        for (int i = 0; i < num_dec_layers; ++i) {
            decoder_layers->push_back(TransformerDecoderLayer(embed_dim, n_heads, ffn_hid_dim, dropout_p));
        }
        final_linear = register_module("final_linear", torch::nn::Linear(embed_dim, tgt_vocab_sz));
    }

    // src: [B, SrcSeqLen], tgt: [B, TgtSeqLen]
    // src_padding_mask: [B, SrcSeqLen] (true for pad), tgt_padding_mask: [B, TgtSeqLen]
    // tgt_causal_mask: [TgtSeqLen, TgtSeqLen] (upper triangle mask)
    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt,
                          const torch::Tensor& src_padding_mask,
                          const torch::Tensor& tgt_padding_mask,
                          const torch::Tensor& tgt_causal_mask) {

        torch::Tensor src_emb = pos_encoder(src_token_emb(src) * std::sqrt(static_cast<double>(embed_dim_)));
        torch::Tensor tgt_emb = pos_encoder(tgt_token_emb(tgt) * std::sqrt(static_cast<double>(embed_dim_)));

        torch::Tensor memory = src_emb; // Initially, memory is the source embeddings
        for (auto& layer : *encoder_layers) {
            memory = layer->as<TransformerEncoderLayer>()->forward(memory, {}, src_padding_mask);
        }

        torch::Tensor dec_output = tgt_emb;
        for (auto& layer : *decoder_layers) {
            dec_output = layer->as<TransformerDecoderLayer>()->forward(
                dec_output, memory,
                tgt_causal_mask, {}, // memory_mask is usually not needed if src_padding_mask handled by encoder
                tgt_padding_mask, src_padding_mask);
        }
        return final_linear(dec_output); // [B, TgtSeqLen, TgtVocabSize]
    }

    // Helper to generate target causal mask
    static torch::Tensor generate_square_subsequent_mask(int64_t sz, torch::Device device) {
        torch::Tensor mask = (torch::triu(torch::ones({sz, sz}, device)) == 1).transpose(0, 1);
        mask = mask.to(torch::kFloat)
                   .masked_fill(mask == 0, -std::numeric_limits<float>::infinity())
                   .masked_fill(mask == 1, static_cast<float>(0.0));
        return mask; // Shape [sz, sz]
    }
};
TORCH_MODULE(TransformerModel);


// --- Dummy Dataset (Character Level Parallel Sentences) ---
struct ParallelTextDataset : torch::data::datasets::Dataset<ParallelTextDataset> {
    std::vector<std::pair<std::string, std::string>> sentence_pairs;
    const SimpleCharTokenizer& src_tokenizer;
    const SimpleCharTokenizer& tgt_tokenizer;
    int64_t max_len;

    ParallelTextDataset(const std::vector<std::pair<std::string, std::string>>& pairs,
                        const SimpleCharTokenizer& src_tok, const SimpleCharTokenizer& tgt_tok, int64_t max_seq_len)
        : sentence_pairs(pairs), src_tokenizer(src_tok), tgt_tokenizer(tgt_tok), max_len(max_seq_len) {}

    // Output: { (src_ids, tgt_ids_input), tgt_ids_output }
    // tgt_ids_input: <sos> w1 w2 ... <eos> <pad> ...
    // tgt_ids_output: w1 w2 ... <eos> <pad> ... <pad>
    torch::data::Example<std::pair<torch::Tensor, torch::Tensor>, torch::Tensor> get(size_t index) override {
        const auto& pair = sentence_pairs[index];
        std::vector<int64_t> src_encoded = src_tokenizer.encode(pair.first, max_len, false);

        // For target, input to decoder will have SOS, output for loss will not have SOS but will have EOS
        std::vector<int64_t> tgt_encoded_full = tgt_tokenizer.encode(pair.second, max_len, true); // Adds SOS, EOS

        std::vector<int64_t> tgt_input_vec(tgt_encoded_full.begin(), tgt_encoded_full.end() -1); // Remove last token (could be EOS or PAD)
                                                                                               // to make it input for decoder
        std::vector<int64_t> tgt_output_vec(tgt_encoded_full.begin() + 1, tgt_encoded_full.end()); // Shifted by one for next token prediction

        // Ensure fixed length for stacking (though encode should handle this)
        while(tgt_input_vec.size() < max_len) tgt_input_vec.push_back(tgt_tokenizer.pad_idx_val);
        if(tgt_input_vec.size() > max_len) tgt_input_vec.resize(max_len);

        while(tgt_output_vec.size() < max_len) tgt_output_vec.push_back(tgt_tokenizer.pad_idx_val);
        if(tgt_output_vec.size() > max_len) tgt_output_vec.resize(max_len);


        torch::Tensor src_tensor = torch::tensor(src_encoded, torch::kLong);
        torch::Tensor tgt_input_tensor = torch::tensor(tgt_input_vec, torch::kLong);
        torch::Tensor tgt_output_tensor = torch::tensor(tgt_output_vec, torch::kLong);

        return {{src_tensor, tgt_input_tensor}, tgt_output_tensor};
    }
    torch::optional<size_t> size() const override { return sentence_pairs.size(); }
};

// Custom collate
struct CustomMTCollate {
    torch::data::Example<std::pair<torch::Tensor, torch::Tensor>, torch::Tensor> operator()(
        std::vector<torch::data::Example<std::pair<torch::Tensor, torch::Tensor>, torch::Tensor>> batch_samples) {

        std::vector<torch::Tensor> batch_src_ids, batch_tgt_input_ids, batch_tgt_output_ids;
        for (const auto& sample : batch_samples) {
            batch_src_ids.push_back(sample.data.first);
            batch_tgt_input_ids.push_back(sample.data.second);
            batch_tgt_output_ids.push_back(sample.target);
        }
        return {
            {torch::stack(batch_src_ids), torch::stack(batch_tgt_input_ids)},
            torch::stack(batch_tgt_output_ids)
        };
    }
};


int main() {
    std::cout << "Machine Translation with Transformer (Conceptual - LibTorch C++)" << std::endl;
    torch::manual_seed(0);
    std::cout << std::fixed << std::setprecision(4);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    // --- Dummy Data & Tokenizers ---
    std::vector<std::pair<std::string, std::string>> dummy_parallel_corpus = {
        {"hello world", "hallo welt"},
        {"this is a test", "das ist ein test"},
        {"good morning", "guten morgen"},
        {"how are you", "wie geht es dir"},
        {"thank you", "danke schon"},
        {"example sentence", "beispielsatz hier"}
    };
    std::vector<std::string> src_texts, tgt_texts;
    for(const auto& p : dummy_parallel_corpus) {
        src_texts.push_back(p.first);
        tgt_texts.push_back(p.second);
    }
    SimpleCharTokenizer src_tokenizer(src_texts, SRC_VOCAB_SIZE);
    SimpleCharTokenizer tgt_tokenizer(tgt_texts, TGT_VOCAB_SIZE, true /*is_target*/);

    // --- Model ---
    TransformerModel model(src_tokenizer.vocab_size_actual, tgt_tokenizer.vocab_size_actual,
                           EMBED_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                           FFN_HIDDEN_DIM, DROPOUT_PROB);
    model->to(device);
    std::cout << "Transformer model created. Parameters: "
              << std::accumulate(model->parameters().begin(), model->parameters().end(), 0L,
                                 [](long sum, const torch::Tensor& t){ return sum + t.numel(); })
              << std::endl;

    // --- DataLoader ---
    auto train_dataset = ParallelTextDataset(dummy_parallel_corpus, src_tokenizer, tgt_tokenizer, MAX_SEQ_LEN)
                            .map(CustomMTCollate());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE));
    std::cout << "DataLoader created." << std::endl;

    // --- Optimizer & Loss ---
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    torch::nn::CrossEntropyLoss criterion(torch::nn::CrossEntropyLossOptions().ignore_index(PAD_IDX));
    std::cout << "Optimizer and Loss created." << std::endl;

    // --- Training Loop ---
    std::cout << "\nStarting Training..." << std::endl;
    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        model->train();
        double epoch_loss = 0.0;
        size_t batch_idx_count = 0;

        for (auto& batch : *train_loader) {
            optimizer.zero_grad();

            torch::Tensor src_seq = batch.data.first.to(device);    // [B, SrcSeqLen]
            torch::Tensor tgt_seq_input = batch.data.second.to(device); // [B, TgtSeqLen] (decoder input)
            torch::Tensor tgt_seq_output = batch.target.to(device); // [B, TgtSeqLen] (expected output for loss)

            // Create masks
            // Source padding mask: true where src_seq is PAD_IDX
            torch::Tensor src_padding_mask = (src_seq == PAD_IDX); // [B, SrcSeqLen]
            // Target padding mask: true where tgt_seq_input is PAD_IDX
            torch::Tensor tgt_padding_mask = (tgt_seq_input == PAD_IDX); // [B, TgtSeqLen]
            // Target causal mask (for self-attention in decoder)
            torch::Tensor tgt_causal_mask = TransformerModelImpl::generate_square_subsequent_mask(tgt_seq_input.size(1), device); // [TgtSeqLen, TgtSeqLen]

            torch::Tensor logits = model->forward(src_seq, tgt_seq_input, src_padding_mask, tgt_padding_mask, tgt_causal_mask);
            // logits: [B, TgtSeqLen, TgtVocabSize]
            // tgt_seq_output: [B, TgtSeqLen]

            torch::Tensor loss = criterion(
                logits.reshape({-1, logits.size(-1)}), // [B * TgtSeqLen, TgtVocabSize]
                tgt_seq_output.reshape(-1)              // [B * TgtSeqLen]
            );

            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
            batch_idx_count++;
            if (batch_idx_count % LOG_INTERVAL == 0) {
                 std::cout << "Epoch: " << epoch << "/" << NUM_EPOCHS
                          << " | Batch: " << batch_idx_count
                          << " | Loss: " << loss.item<double>() << std::endl;
            }
        }
         if (batch_idx_count > 0) {
            std::cout << "-------------------------------------------------------" << std::endl;
            std::cout << "Epoch: " << epoch << " Average Loss: " << (epoch_loss / batch_idx_count) << std::endl;
            std::cout << "-------------------------------------------------------" << std::endl;
        }


        // --- Simple Greedy Decoding Example ---
        if (epoch % (NUM_EPOCHS / 5) == 0 || epoch == NUM_EPOCHS) {
            model->eval();
            torch::NoGradGuard no_grad;
            std::string test_src_sentence = dummy_parallel_corpus[0].first; // "hello world"
            std::cout << "\nTranslating at epoch " << epoch << ": '" << test_src_sentence << "'" << std::endl;

            std::vector<int64_t> src_encoded = src_tokenizer.encode(test_src_sentence, MAX_SEQ_LEN, false);
            torch::Tensor src_tensor = torch::tensor(src_encoded, torch::kLong).unsqueeze(0).to(device); // [1, SrcSeqLen]
            torch::Tensor src_pad_mask_infer = (src_tensor == PAD_IDX);

            // Start decoding with SOS token
            std::vector<int64_t> tgt_decoded_ids = {tgt_tokenizer.sos_idx_val};

            for (int i = 0; i < MAX_SEQ_LEN -1; ++i) { // -1 because SOS is already there
                torch::Tensor tgt_input_tensor_infer = torch::tensor(tgt_decoded_ids, torch::kLong).unsqueeze(0).to(device); // [1, CurrentTgtLen]
                torch::Tensor tgt_pad_mask_infer = (tgt_input_tensor_infer == PAD_IDX);
                torch::Tensor tgt_causal_mask_infer = TransformerModelImpl::generate_square_subsequent_mask(tgt_input_tensor_infer.size(1), device);

                torch::Tensor output_logits = model->forward(src_tensor, tgt_input_tensor_infer, src_pad_mask_infer, tgt_pad_mask_infer, tgt_causal_mask_infer);
                // Get logits for the last token only: output_logits [1, CurrentTgtLen, TgtVocabSize]
                torch::Tensor last_token_logits = output_logits.slice(/*dim=*/1, /*start=*/-1).squeeze(1); // [1, TgtVocabSize]
                int64_t predicted_token_id = torch::argmax(last_token_logits, /*dim=*/1).item<int64_t>();

                tgt_decoded_ids.push_back(predicted_token_id);
                if (predicted_token_id == tgt_tokenizer.eos_idx_val) {
                    break;
                }
            }
            std::cout << "Translation: '" << tgt_tokenizer.decode(tgt_decoded_ids, true) << "'" << std::endl;
        }
    }
    std::cout << "Training finished." << std::endl;
    // torch::save(model, "transformer_mt_model.pt");
    return 0;
}