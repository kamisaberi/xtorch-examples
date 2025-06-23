#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <set>

// --- Configuration (Miniature Model) ---
const int64_t VOCAB_SIZE = 100;       // Combined vocabulary size
const int64_t EMBED_DIM = 64;
const int64_t HIDDEN_DIM = 128;       // LSTM hidden dimension
const int64_t NUM_LSTM_LAYERS = 1;    // For encoder and decoder LSTMs
const int64_t MAX_SRC_LEN = 50;
const int64_t MAX_TGT_LEN = 20;
const double DROPOUT_PROB = 0.1;

const int64_t BATCH_SIZE = 4;
const int64_t NUM_EPOCHS = 50;
const double LEARNING_RATE = 1e-3;
const int64_t LOG_INTERVAL = 5;

// Special token indices
const int64_t PAD_IDX = 0;
const int64_t SOS_IDX = 1;
const int64_t EOS_IDX = 2;
const int64_t UNK_IDX = 3;
const int64_t MAX_OOV_WORDS = 10; // Max OOV words per source document we can point to

// --- Simplified Tokenizer (Character Level) ---
// For a real Pointer-Generator, you'd have a subword tokenizer (BPE/SentencePiece)
// and a mechanism to map OOV source words to temporary vocabulary IDs.
struct SimpleCharTokenizer {
    std::map<char, int64_t> char_to_idx;
    std::map<int64_t, char> idx_to_char;
    int64_t vocab_size_fixed = 0; // Size of fixed vocabulary (excluding OOV slots)
    int64_t pad_idx_val = PAD_IDX;
    int64_t sos_idx_val = SOS_IDX;
    int64_t eos_idx_val = EOS_IDX;
    int64_t unk_idx_val = UNK_IDX;

    SimpleCharTokenizer(const std::vector<std::string>& texts, int64_t max_fixed_vocab_config) {
        char_to_idx['\1'] = pad_idx_val; idx_to_char[pad_idx_val] = '\1'; vocab_size_fixed++;
        char_to_idx['\2'] = sos_idx_val; idx_to_char[sos_idx_val] = '\2'; vocab_size_fixed++;
        char_to_idx['\3'] = eos_idx_val; idx_to_char[eos_idx_val] = '\3'; vocab_size_fixed++;
        char_to_idx['\4'] = unk_idx_val; idx_to_char[unk_idx_val] = '\4'; vocab_size_fixed++;

        std::set<char> unique_chars;
        for (const auto& text : texts) { for (char c : text) unique_chars.insert(c); }

        for (char c : unique_chars) {
            if (vocab_size_fixed >= max_fixed_vocab_config) break;
            if (char_to_idx.find(c) == char_to_idx.end()) {
                char_to_idx[c] = vocab_size_fixed;
                idx_to_char[vocab_size_fixed] = c;
                vocab_size_fixed++;
            }
        }
        std::cout << "Tokenizer: Fixed vocab size: " << vocab_size_fixed << std::endl;
    }

    // Encodes text. For Pointer-Gen, we also need to know OOV words in source.
    // This simplified version doesn't fully handle dynamic OOV mapping for copying.
    std::vector<int64_t> encode(const std::string& text, int64_t max_len, bool add_sos_eos = false,
                                std::map<std::string, int64_t>* src_oov_map = nullptr, // Only for source
                                std::vector<int64_t>* src_extended_ids = nullptr // Only for source
                               ) const {
        std::vector<int64_t> encoded_ids;
        if (add_sos_eos) encoded_ids.push_back(sos_idx_val);

        // Simple word-level split for conceptual OOV handling
        std::stringstream ss(text);
        std::string word;
        std::vector<std::string> words;
        while(ss >> word) words.push_back(word);


        for (const std::string& current_word_str : words) { // Iterate over "words" (chars in this simple case)
            // In a real system, current_word_str would be an actual word.
            // Here, each char is treated as a word for OOV demonstration.
            if (encoded_ids.size() >= (add_sos_eos ? max_len -1 : max_len)) break;

            // Simplified: treat each character as a potential "word" for OOV mapping
            // This is a placeholder for real word-level OOV handling.
            char c = current_word_str.empty() ? ' ' : current_word_str[0]; // Take first char if non-empty

            if (char_to_idx.count(c)) {
                encoded_ids.push_back(char_to_idx.at(c));
                if (src_extended_ids) src_extended_ids->push_back(char_to_idx.at(c));
            } else { // OOV
                encoded_ids.push_back(unk_idx_val); // Put UNK in standard IDs
                if (src_oov_map && src_extended_ids) {
                    // This is for source text only: map OOV word to a temporary ID
                    if (src_oov_map->find(current_word_str) == src_oov_map->end()) {
                        if (src_oov_map->size() < MAX_OOV_WORDS) {
                             // Assign new OOV ID (fixed_vocab_size + current_oov_count)
                            (*src_oov_map)[current_word_str] = vocab_size_fixed + src_oov_map->size();
                        }
                    }
                    if (src_oov_map->count(current_word_str)) {
                        src_extended_ids->push_back((*src_oov_map)[current_word_str]);
                    } else {
                        src_extended_ids->push_back(unk_idx_val); // Cannot store more OOVs
                    }
                } else if (src_extended_ids) { // If target, just use UNK
                    src_extended_ids->push_back(unk_idx_val);
                }
            }
        }

        if (add_sos_eos) {
            if (encoded_ids.size() < max_len) encoded_ids.push_back(eos_idx_val);
            else if (!encoded_ids.empty()) encoded_ids.back() = eos_idx_val;
            if (src_extended_ids) {
                 if (src_extended_ids->size() < max_len) src_extended_ids->push_back(eos_idx_val);
                 else if(!src_extended_ids->empty()) src_extended_ids->back() = eos_idx_val;
            }
        }
        // Padding
        while (encoded_ids.size() < max_len) encoded_ids.push_back(pad_idx_val);
        if (src_extended_ids) {
            while (src_extended_ids->size() < max_len) src_extended_ids->push_back(pad_idx_val);
        }
        return encoded_ids;
    }

    std::string decode(const std::vector<int64_t>& ids,
                       const std::map<int64_t, std::string>& oov_idx_to_str_map = {}) const { // For decoding copied OOV words
        std::string text;
        for (int64_t id : ids) {
            if (id == pad_idx_val && !text.empty()) continue;
            if (id == sos_idx_val) continue;
            if (id == eos_idx_val) break;

            if (id < vocab_size_fixed) { // In fixed vocab
                if (idx_to_char.count(id)) {
                    char c = idx_to_char.at(id);
                    if (c >= '\1' && c <= '\4') continue; // Skip special placeholders
                    text += c;
                }
            } else { // OOV word (copied)
                if (oov_idx_to_str_map.count(id)) {
                    text += "[" + oov_idx_to_str_map.at(id) + "]"; // Indicate copied OOV
                } else {
                    text += "[UNK_COPIED]";
                }
            }
        }
        return text;
    }
};


// --- Pointer-Generator Components (Conceptual LSTM-based) ---
struct EncoderImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTM lstm{nullptr};

    EncoderImpl(int64_t vocab_sz, int64_t embed_d, int64_t hidden_d, int64_t num_layers, double dropout_p) {
        embedding = register_module("embedding", torch::nn::Embedding(vocab_sz, embed_d));
        // Bidirectional LSTM: output hidden_dim will be 2*hidden_d if concatenated
        // For simplicity, let's use unidirectional or ensure decoder handles 2*hidden_d input context
        lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(embed_d, hidden_d)
                                                        .num_layers(num_layers)
                                                        .bidirectional(true) // Common for encoders
                                                        .dropout(dropout_p)
                                                        .batch_first(true)));
    }
    // src: [B, SrcLen]
    // Returns: encoder_outputs [B, SrcLen, 2*HiddenD], hidden_state [2*NumLayers, B, HiddenD]
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor src) {
        torch::Tensor embedded = embedding(src); // [B, SrcLen, EmbedD]
        auto lstm_out_tuple = lstm(embedded);
        torch::Tensor outputs = std::get<0>(lstm_out_tuple); // [B, SrcLen, 2*HiddenD]
        torch::Tensor hidden = std::get<1>(lstm_out_tuple).get().idx({0});  // h_n [2*NumLayers, B, HiddenD]
        // Could also return cell state if needed by decoder init
        return {outputs, hidden};
    }
};
TORCH_MODULE(Encoder);

// Simplified Attention (Luong-style dot product for concept)
struct AttentionImpl : torch::nn::Module {
    torch::nn::Linear W_h{nullptr}; // For transforming decoder hidden state

    AttentionImpl(int64_t hidden_dim) { // Assuming encoder_out_dim and decoder_hidden_dim are compatible
        // W_h = register_module("W_h", torch::nn::Linear(hidden_dim, hidden_dim)); // If using general attention
    }

    // decoder_hidden: [B, HiddenD] (current decoder hidden state)
    // encoder_outputs: [B, SrcLen, EncOutDim] (e.g., 2*HiddenD if bidirectional encoder)
    // Returns: context_vector [B, EncOutDim], attention_weights [B, SrcLen]
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor decoder_hidden, torch::Tensor encoder_outputs) {
        // Simple dot product attention (assuming decoder_hidden can be broadcasted or reshaped)
        // For real Luong: score(h_t, h_s) = h_t^T W_a h_s
        // Here, simplified: score = sum(decoder_hidden * encoder_output_s) element-wise, then softmax
        // This requires decoder_hidden and encoder_outputs to have compatible last dim.
        // Let's assume decoder_hidden is [B, EncOutDim] after some projection
        // For simplicity, let's assume last dim of decoder_hidden matches EncOutDim.
        // decoder_hidden: [B, 1, EncOutDim] for broadcasting with encoder_outputs
        torch::Tensor scores = torch::bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2); // [B, SrcLen]

        torch::Tensor attn_weights = torch::softmax(scores, /*dim=*/1); // [B, SrcLen]
        // context = sum(attn_weights_s * encoder_output_s)
        // attn_weights.unsqueeze(1): [B, 1, SrcLen]
        // encoder_outputs:           [B, SrcLen, EncOutDim]
        torch::Tensor context = torch::bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1); // [B, EncOutDim]
        return {context, attn_weights};
    }
};
TORCH_MODULE(Attention);


struct DecoderImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTMCell lstm_cell{nullptr}; // Use LSTMCell for step-by-step decoding
    Attention attention{nullptr};
    torch::nn::Linear W_c{nullptr};       // Combines context and hidden state
    torch::nn::Linear V{nullptr};         // Projects to vocab size (generation part)
    torch::nn::Linear W_pgen{nullptr};    // For calculating p_gen

    // Decoder hidden_dim should match encoder's (or be configurable)
    // EncOutDim is the dim of each encoder_outputs timestep (e.g., 2*EncoderHidden if BiLSTM)
    DecoderImpl(int64_t vocab_sz, int64_t embed_d, int64_t hidden_d, int64_t enc_out_dim, double dropout_p) {
        embedding = register_module("embedding", torch::nn::Embedding(vocab_sz, embed_d));
        // LSTMCell input: embed_d + enc_out_dim (concatenated input embedding and context vector)
        lstm_cell = register_module("lstm_cell", torch::nn::LSTMCell(embed_d + enc_out_dim, hidden_d));
        attention = register_module("attention", Attention(hidden_d)); // Attention uses decoder hidden state

        // W_c: Takes [context_vector, decoder_hidden_state, decoder_input_embedding] -> some output dim
        // For p_gen calculation. Input size needs to be defined based on these components.
        // Simplified: p_gen input from decoder_hidden, context, current_input_emb
        int64_t pgen_input_dim = hidden_d + enc_out_dim + embed_d;
        W_pgen = register_module("W_pgen", torch::nn::Linear(pgen_input_dim, 1));

        // V: Projects [decoder_hidden + context_vector] to vocab_size
        // This part is for the "generation" probability from fixed vocabulary.
        // Simplified: project combined state to vocab.
        // The original paper has a more nuanced way to combine states before vocab projection.
        W_c = register_module("W_c", torch::nn::Linear(hidden_d + enc_out_dim, hidden_d)); // To combine h_dec and context
        V = register_module("V", torch::nn::Linear(hidden_d, vocab_sz)); // From combined state to vocab
    }

    // One step of decoding
    // prev_token_idx: [B], previous token generated (or SOS for first step)
    // prev_hidden_state: ([B, HiddenD], [B, HiddenD]) (h,c for LSTMCell)
    // encoder_outputs: [B, SrcLen, EncOutDim]
    // src_padding_mask: [B, SrcLen] (true for pad tokens) - for masking attention
    // Returns: final_dist [B, ExtendedVocabSize], new_hidden, attn_weights [B, SrcLen]
    std::tuple<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>, torch::Tensor>
    forward_step(torch::Tensor prev_token_idx,
                 std::pair<torch::Tensor, torch::Tensor> prev_hidden_state,
                 torch::Tensor encoder_outputs,
                 const torch::Tensor& src_padding_mask, // For masking attention
                 const torch::Tensor& src_ids_extended, // [B, SrcLen] with OOV IDs for scattering copy probs
                 int64_t extended_vocab_size) {

        torch::Tensor embedded = embedding(prev_token_idx); // [B, EmbedD]
        torch::Tensor dec_h_prev = prev_hidden_state.first;
        torch::Tensor dec_c_prev = prev_hidden_state.second;

        // Attention: use previous decoder hidden state dec_h_prev to attend over encoder_outputs
        // dec_h_prev might need projection if its dim != encoder_outputs last dim for dot product attention.
        // For simplicity, assume attention module handles this or dimensions match.
        // The Attention module here expects decoder_hidden of dim that matches encoder_outputs' feature dim after projection.
        // A common setup is to project dec_h_prev to match encoder_outputs dim or vice-versa.
        // Let's assume our Attention takes dec_h_prev [B, HiddenD] and encoder_outputs [B, SrcLen, EncOutDim]
        auto [context_vector, attn_weights] = attention(dec_h_prev, encoder_outputs); // attn_weights: [B, SrcLen]
        // Apply padding mask to attention weights BEFORE softmax (done inside attention if it handles masks)
        // If not, it should be: attn_weights.masked_fill_(src_padding_mask, -1e9); attn_weights = softmax(attn_weights);

        // LSTM input: concatenate current embedding and context vector
        torch::Tensor lstm_input = torch::cat({embedded, context_vector}, /*dim=*/1); // [B, EmbedD + EncOutDim]
        auto [dec_h_next, dec_c_next] = lstm_cell(lstm_input, prev_hidden_state); // [B, HiddenD] each

        // Calculate p_gen: probability of generating from vocab vs copying
        // Input to W_pgen: [dec_h_next, context_vector, embedded_input]
        torch::Tensor pgen_input = torch::cat({dec_h_next, context_vector, embedded}, 1);
        torch::Tensor p_gen = torch::sigmoid(W_pgen(pgen_input)); // [B, 1]

        // Calculate vocabulary distribution (generation)
        // Simplified: use combined state of dec_h_next and context_vector
        torch::Tensor combined_state_for_vocab = torch::relu(W_c(torch::cat({dec_h_next, context_vector}, 1)));
        torch::Tensor vocab_dist = torch::softmax(V(combined_state_for_vocab), /*dim=*/-1); // [B, FixedVocabSize]

        // Final distribution: p_gen * P_vocab + (1 - p_gen) * P_copy
        // P_copy is attn_weights scattered to positions of source words.
        torch::Tensor p_vocab_weighted = p_gen * vocab_dist; // [B, FixedVocabSize]
        torch::Tensor p_copy_weighted_factor = (1.0 - p_gen); // [B, 1]

        // Create extended vocab distribution (zeros up to extended_vocab_size)
        torch::Tensor final_dist = torch::zeros({prev_token_idx.size(0), extended_vocab_size}, V->weight.options());
        final_dist.index_add_(/*dim=*/1, src_ids_extended, attn_weights * p_copy_weighted_factor); // Scatter copy probs
                                                                                              // src_ids_extended: [B, SrcLen]
                                                                                              // attn_weights * factor: [B, SrcLen]
                                                                                              // index_add_ needs careful shape alignment for batching.
                                                                                              // PyTorch index_add_ sums values at repeated indices.
                                                                                              // This needs careful batch-wise scattering.

        // A more robust way for batch-wise scatter-add: iterate batch or use advanced indexing
        // For simplicity, let's assume a loop for clarity (inefficient for GPU):
        for (int b = 0; b < prev_token_idx.size(0); ++b) {
            final_dist[b].index_add_(/*dim=*/0, src_ids_extended[b], (attn_weights[b] * p_copy_weighted_factor[b]));
        }
        // Add generation probabilities to the fixed vocab part
        final_dist.slice(/*dim=*/1, /*start=*/0, /*end=*/vocab_dist.size(1)) += p_vocab_weighted;

        // Small epsilon to prevent log(0) if this output goes to NLLLoss directly
        final_dist = final_dist + 1e-12;

        return {final_dist, {dec_h_next, dec_c_next}, attn_weights};
    }

    // Helper to initialize decoder hidden state from encoder's final hidden state
    // enc_hidden: [2*NumEncLayers, B, EncHiddenD] if bidirectional
    // We need to adapt this to [NumDecLayers, B, DecHiddenD]
    std::pair<torch::Tensor, torch::Tensor> init_hidden(torch::Tensor enc_hidden_h) const {
         // Simplistic: take last layer forward and backward, sum or concat, then project if needed
         // enc_hidden_h is [2 (bi) * N_enc_layer, B, H_enc]
         // For uni-LSTM decoder init, we might take the last layer of encoder
         torch::Tensor h = enc_hidden_h.slice(0, -2, -1, 2).sum(0); // Sum forward and backward of last enc layer
                                                                // This makes it [B, H_enc]
         // If dec_hidden_dim != enc_hidden_dim, a projection layer would be needed.
         // Assume they are same for now.
         // For LSTMCell, hidden is (h,c). We only have h from encoder. Init c to zeros.
         return {h, torch::zeros_like(h)};
    }
};
TORCH_MODULE(Decoder);


// --- Full Pointer-Generator Network ---
struct PointerGeneratorNetworkImpl : torch::nn::Module {
    Encoder encoder{nullptr};
    Decoder decoder{nullptr};
    int64_t fixed_vocab_size; // For knowing where OOV IDs start

    PointerGeneratorNetworkImpl(int64_t vocab_sz, int64_t embed_d, int64_t hidden_d,
                                int64_t enc_lstm_layers, int64_t dec_lstm_layers, double dropout_p)
        : fixed_vocab_size(vocab_sz) { // Store fixed vocab size

        // Encoder output dim will be 2*hidden_d if bidirectional
        int64_t encoder_output_dim = hidden_d * (NUM_LSTM_LAYERS > 0 && true ? 2:1) ; // 2 if bidirectional

        encoder = register_module("encoder", Encoder(vocab_sz, embed_d, hidden_d, enc_lstm_layers, dropout_p));
        decoder = register_module("decoder", Decoder(vocab_sz, embed_d, hidden_d, encoder_output_dim, dropout_p));
    }

    // Training forward pass (uses teacher forcing)
    // src_seq: [B, SrcLen], src_ids_extended: [B, SrcLen] (with OOV IDs)
    // tgt_seq_input: [B, TgtLen] (decoder input, starts with SOS)
    // src_padding_mask: [B, SrcLen]
    // Returns: all_step_outputs [B, TgtLen, ExtendedVocabSize]
    torch::Tensor forward(torch::Tensor src_seq, const torch::Tensor& src_ids_extended,
                          torch::Tensor tgt_seq_input, const torch::Tensor& src_padding_mask,
                          int64_t extended_vocab_size_batch // Max OOV ID + fixed_vocab_size for this batch
                         ) {
        auto [encoder_outputs, encoder_hidden_h] = encoder(src_seq);
        // encoder_hidden_h is [2*NumLayers, B, HiddenD]
        // encoder_outputs is [B, SrcLen, 2*HiddenD]

        std::pair<torch::Tensor, torch::Tensor> decoder_hidden = decoder->init_hidden(encoder_hidden_h);

        int64_t batch_size = src_seq.size(0);
        int64_t target_len = tgt_seq_input.size(1);
        std::vector<torch::Tensor> all_step_outputs;

        // Teacher forcing: input previous ground truth token at each step
        for (int64_t t = 0; t < target_len; ++t) {
            torch::Tensor current_tgt_token = tgt_seq_input.select(/*dim=*/1, /*index=*/t); // [B]

            auto [output_dist, new_hidden, attn_weights] = decoder->forward_step(
                current_tgt_token, decoder_hidden, encoder_outputs, src_padding_mask,
                src_ids_extended, extended_vocab_size_batch
            );
            all_step_outputs.push_back(output_dist);
            decoder_hidden = new_hidden;
            // Could use attn_weights for coverage if implemented
        }
        return torch::stack(all_step_outputs, /*dim=*/1); // [B, TgtLen, ExtendedVocabSize]
    }
};
TORCH_MODULE(PointerGeneratorNetwork);

// --- Dummy Dataset ---
// For Pointer-Gen, dataset needs to provide:
// src_ids, src_ids_extended_with_oov, src_oov_map, tgt_ids_input, tgt_ids_output
struct SummDataset : torch::data::datasets::Dataset<SummDataset> {
    std::vector<std::pair<std::string, std::string>> data_pairs;
    const SimpleCharTokenizer& tokenizer; // Assuming shared tokenizer for src/tgt
    int64_t max_src, max_tgt;

    SummDataset(const std::vector<std::pair<std::string, std::string>>& pairs,
                const SimpleCharTokenizer& tok, int64_t max_s, int64_t max_t)
        : data_pairs(pairs), tokenizer(tok), max_src(max_s), max_tgt(max_t) {}

    // Output: { (src_ids, src_ext_ids, tgt_in_ids, src_oov_map_for_decode), tgt_out_ids_maybe_extended }
    // src_oov_map_for_decode is std::map<int64_t, std::string> (oov_temp_id -> oov_string)
    // For loss, tgt_out_ids_maybe_extended should use OOV_IDs if the target word was OOV in source.
    // This part is very complex to align perfectly with the final distribution.
    // The loss should be calculated against a target tensor that has indices up to extended_vocab_size.
    struct ExampleOutput {
        torch::Tensor src_ids, src_ids_extended, tgt_ids_input, tgt_ids_output_extended;
        std::map<std::string, int64_t> src_oov_word_to_id_map; // word_str -> temp_oov_id
        int64_t num_oov_in_src = 0;
    };


    ExampleOutput get_raw(size_t index) { // Changed return type to custom struct
        const auto& pair = data_pairs[index];
        std::map<std::string, int64_t> src_oov_map_for_this_item;
        std::vector<int64_t> src_ext_ids_vec;

        std::vector<int64_t> src_ids_vec = tokenizer.encode(pair.first, max_src, false, &src_oov_map_for_this_item, &src_ext_ids_vec);

        std::vector<int64_t> tgt_ids_full = tokenizer.encode(pair.second, max_tgt, true); // has SOS, EOS

        std::vector<int64_t> tgt_input_vec(tgt_ids_full.begin(), tgt_ids_full.end() - 1); // For decoder input
        std::vector<int64_t> tgt_output_raw_vec(tgt_ids_full.begin() + 1, tgt_ids_full.end()); // For loss target

        // For tgt_output_extended, map OOV words if they appear in src_oov_map
        // This is simplified: assumes target words are single chars from the source OOVs
        std::vector<int64_t> tgt_output_extended_vec;
        tgt_output_extended_vec.reserve(tgt_output_raw_vec.size());
        std::stringstream ss_tgt(pair.second); std::string tgt_word_str;
        std::vector<std::string> tgt_words; while(ss_tgt >> tgt_word_str) tgt_words.push_back(tgt_word_str);

        size_t tgt_word_idx = 0;
        for(size_t i=0; i<tgt_output_raw_vec.size(); ++i){ // Iterate based on char-tokenized length
            if(tgt_word_idx < tgt_words.size()){
                const std::string& current_tgt_word_for_oov = tgt_words[tgt_word_idx]; // Conceptual word
                char current_tgt_char = pair.second[i]; // Actual char being processed

                if (src_oov_map_for_this_item.count(std::string(1, current_tgt_char))) { // If this char was an OOV "word" in source
                    tgt_output_extended_vec.push_back(src_oov_map_for_this_item.at(std::string(1, current_tgt_char)));
                } else if (tokenizer.char_to_idx.count(current_tgt_char)) {
                    tgt_output_extended_vec.push_back(tokenizer.char_to_idx.at(current_tgt_char));
                } else {
                    tgt_output_extended_vec.push_back(tokenizer.unk_idx_val);
                }
                // Advance conceptual word index if space or end of word (very rough)
                if (i > 0 && pair.second[i-1] == ' ') tgt_word_idx++;
            } else { // Ran out of conceptual words, just use fixed vocab
                 char current_tgt_char = pair.second[i];
                 if (tokenizer.char_to_idx.count(current_tgt_char)) {
                    tgt_output_extended_vec.push_back(tokenizer.char_to_idx.at(current_tgt_char));
                } else {
                    tgt_output_extended_vec.push_back(tokenizer.unk_idx_val);
                }
            }
             if (i == pair.second.length()-1 && tgt_word_idx < tgt_words.size()-1) tgt_word_idx++; // Last char
        }
        while(tgt_output_extended_vec.size() < max_tgt) tgt_output_extended_vec.push_back(PAD_IDX);


        return {
            torch::tensor(src_ids_vec, torch::kLong),
            torch::tensor(src_ext_ids_vec, torch::kLong),
            torch::tensor(tgt_input_vec, torch::kLong),
            torch::tensor(tgt_output_extended_vec, torch::kLong),
            src_oov_map_for_this_item,
            static_cast<int64_t>(src_oov_map_for_this_item.size())
        };
    }
    // This get() is problematic for direct use with DataLoader if returning complex struct.
    // Usually, a custom collate function handles this complexity.
    // For simplicity, we'll try to return tensors directly for the Example.
    torch::data::Example<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>, torch::Tensor> get(size_t index) override {
        ExampleOutput raw = get_raw(index);
        // src_oov_map and num_oov_in_src are lost here, need to be handled by collate or passed differently
        // For NLLLoss, target must be LongTensor.
        // We'll pass num_oov_in_src via an extra tensor in input tuple. Hacky.
        torch::Tensor num_oov_tensor = torch::tensor({raw.num_oov_in_src}, torch::kLong);
        return { {raw.src_ids, raw.src_ids_extended, raw.tgt_ids_input, num_oov_tensor}, raw.tgt_ids_output_extended};
    }

    torch::optional<size_t> size() const override { return data_pairs.size(); }
};

// Custom Collate for Pointer-Gen
struct PointerGenCollate {
    // This needs to return what the training loop expects for a batch
    // Batch Output: { (stacked_src, stacked_src_ext, stacked_tgt_in, max_oov_count_tensor), stacked_tgt_out_ext }
    auto operator()(std::vector<torch::data::Example<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>, torch::Tensor>> batch_samples) {
        std::vector<torch::Tensor> batch_src, batch_src_ext, batch_tgt_in, batch_tgt_out_ext;
        std::vector<torch::Tensor> batch_num_oov_counts; // Stores the num_oov_in_src for each sample

        for(const auto& sample : batch_samples) {
            batch_src.push_back(std::get<0>(sample.data));
            batch_src_ext.push_back(std::get<1>(sample.data));
            batch_tgt_in.push_back(std::get<2>(sample.data));
            batch_num_oov_counts.push_back(std::get<3>(sample.data)); // The hacky num_oov tensor
            batch_tgt_out_ext.push_back(sample.target);
        }
        // The "max_oov_count_in_batch" is needed to determine the final dimension of model output distribution.
        // This is one of the trickiest parts. The model's final linear layer might only go up to fixed_vocab_size.
        // The pointer mechanism effectively extends this.
        // For loss calculation, we need to know the true extended vocab size for *this batch*.
        int64_t max_oov_count_in_batch = 0;
        for(const auto& count_tensor : batch_num_oov_counts) {
            max_oov_count_in_batch = std::max(max_oov_count_in_batch, count_tensor.item<int64_t>());
        }
        torch::Tensor max_oov_tensor = torch::tensor({max_oov_count_in_batch}, torch::kLong);


        return std::make_tuple(
            torch::stack(batch_src),
            torch::stack(batch_src_ext),
            torch::stack(batch_tgt_in),
            max_oov_tensor, // Pass max OOV count in batch
            torch::stack(batch_tgt_out_ext)
        );
    }
};


int main() {
    std::cout << "Pointer-Generator Network for Summarization (Conceptual - LibTorch C++)" << std::endl;
    torch::manual_seed(0);
    std::cout << std::fixed << std::setprecision(4);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    std::vector<std::pair<std::string, std::string>> dummy_corpus = {
        {"this is a long document about cats and dogs living together in harmony it is very nice", "cats and dogs friends"},
        {"another example document which needs summarization for a test case", "example summary test"},
        {"the quick brown fox jumps over the lazy dog near the river bank on a sunny day", "fox jumps over dog"},
        {"a b c d e f g h i j k l m n o p q r s t u v w x y z this is alphabet", "alphabet a to z"}
    };
    std::vector<std::string> all_texts;
    for(const auto& p : dummy_corpus) { all_texts.push_back(p.first); all_texts.push_back(p.second); }
    SimpleCharTokenizer tokenizer(all_texts, VOCAB_SIZE);

    PointerGeneratorNetwork model(tokenizer.vocab_size_fixed, EMBED_DIM, HIDDEN_DIM,
                                  NUM_LSTM_LAYERS, NUM_LSTM_LAYERS, DROPOUT_PROB);
    model->to(device);
    std::cout << "Pointer-Generator model created. Params: "
              << std::accumulate(model->parameters().begin(), model->parameters().end(), 0L,
                                 [](long sum, const torch::Tensor& t){ return sum + t.numel(); })
              << std::endl;

    auto train_dataset = SummDataset(dummy_corpus, tokenizer, MAX_SRC_LEN, MAX_TGT_LEN);
    // Use custom collate. The DataLoader type needs to match what collate returns.
    // The default DataLoader map(Stack<>()) won't work with the tuple from PointerGenCollate.
    // This part is tricky. Let's simplify the DataLoader for this conceptual example.
    // We will manually iterate and collate for now. A full DataLoader setup with custom collate is more involved.
    std::cout << "Dataset created." << std::endl;

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    // NLLLoss for when probabilities are already computed (after log_softmax)
    // Or CrossEntropyLoss if model outputs logits. Our final_dist is probs, so need NLLLoss on log(final_dist)
    // The target indices can be > fixed_vocab_size, so CrossEntropy won't work directly if its C param is fixed_vocab_size.
    // We need a custom way to compute NLL loss with the extended vocabulary.
    auto custom_nll_loss = [&](const torch::Tensor& pred_dist_log, const torch::Tensor& target_extended) {
        // pred_dist_log: [B, TgtLen, ExtendedVocabSize], log probabilities
        // target_extended: [B, TgtLen], indices up to ExtendedVocabSize
        // We need to gather the log_probs for the target indices.
        torch::Tensor gathered_log_probs = pred_dist_log.gather(
            /*dim=*/2, target_extended.unsqueeze(2) // [B, TgtLen, 1] for gather
        ).squeeze(2); // [B, TgtLen]

        // Mask out PAD tokens from loss
        torch::Tensor non_pad_mask = (target_extended != PAD_IDX);
        torch::Tensor masked_loss = -gathered_log_probs.masked_select(non_pad_mask);
        return masked_loss.mean(); // Or sum / num_non_pad_tokens
    };
    std::cout << "Optimizer and custom NLL loss created." << std::endl;

    PointerGenCollate collator; // Instantiate our custom collator

    std::cout << "\nStarting Training..." << std::endl;
    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        model->train();
        double epoch_loss = 0.0;
        size_t batch_idx_count = 0;

        // Manual batching and collation for simplicity
        std::vector<size_t> indices(train_dataset.size().value());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

        for (size_t i = 0; i < indices.size(); i += BATCH_SIZE) {
            optimizer.zero_grad();

            std::vector<torch::data::Example<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>, torch::Tensor>> current_batch_samples;
            for (size_t j = i; j < std::min(i + BATCH_SIZE, indices.size()); ++j) {
                current_batch_samples.push_back(train_dataset.get(indices[j]));
            }
            if (current_batch_samples.empty()) continue;

            auto collated_batch = collator(current_batch_samples);
            torch::Tensor src_seq = std::get<0>(collated_batch).to(device);
            torch::Tensor src_ext_seq = std::get<1>(collated_batch).to(device);
            torch::Tensor tgt_in_seq = std::get<2>(collated_batch).to(device);
            int64_t max_oov_in_batch = std::get<3>(collated_batch).item<int64_t>();
            torch::Tensor tgt_out_ext_seq = std::get<4>(collated_batch).to(device);

            int64_t current_extended_vocab_size = tokenizer.vocab_size_fixed + max_oov_in_batch;
            torch::Tensor src_padding_mask = (src_seq == PAD_IDX);

            torch::Tensor final_prob_dists = model->forward(src_seq, src_ext_seq, tgt_in_seq, src_padding_mask, current_extended_vocab_size);
            // final_prob_dists: [B, TgtLen, current_extended_vocab_size]

            torch::Tensor loss = custom_nll_loss(torch::log(final_prob_dists), tgt_out_ext_seq);

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
        if (epoch % (NUM_EPOCHS / 2) == 0 || epoch == NUM_EPOCHS) {
            model->eval();
            torch::NoGradGuard no_grad;

            // Get a sample for inference (first one)
            auto raw_infer_sample = train_dataset.get_raw(0);
            torch::Tensor src_infer = raw_infer_sample.src_ids.unsqueeze(0).to(device);
            torch::Tensor src_ext_infer = raw_infer_sample.src_ids_extended.unsqueeze(0).to(device);
            std::map<std::string, int64_t> src_oov_word_to_id_map_infer = raw_infer_sample.src_oov_word_to_id_map;
            int64_t num_oov_infer = raw_infer_sample.num_oov_in_src;
            int64_t infer_extended_vocab_size = tokenizer.vocab_size_fixed + num_oov_infer;

            std::map<int64_t, std::string> oov_id_to_word_map_infer;
            for(const auto& pair : src_oov_word_to_id_map_infer) {
                oov_id_to_word_map_infer[pair.second] = pair.first;
            }

            std::cout << "\nSummarizing at epoch " << epoch << ": '" << dummy_corpus[0].first << "'" << std::endl;

            auto [encoder_outputs_infer, encoder_hidden_infer] = model->encoder(src_infer);
            std::pair<torch::Tensor, torch::Tensor> decoder_hidden_infer = model->decoder->init_hidden(encoder_hidden_infer);

            std::vector<int64_t> decoded_ids_infer = {SOS_IDX};
            torch::Tensor src_pad_mask_infer = (src_infer == PAD_IDX);

            for (int i = 0; i < MAX_TGT_LEN -1; ++i) {
                torch::Tensor current_tgt_token_infer = torch::tensor({decoded_ids_infer.back()}, torch::kLong).unsqueeze(0).to(device); // [1,1]

                auto [output_dist_infer, new_hidden_infer, attn_weights_infer] = model->decoder->forward_step(
                    current_tgt_token_infer.squeeze(0), // Pass [1] for current token
                    decoder_hidden_infer,
                    encoder_outputs_infer,
                    src_pad_mask_infer,
                    src_ext_infer,
                    infer_extended_vocab_size
                );
                decoder_hidden_infer = new_hidden_infer;
                int64_t predicted_token_id = torch::argmax(output_dist_infer.squeeze(0), /*dim=*/0).item<int64_t>();
                decoded_ids_infer.push_back(predicted_token_id);
                if (predicted_token_id == EOS_IDX) break;
            }
            std::cout << "Summary: '" << tokenizer.decode(decoded_ids_infer, oov_id_to_word_map_infer) << "'" << std::endl;
        }
    }
    std::cout << "Training finished." << std::endl;
    // torch::save(model, "pointer_gen_model.pt");
    return 0;
}