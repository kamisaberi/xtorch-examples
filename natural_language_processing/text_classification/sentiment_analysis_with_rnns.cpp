#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>      // For string splitting (basic tokenization)
#include <map>
#include <algorithm>    // For std::transform, std::max_element
#include <iomanip>      // For std::fixed, std::setprecision

// --- Configuration ---
const int64_t VOCAB_SIZE = 1000;      // Max vocabulary size
const int64_t EMBED_DIM = 100;        // Embedding dimension
const int64_t HIDDEN_DIM = 128;       // LSTM hidden dimension
const int64_t NUM_LSTM_LAYERS = 1;    // Number of LSTM layers
const int64_t NUM_CLASSES = 2;        // e.g., Positive (1), Negative (0)
const double DROPOUT_PROB = 0.5;
const bool BIDIRECTIONAL_LSTM = true; // Whether to use a bidirectional LSTM

const int64_t BATCH_SIZE = 4;
const int64_t NUM_EPOCHS = 30;
const double LEARNING_RATE = 1e-3;
const int64_t LOG_INTERVAL = 5;

// Special token indices
const int64_t PAD_IDX = 0;
const int64_t UNK_IDX = 1;


// --- Simple Tokenizer & Vocabulary Builder ---
struct SimpleTokenizer {
    std::map<std::string, int64_t> word_to_idx;
    std::map<int64_t, std::string> idx_to_word;
    int64_t vocab_size_actual = 0;
    int64_t pad_idx_val = PAD_IDX;
    int64_t unk_idx_val = UNK_IDX;

    SimpleTokenizer(const std::vector<std::string>& texts, int64_t max_vocab_size_config) {
        word_to_idx["<pad>"] = pad_idx_val; idx_to_word[pad_idx_val] = "<pad>";
        word_to_idx["<unk>"] = unk_idx_val; idx_to_word[unk_idx_val] = "<unk>";
        vocab_size_actual = 2;

        std::map<std::string, int64_t> word_counts;
        for (const auto& text : texts) {
            std::string lower_text = text;
            std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
            std::stringstream ss(lower_text);
            std::string word;
            while (ss >> word) {
                // Basic punctuation removal (can be improved)
                word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
                if (!word.empty()) {
                    word_counts[word]++;
                }
            }
        }

        // Sort words by frequency (most frequent first)
        std::vector<std::pair<std::string, int64_t>> sorted_words(word_counts.begin(), word_counts.end());
        std::sort(sorted_words.begin(), sorted_words.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& pair : sorted_words) {
            if (vocab_size_actual >= max_vocab_size_config) break;
            if (word_to_idx.find(pair.first) == word_to_idx.end()) {
                word_to_idx[pair.first] = vocab_size_actual;
                idx_to_word[vocab_size_actual] = pair.first;
                vocab_size_actual++;
            }
        }
        std::cout << "Tokenizer: Actual vocab size: " << vocab_size_actual << std::endl;
    }

    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> encoded;
        std::string lower_text = text;
        std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
        std::stringstream ss(lower_text);
        std::string word;
        while (ss >> word) {
            word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
            if (word.empty()) continue;

            if (word_to_idx.count(word)) {
                encoded.push_back(word_to_idx.at(word));
            } else {
                encoded.push_back(unk_idx_val);
            }
        }
        return encoded;
    }
};


// --- Sentiment RNN Model ---
struct SentimentRNNImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};
    torch::nn::Dropout dropout{nullptr};
    bool is_bidirectional;

    SentimentRNNImpl(int64_t vocab_sz, int64_t embed_d, int64_t hidden_d, int64_t num_layers,
                     int64_t output_dim, double dropout_p, bool bidirectional, int64_t pad_idx)
        : is_bidirectional(bidirectional) {

        embedding = register_module("embedding", torch::nn::Embedding(
            torch::nn::EmbeddingOptions(vocab_sz, embed_d).padding_idx(pad_idx)));

        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(embed_d, hidden_d)
                .num_layers(num_layers)
                .bidirectional(bidirectional)
                .dropout(dropout_p) // Dropout between LSTM layers if num_layers > 1
                .batch_first(true))); // Input/output tensors have batch dim first

        // If bidirectional, output from LSTM is 2*hidden_d (concatenation of fwd and bwd)
        int64_t fc_input_dim = bidirectional ? hidden_d * 2 : hidden_d;
        fc = register_module("fc", torch::nn::Linear(fc_input_dim, output_dim));

        dropout = register_module("dropout", torch::nn::Dropout(dropout_p)); // Dropout before final FC
    }

    // input_seqs: [B, SeqLen] (padded sequence of token indices)
    // lengths: [B] (original lengths of sequences before padding) - for PackedSequence
    torch::Tensor forward(torch::Tensor input_seqs, torch::Tensor lengths) {
        torch::Tensor embedded = dropout(embedding(input_seqs)); // [B, SeqLen, EmbedDim]

        // Pack sequence to handle variable lengths efficiently (avoids computation on padding)
        // Ensure lengths are on CPU for pack_padded_sequence
        torch::Tensor packed_embedded = torch::nn::utils::rnn::pack_padded_sequence(
            embedded, lengths.cpu(), /*batch_first=*/true, /*enforce_sorted=*/false);

        auto lstm_out_packed_tuple = lstm(packed_embedded);
        // We are interested in the final hidden state (or concatenation of them for bidirectional)
        // lstm_out_packed_tuple.hidden is a tuple (h_n, c_n)
        // h_n: [num_layers * num_directions, B, HiddenDim]
        torch::Tensor hidden_n = std::get<0>(lstm_out_packed_tuple.get()); // h_n

        // If bidirectional: hidden_n contains [fwd_last_layer, bwd_last_layer, fwd_prev_layer, bwd_prev_layer, ...]
        // We want the last hidden state of the last layer.
        // For uni: hidden_n[-1]
        // For bi: concat(hidden_n[-2,:,:], hidden_n[-1,:,:])
        torch::Tensor last_hidden;
        if (is_bidirectional) {
            // hidden_n shape: [num_layers*2, B, H]
            // Last forward hidden state: hidden_n.view(num_layers, 2, B, H)[-1, 0]
            // Last backward hidden state: hidden_n.view(num_layers, 2, B, H)[-1, 1]
             last_hidden = torch::cat({hidden_n.index({-2, torch::indexing::Slice(), torch::indexing::Slice()}),
                                      hidden_n.index({-1, torch::indexing::Slice(), torch::indexing::Slice()})},
                                     /*dim=*/1); // [B, 2*HiddenDim]
        } else {
            last_hidden = hidden_n.index({-1, torch::indexing::Slice(), torch::indexing::Slice()}); // [B, HiddenDim]
        }

        last_hidden = dropout(last_hidden);
        torch::Tensor logits = fc(last_hidden); // [B, NumClasses]
        return logits; // No sigmoid needed if using BCEWithLogitsLoss or CrossEntropyLoss
    }
};
TORCH_MODULE(SentimentRNN);


// --- Dataset for Sentiment Analysis ---
struct SentimentDataset : torch::data::datasets::Dataset<SentimentDataset> {
    std::vector<std::vector<int64_t>> encoded_texts;
    std::vector<int64_t> labels;
    std::vector<int64_t> lengths; // Store original lengths

    SentimentDataset(const std::vector<std::string>& texts_in,
                     const std::vector<int64_t>& labels_in,
                     const SimpleTokenizer& tokenizer) {
        labels = labels_in;
        for (const auto& text : texts_in) {
            std::vector<int64_t> encoded = tokenizer.encode(text);
            encoded_texts.push_back(encoded);
            lengths.push_back(static_cast<int64_t>(encoded.size()));
        }
    }

    torch::data::Example<std::pair<torch::Tensor, torch::Tensor>, torch::Tensor> get(size_t index) override {
        // Here, we don't pad yet. Padding will be done in the custom collate function.
        torch::Tensor text_tensor = torch::tensor(encoded_texts[index], torch::kLong);
        torch::Tensor length_tensor = torch::tensor({lengths[index]}, torch::kLong); // Pass as tensor
        torch::Tensor label_tensor = torch::tensor(labels[index], torch::kLong);
        return {{text_tensor, length_tensor}, label_tensor};
    }

    torch::optional<size_t> size() const override {
        return labels.size();
    }
};

// Custom collate function to handle padding and lengths
struct PadCollate {
    int64_t pad_idx;
    PadCollate(int64_t pad_val) : pad_idx(pad_val) {}

    // Output: { (padded_sequences, lengths_tensor), stacked_labels }
    std::tuple<std::pair<torch::Tensor, torch::Tensor>, torch::Tensor> operator()(
        std::vector<torch::data::Example<std::pair<torch::Tensor, torch::Tensor>, torch::Tensor>> batch_samples) {

        std::vector<torch::Tensor> seqs, lengths_vec, labels_vec;
        int64_t max_len = 0;

        for (const auto& sample : batch_samples) {
            seqs.push_back(sample.data.first); // Unpadded sequence
            lengths_vec.push_back(sample.data.second); // Length as tensor {len}
            labels_vec.push_back(sample.target);
            if (sample.data.first.size(0) > max_len) {
                max_len = sample.data.first.size(0);
            }
        }
        if (max_len == 0 && !seqs.empty()) max_len = 1; // Handle empty sequences if any

        std::vector<torch::Tensor> padded_seqs;
        for (const auto& seq : seqs) {
            torch::Tensor padded_seq = torch::full({max_len}, pad_idx, seq.options());
            if (seq.size(0) > 0) { // Handle empty sequences correctly
                 padded_seq.slice(0, 0, seq.size(0)).copy_(seq);
            }
            padded_seqs.push_back(padded_seq);
        }

        return {
            {torch::stack(padded_seqs), torch::cat(lengths_vec)}, // Stacked lengths [B]
            torch::stack(labels_vec)
        };
    }
};


int main() {
    std::cout << "Sentiment Analysis with RNNs (LibTorch C++)" << std::endl;
    torch::manual_seed(1);
    std::cout << std::fixed << std::setprecision(4);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    // --- Dummy Data & Tokenizer ---
    std::vector<std::string> train_texts_raw = {
        "this movie is great and fantastic", "i really loved this film", "awesome experience wonderful acting",
        "absolutely terrible waste of time", "hated every minute of it", "poor plot and bad characters",
        "a decent watch enjoyable for most parts", "not bad but not amazing either",
        "i am happy with this product", "very sad and disappointing service"
    };
    std::vector<int64_t> train_labels_raw = {1, 1, 1, 0, 0, 0, 1, 0, 1, 0}; // 1: positive, 0: negative

    SimpleTokenizer tokenizer(train_texts_raw, VOCAB_SIZE);

    // --- Model ---
    SentimentRNN model(tokenizer.vocab_size_actual, EMBED_DIM, HIDDEN_DIM, NUM_LSTM_LAYERS,
                       NUM_CLASSES, DROPOUT_PROB, BIDIRECTIONAL_LSTM, PAD_IDX);
    model->to(device);
    std::cout << "Sentiment RNN model created. Parameters: "
              << std::accumulate(model->parameters().begin(), model->parameters().end(), 0L,
                                 [](long sum, const torch::Tensor& t){ return sum + t.numel(); })
              << std::endl;

    // --- DataLoader ---
    PadCollate collator(PAD_IDX);
    auto train_dataset = SentimentDataset(train_texts_raw, train_labels_raw, tokenizer);
    // The DataLoader type needs to match what collator returns for batch.data and batch.target
    // This is simplified here as we are manually iterating. For a full DataLoader:
    // auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    //     std::move(train_dataset), // This should be .map(collator) or collator passed to loader options
    //     torch::data::DataLoaderOptions().batch_size(BATCH_SIZE)
    // );
    // For now, manual batching:
    std::cout << "Dataset created." << std::endl;

    // --- Optimizer & Loss ---
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    // For binary classification (0 or 1), BCEWithLogitsLoss is suitable if output is 1 logit.
    // If output is 2 logits (one for each class), CrossEntropyLoss is fine.
    // Our model outputs NUM_CLASSES logits.
    torch::nn::CrossEntropyLoss criterion;
    std::cout << "Optimizer and Loss created." << std::endl;

    // --- Training Loop (Manual Batching) ---
    std::cout << "\nStarting Training..." << std::endl;
    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        model->train();
        double epoch_loss = 0.0;
        int64_t num_correct_epoch = 0;
        int64_t num_samples_epoch = 0;
        size_t batch_idx_count = 0;

        // Manual shuffling of indices for batching
        std::vector<size_t> epoch_indices(train_dataset.size().value());
        std::iota(epoch_indices.begin(), epoch_indices.end(), 0);
        std::shuffle(epoch_indices.begin(), epoch_indices.end(), std::mt19937(std::random_device{}()));

        for (size_t i = 0; i < epoch_indices.size(); i += BATCH_SIZE) {
            optimizer.zero_grad();

            std::vector<torch::data::Example<std::pair<torch::Tensor, torch::Tensor>, torch::Tensor>> current_batch_samples;
            for (size_t j = i; j < std::min(i + BATCH_SIZE, epoch_indices.size()); ++j) {
                current_batch_samples.push_back(train_dataset.get(epoch_indices[j]));
            }
            if (current_batch_samples.empty()) continue;

            auto collated_batch = collator(current_batch_samples);
            torch::Tensor padded_seqs = collated_batch.first.first.to(device);
            torch::Tensor lengths = collated_batch.first.second.to(device); // Keep on device for model if needed
            torch::Tensor labels = collated_batch.second.to(device);

            torch::Tensor logits = model->forward(padded_seqs, lengths); // lengths to CPU inside model
            torch::Tensor loss = criterion(logits, labels);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
            torch::Tensor predictions = torch::argmax(logits, /*dim=*/1);
            num_correct_epoch += (predictions == labels).sum().item<int64_t>();
            num_samples_epoch += labels.size(0);
            batch_idx_count++;

            if (batch_idx_count % LOG_INTERVAL == 0 || i + BATCH_SIZE >= epoch_indices.size()) {
                 std::cout << "Epoch: " << epoch << "/" << NUM_EPOCHS
                          << " | Batch: " << batch_idx_count
                          << " | Loss: " << loss.item<double>() << std::endl;
            }
        }
        double avg_epoch_loss = (num_samples_epoch > 0) ? (epoch_loss / batch_idx_count) : 0.0;
        double epoch_accuracy = (num_samples_epoch > 0) ? (static_cast<double>(num_correct_epoch) / num_samples_epoch) : 0.0;
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "Epoch: " << epoch << " Avg Loss: " << avg_epoch_loss
                  << " | Accuracy: " << epoch_accuracy << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
    }
    std::cout << "Training finished." << std::endl;

    // --- Example Inference ---
    model->eval();
    torch::NoGradGuard no_grad;
    std::vector<std::string> test_sentences = {"this is a wonderful film", "i really disliked the movie"};
    for(const auto& sentence : test_sentences) {
        std::vector<int64_t> encoded_test = tokenizer.encode(sentence);
        if (encoded_test.empty()) { // Handle empty encoding (e.g., all OOV and no UNK or short)
             std::cout << "Could not encode sentence for inference: \"" << sentence << "\"" << std::endl;
             continue;
        }
        torch::Tensor test_seq = torch::tensor(encoded_test, torch::kLong).unsqueeze(0).to(device); // [1, SeqLen]
        torch::Tensor test_len = torch::tensor({static_cast<int64_t>(encoded_test.size())}, torch::kLong).to(device); // [1]

        torch::Tensor test_logits = model->forward(test_seq, test_len);
        torch::Tensor probabilities = torch::softmax(test_logits, /*dim=*/1);
        torch::Tensor predicted_class = torch::argmax(probabilities, /*dim=*/1);

        std::cout << "\nSentence: \"" << sentence << "\"" << std::endl;
        std::cout << "Logits: " << test_logits << std::endl;
        std::cout << "Probabilities (Neg, Pos): " << probabilities << std::endl;
        std::cout << "Predicted class: " << predicted_class.item<int64_t>()
                  << (predicted_class.item<int64_t>() == 1 ? " (Positive)" : " (Negative)") << std::endl;
    }
    // torch::save(model, "sentiment_rnn_model.pt");
    return 0;
}