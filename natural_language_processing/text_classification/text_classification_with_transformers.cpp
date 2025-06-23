#include <torch/torch.h>
#include <torch/script.h> // For torch::jit::load
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <fstream>
#include <algorithm>
#include <iomanip>

// --- Configuration ---
const int64_t MAX_SEQ_LENGTH = 128;
// For 'distilbert-base-uncased', hidden_dim is 768. BERT-base is also 768.
const int64_t TRANSFORMER_HIDDEN_DIM = 768;
const int64_t NUM_CLASSES = 2;        // e.g., Positive/Negative
const int64_t BATCH_SIZE = 8;
const int64_t NUM_EPOCHS = 3;         // Fine-tuning usually requires fewer epochs
const double LEARNING_RATE = 2e-5;    // Common fine-tuning LR
const int64_t LOG_INTERVAL = 5;
const std::string TRANSFORMER_MODEL_PATH = "transformer_traced.pt"; // From Python script
const std::string VOCAB_PATH = "./transformer_vocab/vocab.txt";     // From Python script


// --- Simplified Tokenizer (Conceptual - NOT WordPiece/BPE) ---
// This is a major placeholder. Real Transformer models need their specific tokenizers.
struct SimpleVocabTokenizer {
    std::map<std::string, int64_t> vocab;
    int64_t unk_token_id_ = 100; // Default [UNK] id from bert-base-uncased vocab
    int64_t cls_token_id_ = 101; // Default [CLS]
    int64_t sep_token_id_ = 102; // Default [SEP]
    int64_t pad_token_id_ = 0;   // Default [PAD]
    std::string cls_token_str_ = "[CLS]";
    std::string sep_token_str_ = "[SEP]";
    std::string pad_token_str_ = "[PAD]";
    std::string unk_token_str_ = "[UNK]";


    SimpleVocabTokenizer(const std::string& vocab_file_path) {
        std::ifstream vocab_file(vocab_file_path);
        std::string line;
        int64_t id = 0;
        if (vocab_file.is_open()) {
            while (getline(vocab_file, line)) {
                // Remove \r if present (for files from Windows)
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                vocab[line] = id++;
            }
            vocab_file.close();
            // Try to set special token IDs from loaded vocab
            if (vocab.count(cls_token_str_)) cls_token_id_ = vocab[cls_token_str_]; else std::cerr << "Warning: CLS token not in vocab!\n";
            if (vocab.count(sep_token_str_)) sep_token_id_ = vocab[sep_token_str_]; else std::cerr << "Warning: SEP token not in vocab!\n";
            if (vocab.count(pad_token_str_)) pad_token_id_ = vocab[pad_token_str_]; else std::cerr << "Warning: PAD token not in vocab!\n";
            if (vocab.count(unk_token_str_)) unk_token_id_ = vocab[unk_token_str_]; else std::cerr << "Warning: UNK token not in vocab!\n";
            std::cout << "Tokenizer: Loaded " << vocab.size() << " tokens. PAD=" << pad_token_id_ << ", CLS=" << cls_token_id_ << std::endl;
        } else {
            std::cerr << "SimpleTokenizer: Unable to open vocab file: " << vocab_file_path << std::endl;
            // Fallback for placeholder
            vocab[pad_token_str_] = 0; vocab[unk_token_str_] = 1; vocab[cls_token_str_] = 2; vocab[sep_token_str_] = 3;
            vocab["this"] = 4; vocab["is"] = 5; vocab["a"] = 6; vocab["sentence"] = 7;
            pad_token_id_ = 0; unk_token_id_ = 1; cls_token_id_ = 2; sep_token_id_ = 3;
        }
    }

    std::vector<std::string> tokenize_text_basic(const std::string& text) const {
        std::vector<std::string> tokens;
        std::string lower_text = text;
        std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
        std::stringstream ss(lower_text);
        std::string word;
        while (ss >> word) { // Basic whitespace split
             // A real tokenizer (WordPiece/BPE) would do subword tokenization here
            tokens.push_back(word);
        }
        return tokens;
    }

    // Encodes a single sentence for classification
    std::tuple<std::vector<int64_t>, std::vector<int64_t>> encode_for_classification(
        const std::string& text, int64_t max_len) const {

        std::vector<std::string> token_strings = tokenize_text_basic(text); // Placeholder

        std::vector<int64_t> input_ids;
        input_ids.push_back(cls_token_id_);

        for (const auto& token_str : token_strings) {
            if (input_ids.size() >= max_len - 1) break; // -1 for [SEP]
            if (vocab.count(token_str)) {
                input_ids.push_back(vocab.at(token_str));
            } else {
                input_ids.push_back(unk_token_id_);
            }
        }
        input_ids.push_back(sep_token_id_);

        std::vector<int64_t> attention_mask(input_ids.size(), 1);

        // Padding
        while (input_ids.size() < max_len) {
            input_ids.push_back(pad_token_id_);
            attention_mask.push_back(0);
        }
        // Truncation (if somehow still too long, though break above should prevent)
        if (input_ids.size() > max_len) {
            input_ids.resize(max_len);
            attention_mask.resize(max_len);
            if (input_ids.back() != sep_token_id_ ) input_ids.back() = sep_token_id_; // Ensure SEP if truncated
        }

        return {input_ids, attention_mask};
    }
};

// --- Model: Transformer (from TorchScript) + Classification Head ---
struct TransformerForSequenceClassificationImpl : torch::nn::Module {
    torch::jit::script::Module transformer_module;
    torch::nn::Linear classifier{nullptr};
    torch::nn::Dropout dropout{nullptr}; // Often a dropout before the classifier

    TransformerForSequenceClassificationImpl(const std::string& transformer_path, int64_t transformer_hidden_dim, int64_t num_classes) {
        try {
            transformer_module = torch::jit::load(transformer_path);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading TorchScript Transformer model from " << transformer_path << ": " << e.what() << std::endl;
            throw; // Critical error
        }
        dropout = register_module("dropout", torch::nn::Dropout(0.1)); // Typical dropout rate
        classifier = register_module("classifier", torch::nn::Linear(transformer_hidden_dim, num_classes));
    }

    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor attention_mask) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_ids);
        inputs.push_back(attention_mask);
        // If your traced model takes token_type_ids, add it here:
        // inputs.push_back(token_type_ids);

        // Output from the traced base model (e.g., from AutoModel) is typically last_hidden_state
        // Shape: [batch_size, seq_len, hidden_dim]
        auto transformer_output_ivalue = transformer_module.forward(inputs);
        torch::Tensor last_hidden_state;

        if (transformer_output_ivalue.isTensor()) {
            last_hidden_state = transformer_output_ivalue.toTensor();
        } else if (transformer_output_ivalue.isTuple() && transformer_output_ivalue.toTuple()->elements().size() > 0 &&
                   transformer_output_ivalue.toTuple()->elements()[0].isTensor()) {
            // If traced model returns a tuple, e.g. (last_hidden_state, other_outputs...)
            last_hidden_state = transformer_output_ivalue.toTuple()->elements()[0].toTensor();
        } else {
             std::cerr << "Unexpected output type from traced transformer model." << std::endl;
             throw std::runtime_error("Unexpected transformer output type.");
        }


        // For classification, we typically use the representation of the [CLS] token,
        // which is the first token in the sequence.
        // last_hidden_state: [batch_size, seq_len, hidden_dim]
        // cls_representation: [batch_size, hidden_dim]
        torch::Tensor cls_representation = last_hidden_state.select(/*dim=*/1, /*index=*/0);
        // Alternative: Some models (like BERT) have a 'pooler_output' which is a linear transformation
        // of the [CLS] token's representation. If your traced model provides that, use it.
        // If DistilBERT, this CLS token approach is standard.

        cls_representation = dropout(cls_representation);
        torch::Tensor logits = classifier(cls_representation);
        return logits;
    }
};
TORCH_MODULE(TransformerForSequenceClassification);

// --- Dataset & DataLoader ---
struct TextDataset : torch::data::datasets::Dataset<TextDataset> {
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> input_features; // pairs of (input_ids, attention_mask)
    std::vector<int64_t> labels_vec;

    TextDataset(const std::vector<std::string>& texts,
                const std::vector<int64_t>& sentiments,
                const SimpleVocabTokenizer& tokenizer,
                int64_t max_len) {
        labels_vec = sentiments;
        for (const auto& text : texts) {
            auto [ids, mask] = tokenizer.encode_for_classification(text, max_len);
            input_features.push_back({ids, mask});
        }
    }

    torch::data::Example<std::pair<torch::Tensor, torch::Tensor>, torch::Tensor> get(size_t index) override {
        torch::Tensor input_ids_tensor = torch::tensor(input_features[index].first, torch::kLong);
        torch::Tensor attention_mask_tensor = torch::tensor(input_features[index].second, torch::kLong);
        torch::Tensor label_tensor = torch::tensor(labels_vec[index], torch::kLong);
        return {{input_ids_tensor, attention_mask_tensor}, label_tensor};
    }

    torch::optional<size_t> size() const override { return labels_vec.size(); }
};

struct TextClassificationCollate {
    auto operator()(std::vector<torch::data::Example<std::pair<torch::Tensor, torch::Tensor>, torch::Tensor>> batch_samples) {
        std::vector<torch::Tensor> batch_input_ids, batch_attention_masks, batch_labels;
        for (const auto& sample : batch_samples) {
            batch_input_ids.push_back(sample.data.first);
            batch_attention_masks.push_back(sample.data.second);
            batch_labels.push_back(sample.target);
        }
        return std::make_tuple(torch::stack(batch_input_ids),
                               torch::stack(batch_attention_masks),
                               torch::stack(batch_labels));
    }
};


int main() {
    std::cout << "Text Classification with Transformers (Conceptual - LibTorch C++)" << std::endl;
    torch::manual_seed(42);
    std::cout << std::fixed << std::setprecision(4);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    // --- Create a dummy vocab.txt if it doesn't exist (using python script's output dir) ---
    // Ensure the directory ./transformer_vocab exists
    // std::filesystem::create_directories("./transformer_vocab"); // C++17
    // For older C++, you might need system calls or ensure dir exists manually
    std::ofstream vocab_out(VOCAB_PATH, std::ios_base::trunc); // Overwrite or create
    if (vocab_out.is_open()) {
        vocab_out << "[PAD]\n[UNK]\n[CLS]\n[SEP]\nthis\nis\na\nsentence\nexample\ngood\nbad\npositive\nnegative\ntext\n";
        vocab_out.close();
        std::cout << "Created/Overwrote dummy " << VOCAB_PATH << std::endl;
    } else {
        std::cerr << "Could not open " << VOCAB_PATH << " for writing dummy vocab. Ensure './transformer_vocab/' directory exists." << std::endl;
        return 1;
    }
    // IMPORTANT: For a real run, use the vocab.txt generated by the Python script.

    SimpleVocabTokenizer tokenizer(VOCAB_PATH);

    TransformerForSequenceClassification model(TRANSFORMER_MODEL_PATH, TRANSFORMER_HIDDEN_DIM, NUM_CLASSES);
    model->to(device);
    std::cout << "Transformer classification model created." << std::endl;

    std::vector<std::string> train_texts_raw = {
        "this movie is great and fantastic", "i really loved this film", "awesome experience wonderful acting",
        "absolutely terrible waste of time", "hated every minute of it", "poor plot and bad characters",
        "a decent watch enjoyable for most parts", "not bad but not amazing either",
        "i am happy with this product", "very sad and disappointing service"
    };
    std::vector<int64_t> train_labels_raw = {1, 1, 1, 0, 0, 0, 1, 0, 1, 0}; // 1: positive, 0: negative

    // Manual dataset creation and batching for simplicity with custom collate
    auto train_dataset = TextDataset(train_texts_raw, train_labels_raw, tokenizer, MAX_SEQ_LENGTH);
    TextClassificationCollate collator;
    std::cout << "Dataset created." << std::endl;


    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(LEARNING_RATE));
    torch::nn::CrossEntropyLoss criterion;
    std::cout << "Optimizer and Loss created." << std::endl;

    std::cout << "\nStarting Fine-tuning..." << std::endl;
    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        model->train();
        double epoch_loss = 0.0;
        int64_t num_correct_epoch = 0;
        int64_t num_samples_epoch = 0;
        size_t batch_idx_count = 0;

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
            torch::Tensor input_ids = std::get<0>(collated_batch).to(device);
            torch::Tensor attention_mask = std::get<1>(collated_batch).to(device);
            torch::Tensor labels = std::get<2>(collated_batch).to(device);

            torch::Tensor logits = model->forward(input_ids, attention_mask);
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
    std::cout << "Fine-tuning finished." << std::endl;

    // --- Example Inference ---
    model->eval();
    torch::NoGradGuard no_grad;
    std::string test_sentence = "this is a positive test example";
    auto [test_ids_vec, test_mask_vec] = tokenizer.encode_for_classification(test_sentence, MAX_SEQ_LENGTH);

    torch::Tensor test_input_ids = torch::tensor(test_ids_vec, torch::kLong).unsqueeze(0).to(device);
    torch::Tensor test_attention_mask = torch::tensor(test_mask_vec, torch::kLong).unsqueeze(0).to(device);

    torch::Tensor test_logits = model->forward(test_input_ids, test_attention_mask);
    torch::Tensor probabilities = torch::softmax(test_logits, /*dim=*/1);
    torch::Tensor predicted_class = torch::argmax(probabilities, /*dim=*/1);

    std::cout << "\nInference on: \"" << test_sentence << "\"" << std::endl;
    std::cout << "Logits: " << test_logits << std::endl;
    std::cout << "Probabilities (Neg, Pos): " << probabilities << std::endl;
    std::cout << "Predicted class: " << predicted_class.item<int64_t>()
              << (predicted_class.item<int64_t>() == 1 ? " (Positive)" : " (Negative)") << std::endl;

    // torch::save(model, "finetuned_transformer_classifier.pt");
    return 0;
}