#include <torch/torch.h>
#include <torch/script.h> // For torch::jit::load
#include <iostream>
#include <vector>
#include <string>
#include <sstream>      // For string splitting
#include <map>
#include <fstream>      // For reading vocab

// --- Configuration ---
const int64_t MAX_SEQ_LENGTH = 128;   // Max sequence length for BERT
const int64_t BERT_HIDDEN_DIM = 768;  // For bert-base-uncased
const int64_t NUM_CLASSES = 2;        // E.g., for sentiment classification (positive/negative)
const int64_t BATCH_SIZE = 8;
const int64_t NUM_EPOCHS = 3;
const double LEARNING_RATE = 2e-5;    // Common fine-tuning LR for BERT
const int64_t LOG_INTERVAL = 5;
const std::string BERT_MODEL_PATH = "bert_traced.pt"; // Path to your TorchScript BERT model
const std::string VOCAB_PATH = "vocab.txt"; // Simplified vocab path

// --- Simplified Tokenizer (Conceptual - NOT WordPiece) ---
// This is a placeholder. A real BERT requires a WordPiece tokenizer.
struct SimpleTokenizer {
    std::map<std::string, int64_t> vocab;
    int64_t unk_token_id = 100; // [UNK]
    int64_t cls_token_id = 101; // [CLS]
    int64_t sep_token_id = 102; // [SEP]
    int64_t pad_token_id = 0;   // [PAD]

    SimpleTokenizer(const std::string& vocab_file) {
        std::ifstream file(vocab_file);
        std::string line;
        int64_t id = 0;
        if (file.is_open()) {
            while (getline(file, line)) {
                vocab[line] = id++;
            }
            file.close();
            // Ensure special tokens are there if not in vocab.txt (usually they are)
            if (vocab.find("[UNK]") == vocab.end()) vocab["[UNK]"] = unk_token_id; else unk_token_id = vocab["[UNK]"];
            if (vocab.find("[CLS]") == vocab.end()) vocab["[CLS]"] = cls_token_id; else cls_token_id = vocab["[CLS]"];
            if (vocab.find("[SEP]") == vocab.end()) vocab["[SEP]"] = sep_token_id; else sep_token_id = vocab["[SEP]"];
            if (vocab.find("[PAD]") == vocab.end()) vocab["[PAD]"] = pad_token_id; else pad_token_id = vocab["[PAD]"];
            std::cout << "SimpleTokenizer: Loaded " << vocab.size() << " tokens. UNK="<<unk_token_id << ", CLS="<<cls_token_id << std::endl;
        } else {
            std::cerr << "SimpleTokenizer: Unable to open vocab file: " << vocab_file << std::endl;
            // Add minimal vocab for placeholder to run
            vocab["[PAD]"] = 0; vocab["[UNK]"] = 1; vocab["[CLS]"] = 2; vocab["[SEP]"] = 3;
            vocab["this"] = 4; vocab["is"] = 5; vocab["a"] = 6; vocab["sentence"] = 7; vocab["example"] = 8;
            unk_token_id = 1; cls_token_id = 2; sep_token_id = 3; pad_token_id = 0;
        }
    }

    std::vector<std::string> tokenize_text(const std::string& text) const {
        std::vector<std::string> tokens;
        std::string lower_text = text;
        std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
        std::stringstream ss(lower_text);
        std::string word;
        while (ss >> word) { // Basic whitespace split
            tokens.push_back(word);
        }
        return tokens;
    }

    // Encodes a single sentence
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> encode(
        const std::string& text, int64_t max_len) const {

        std::vector<std::string> tokens = tokenize_text(text);

        std::vector<int64_t> input_ids;
        input_ids.push_back(cls_token_id); // Start with [CLS]

        for (const auto& token_str : tokens) {
            if (input_ids.size() >= max_len - 1) break; // -1 for [SEP]
            if (vocab.count(token_str)) {
                input_ids.push_back(vocab.at(token_str));
            } else {
                input_ids.push_back(unk_token_id);
            }
        }
        input_ids.push_back(sep_token_id); // End with [SEP]

        std::vector<int64_t> attention_mask(input_ids.size(), 1);
        std::vector<int64_t> token_type_ids(input_ids.size(), 0); // All 0 for single sentence

        // Padding
        while (input_ids.size() < max_len) {
            input_ids.push_back(pad_token_id);
            attention_mask.push_back(0);
            token_type_ids.push_back(0);
        }

        return {input_ids, attention_mask, token_type_ids};
    }
};

// --- Model: BERT (from TorchScript) + Classification Head ---
struct BertForSequenceClassificationImpl : torch::nn::Module {
    torch::jit::script::Module bert_module; // Loaded TorchScript BERT
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear classifier{nullptr};

    BertForSequenceClassificationImpl(const std::string& bert_path, int64_t bert_hidden_dim, int64_t num_classes) {
        try {
            bert_module = torch::jit::load(bert_path);
            // No need to register_module for jit::ScriptModule if just using its forward
        } catch (const c10::Error& e) {
            std::cerr << "Error loading TorchScript BERT model from " << bert_path << ": " << e.what() << std::endl;
            // You might want to throw or exit here
        }
        dropout = register_module("dropout", torch::nn::Dropout(0.1)); // Standard dropout for BERT heads
        classifier = register_module("classifier", torch::nn::Linear(bert_hidden_dim, num_classes));
    }

    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor attention_mask, torch::Tensor token_type_ids) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_ids);
        inputs.push_back(attention_mask);
        // inputs.push_back(token_type_ids); // Add if your traced model expects it

        // The output of a traced Hugging Face BertModel is often a tuple (last_hidden_state, pooler_output)
        // Or it could be a custom output object if not wrapped.
        // We typically use the pooler_output for classification.
        // If it's an object, you'd access .pooler_output
        // If it's a tuple, pooler_output is usually the second element.
        auto bert_outputs_ivalue = bert_module.forward(inputs);

        torch::Tensor pooler_output;
        if (bert_outputs_ivalue.isTuple()) {
            // Assuming pooler_output is the second element if traced from base BertModel
            // (last_hidden_state, pooler_output) = bert_model(...)
            pooler_output = bert_outputs_ivalue.toTuple()->elements()[1].toTensor();
        } else if (bert_outputs_ivalue.isTensor()) {
            // If the traced model was modified to ONLY return the pooler_output
            pooler_output = bert_outputs_ivalue.toTensor();
        } else {
            std::cerr << "BERT output type not recognized. Expected Tensor or Tuple." << std::endl;
            // Return a dummy tensor or throw
            return torch::empty({input_ids.size(0), classifier->options.out_features()});
        }
        // pooler_output should be [batch_size, bert_hidden_dim]

        pooler_output = dropout(pooler_output);
        torch::Tensor logits = classifier(pooler_output);
        return logits;
    }
};
TORCH_MODULE(BertForSequenceClassification);

// --- Dummy Dataset ---
struct TextClassificationDataset : torch::data::datasets::Dataset<TextClassificationDataset> {
    std::vector<std::string> texts;
    std::vector<int64_t> labels;
    const SimpleTokenizer& tokenizer; // Reference to the tokenizer
    int64_t max_len;

    TextClassificationDataset(const std::vector<std::string>& texts_in,
                              const std::vector<int64_t>& labels_in,
                              const SimpleTokenizer& tok, int64_t max_seq_len)
        : texts(texts_in), labels(labels_in), tokenizer(tok), max_len(max_seq_len) {}

    // Output: {input_ids, attention_mask, token_type_ids}, label
    torch::data::Example<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>, torch::Tensor> get(size_t index) override {
        auto [input_ids_vec, attention_mask_vec, token_type_ids_vec] = tokenizer.encode(texts[index], max_len);

        torch::Tensor input_ids_tensor = torch::tensor(input_ids_vec, torch::kLong);
        torch::Tensor attention_mask_tensor = torch::tensor(attention_mask_vec, torch::kLong);
        torch::Tensor token_type_ids_tensor = torch::tensor(token_type_ids_vec, torch::kLong);
        torch::Tensor label_tensor = torch::tensor(labels[index], torch::kLong);

        return {{input_ids_tensor, attention_mask_tensor, token_type_ids_tensor}, label_tensor};
    }

    torch::optional<size_t> size() const override {
        return texts.size();
    }
};

// Custom collate function for batching
struct CustomCollate {
    torch::data::Example<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>, torch::Tensor> operator()(
        std::vector<torch::data::Example<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>, torch::Tensor>> batch_samples) {

        std::vector<torch::Tensor> batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_labels;
        batch_input_ids.reserve(batch_samples.size());
        batch_attention_masks.reserve(batch_samples.size());
        batch_token_type_ids.reserve(batch_samples.size());
        batch_labels.reserve(batch_samples.size());

        for (const auto& sample : batch_samples) {
            batch_input_ids.push_back(std::get<0>(sample.data));
            batch_attention_masks.push_back(std::get<1>(sample.data));
            batch_token_type_ids.push_back(std::get<2>(sample.data));
            batch_labels.push_back(sample.target);
        }

        return {
            {torch::stack(batch_input_ids), torch::stack(batch_attention_masks), torch::stack(batch_token_type_ids)},
            torch::stack(batch_labels)
        };
    }
};


int main() {
    std::cout << "Fine-tuning BERT (Conceptual - LibTorch C++)" << std::endl;
    torch::manual_seed(1);
    std::cout << std::fixed << std::setprecision(4);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    // --- Create a dummy vocab.txt if it doesn't exist ---
    std::ofstream vocab_out(VOCAB_PATH, std::ios_base::app); // Append or create
    if (vocab_out.is_open()) {
        // Check if file is empty or very small (might indicate it needs basic vocab)
        vocab_out.seekp(0, std::ios::end);
        if (vocab_out.tellp() < 50) { // Arbitrary small size
             vocab_out << "[PAD]\n[UNK]\n[CLS]\n[SEP]\nthis\nis\na\nsentence\nexample\ngood\nbad\npositive\nnegative\ntext\n";
        }
        vocab_out.close();
    } else {
        std::cerr << "Could not open " << VOCAB_PATH << " for writing dummy vocab." << std::endl;
    }


    SimpleTokenizer tokenizer(VOCAB_PATH); // Initialize tokenizer

    // --- Model ---
    // Ensure bert_traced.pt exists in your build directory or provide full path
    BertForSequenceClassification model(BERT_MODEL_PATH, BERT_HIDDEN_DIM, NUM_CLASSES);
    model->to(device);
    // Freeze BERT layers initially (optional, common for fine-tuning)
    // for (auto& param : model->bert_module.parameters()) { // This won't work directly with jit::Module
    //    param.set_requires_grad(false);
    // }
    // Note: Freezing parameters of a torch::jit::script::Module is tricky.
    // Usually, you'd handle freezing in Python *before* tracing if needed,
    // or fine-tune all layers with a small LR.
    std::cout << "BERT with classification head model created." << std::endl;

    // --- Dummy Data ---
    std::vector<std::string> train_texts = {
        "this is a good positive example sentence", "what a bad negative text",
        "another positive sentence here", "this example is quite negative" ,
        "this is great and good", "this is so bad and terrible"
    };
    std::vector<int64_t> train_labels = {1, 0, 1, 0, 1, 0}; // 1 for positive, 0 for negative

    auto train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH)
                             .map(CustomCollate());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE));
    std::cout << "Dummy DataLoader created." << std::endl;

    // --- Optimizer & Loss ---
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(LEARNING_RATE));
    torch::nn::CrossEntropyLoss criterion;
    std::cout << "Optimizer and Loss created." << std::endl;

    // --- Training Loop ---
    std::cout << "\nStarting Fine-tuning..." << std::endl;
    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        model->train();
        double epoch_loss = 0.0;
        size_t batch_idx = 0;

        for (auto& batch : *train_loader) {
            optimizer.zero_grad();

            auto inputs_tuple = batch.data; // This is std::tuple<Tensor, Tensor, Tensor>
            torch::Tensor input_ids = std::get<0>(inputs_tuple).to(device);
            torch::Tensor attention_mask = std::get<1>(inputs_tuple).to(device);
            torch::Tensor token_type_ids = std::get<2>(inputs_tuple).to(device);
            torch::Tensor labels = batch.target.to(device);

            torch::Tensor logits = model->forward(input_ids, attention_mask, token_type_ids);
            torch::Tensor loss = criterion(logits, labels);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
            if (batch_idx % LOG_INTERVAL == 0 || batch_idx == (train_texts.size()/BATCH_SIZE) ) {
                 std::cout << "Epoch: " << epoch << "/" << NUM_EPOCHS
                          << " | Batch: " << batch_idx << "/" << (train_texts.size()/BATCH_SIZE)
                          << " | Loss: " << loss.item<double>() << std::endl;
            }
            batch_idx++;
        }
         std::cout << "-------------------------------------------------------" << std::endl;
         std::cout << "Epoch: " << epoch << " Average Loss: " << (epoch_loss / std::max(1.0, (double)batch_idx)) << std::endl;
         std::cout << "-------------------------------------------------------" << std::endl;
    }
    std::cout << "Fine-tuning finished." << std::endl;

    // --- Example Inference ---
    model->eval();
    torch::NoGradGuard no_grad;
    std::string test_sentence = "this is a positive test";
    auto [test_ids_vec, test_mask_vec, test_ttids_vec] = tokenizer.encode(test_sentence, MAX_SEQ_LENGTH);

    torch::Tensor test_input_ids = torch::tensor(test_ids_vec, torch::kLong).unsqueeze(0).to(device); // Add batch dim
    torch::Tensor test_attention_mask = torch::tensor(test_mask_vec, torch::kLong).unsqueeze(0).to(device);
    torch::Tensor test_token_type_ids = torch::tensor(test_ttids_vec, torch::kLong).unsqueeze(0).to(device);

    torch::Tensor test_logits = model->forward(test_input_ids, test_attention_mask, test_token_type_ids);
    torch::Tensor probabilities = torch::softmax(test_logits, /*dim=*/1);
    torch::Tensor predicted_class = torch::argmax(probabilities, /*dim=*/1);

    std::cout << "\nInference on: \"" << test_sentence << "\"" << std::endl;
    std::cout << "Logits: " << test_logits << std::endl;
    std::cout << "Probabilities: " << probabilities << std::endl;
    std::cout << "Predicted class: " << predicted_class.item<int64_t>() << std::endl;

    // torch::save(model, "finetuned_bert_classifier.pt"); // Save the whole classifier
    return 0;
}