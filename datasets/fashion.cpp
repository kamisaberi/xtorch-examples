#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <vector>

// Helper function to read MNIST data
std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& file_path, int num_images) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    // Read metadata
    int32_t magic_number, num_items, rows, cols;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_items), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    // Convert endianess
    magic_number = __builtin_bswap32(magic_number);
    num_items = __builtin_bswap32(num_items);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(rows * cols));
    for (int i = 0; i < num_images; i++) {
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }

    file.close();
    return images;
}

std::vector<uint8_t> read_mnist_labels(const std::string& file_path, int num_labels) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    // Read metadata
    int32_t magic_number, num_items;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_items), 4);

    // Convert endianess
    magic_number = __builtin_bswap32(magic_number);
    num_items = __builtin_bswap32(num_items);

    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    file.close();
    return labels;
}

// Define the FashionMNIST dataset class
class FashionMNISTDataset : public torch::data::datasets::Dataset<FashionMNISTDataset> {
private:
    torch::Tensor images_;
    torch::Tensor labels_;

public:
    // Constructor: Loads images and labels from files
    FashionMNISTDataset(const std::string& images_path, const std::string& labels_path, int num_samples) {
        auto images_data = read_mnist_images(images_path, num_samples);
        auto labels_data = read_mnist_labels(labels_path, num_samples);

        images_ = torch::empty({num_samples, 1, 28, 28}, torch::kUInt8);
        labels_ = torch::empty(num_samples, torch::kUInt8);

        for (int i = 0; i < num_samples; i++) {
            images_[i] = torch::from_blob(images_data[i].data(), {1, 28, 28}, torch::kUInt8).clone();
            labels_[i] = labels_data[i];
        }

        images_ = images_.to(torch::kFloat32).div_(255.0); // Normalize to [0, 1]
        labels_ = labels_.to(torch::kInt64);               // Convert to int64 for loss functions
    }

    // Override `get` method to return a single data sample
    torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }

    // Override `size` method to return the number of samples
    torch::optional<size_t> size() const override {
        return labels_.size(0);
    }
};

// Main function to test the dataset
int main() {
    const std::string images_path = "/home/kami/datasets/FashionMNIST/raw/train-images-idx3-ubyte"; // Replace with actual path
    const std::string labels_path = "/home/kami/datasets/FashionMNIST/raw/train-labels-idx1-ubyte"; // Replace with actual path

    try {
        auto dataset = FashionMNISTDataset(images_path, labels_path, 60000)
                .map(torch::data::transforms::Normalize<>(0.5, 0.5)) // Normalize to [-1, 1]
                .map(torch::data::transforms::Stack<>());

        auto data_loader = torch::data::make_data_loader(dataset, /*batch_size=*/64);

        for (auto& batch : *data_loader) {
            auto images = batch.data;
            auto labels = batch.target;

            std::cout << "Batch size: " << images.size(0) << std::endl;
            std::cout << labels << std::endl;
            break; // Test one batch
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
