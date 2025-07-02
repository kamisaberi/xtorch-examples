#include <onnxruntime/onnxruntime_cxx_api.h>
#include <curl/curl.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <random>

// Callback function for libcurl to write data to file
size_t write_data(void* ptr, size_t size, size_t nmemb, void* stream) {
    size_t written = fwrite(ptr, size, nmemb, (FILE*)stream);
    return written;
}

// Function to download model using libcurl
bool download_model(const std::string& url, const std::string& output_path) {
    FILE* fp = fopen(output_path.c_str(), "wb");
    if (!fp) {
        std::cerr << "Failed to open file: " << output_path << std::endl;
        return false;
    }

    CURL* handle = curl_easy_init();
    if (!handle) {
        std::cerr << "Failed to create curl handle." << std::endl;
        fclose(fp);
        return false;
    }

    curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode res = curl_easy_perform(handle);
    if (res != CURLE_OK) {
        std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
        fclose(fp);
        curl_easy_cleanup(handle);
        return false;
    }

    long http_code = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code != 200) {
        std::cerr << "Download failed with HTTP status: " << http_code << std::endl;
        fclose(fp);
        curl_easy_cleanup(handle);
        return false;
    }

    fclose(fp);
    curl_easy_cleanup(handle);
    std::cout << "Downloaded model to " << output_path << std::endl;
    return true;
}

// Simple resize function (bilinear interpolation) for 28x28 to 224x224
std::vector<float> resize_28x28_to_224x224(const std::vector<float>& input) {
    std::vector<float> output(224 * 224, 0.0f);
    float scale = 224.0f / 28.0f;
    for (int y = 0; y < 224; ++y) {
        for (int x = 0; x < 224; ++x) {
            float src_x = x / scale;
            float src_y = y / scale;
            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, 27);
            int y1 = std::min(y0 + 1, 27);
            float dx = src_x - x0;
            float dy = src_y - y0;

            float val = (1 - dx) * (1 - dy) * input[y0 * 28 + x0] +
                        dx * (1 - dy) * input[y0 * 28 + x1] +
                        (1 - dx) * dy * input[y1 * 28 + x0] +
                        dx * dy * input[y1 * 28 + x1];
            output[y * 224 + x] = val;
        }
    }
    return output;
}


std::vector<float> read_mnist_image(const std::string& path, int idx) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    file.seekg(16 + idx * 28 * 28);
    std::vector<float> img(28 * 28);
    for (int i = 0; i < 28 * 28; ++i) {
        unsigned char pixel;
        file.read((char*)&pixel, 1);
        img[i] = pixel / 255.0f;
    }
    return img;
}

// Function to adapt final layer weights for MNIST (1000 -> 10 classes)
std::vector<float> adapt_final_layer(const float* weights, size_t in_features, size_t out_features) {
    std::vector<float> new_weights(in_features * 10, 0.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.01f);

    // Random projection from 1000 to 10 classes
    for (size_t i = 0; i < in_features; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            new_weights[i * 10 + j] = weights[i * 1000 + j % 1000] * dist(gen);
        }
    }
    return new_weights;
}

int main() {
    CURLcode res = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (res != CURLE_OK) {
        std::cerr << "Failed to initialize libcurl globally: " << curl_easy_strerror(res) << std::endl;
        return -1;
    }

    const std::string model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx";
    const std::string model_path = "resnet18.onnx";

    if (!download_model(model_url, model_path)) {
        std::cerr << "Model download failed." << std::endl;
        curl_global_cleanup();
        return -1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hello_world");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) {
        std::cerr << "Failed to get ONNX Runtime API." << std::endl;
        curl_global_cleanup();
        return -1;
    }

    Ort::Session session(nullptr);
    try {
        session = Ort::Session(env, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        curl_global_cleanup();
        return -1;
    }

    // Get output name dynamically
    size_t num_outputs = session.GetOutputCount();
    if (num_outputs < 1) {
        std::cerr << "Model has no outputs." << std::endl;
        curl_global_cleanup();
        return -1;
    }
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> output_names = {output_name.get()};
    std::cout << "Using output name: " << output_name.get() << std::endl;

    // Prepare dummy MNIST-like input (batch=1, channels=1, height=28, width=28)
    // std::vector<float> input_28x28(1 * 1 * 28 * 28, 0.5f); // Normalized to [0,1]
    // std::vector<float> input_224x224 = resize_28x28_to_224x224(input_28x28);


    std::vector<float> input_28x28 = read_mnist_image("t10k-images-idx3-ubyte", 0);
    std::vector<float> input_224x224 = resize_28x28_to_224x224(input_28x28);

    std::vector<float> input_data(1 * 3 * 224 * 224, 0.0f);
    for (size_t i = 0; i < 224 * 224; ++i) {
        input_data[i] = input_data[i + 224 * 224] = input_data[i + 2 * 224 * 224] = input_224x224[i];
    }

    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    std::vector<const char*> input_names = {"data"};

    // Run inference
    try {
        auto output_tensors = session.Run(Ort::RunOptions{}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        std::cout << "Inference output shape: ";
        for (auto dim : output_shape) std::cout << dim << " ";
        std::cout << std::endl;

        // Simulate MNIST adaptation
        std::vector<float> mnist_output = adapt_final_layer(output_data, output_shape[1], 10);
        int max_idx = 0;
        float max_val = mnist_output[0];
        for (int i = 1; i < 10; ++i) {
            if (mnist_output[i] > max_val) {
                max_val = mnist_output[i];
                max_idx = i;
            }
        }
        std::cout << "Predicted MNIST class: " << max_idx << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        curl_global_cleanup();
        return -1;
    }

    curl_global_cleanup();
    return 0;
}