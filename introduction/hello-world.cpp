#include <onnxruntime/onnxruntime_cxx_api.h>
#include "httplib.h"
#include <fstream>
#include <iostream>
#include <vector>

// Function to download model
bool download_model(const std::string& url, const std::string& output_path) {
    httplib::Client cli("https://github.com");
    auto path = url.substr(url.find("github.com") + 10);
    auto res = cli.Get(path.c_str());
    if (res && res->status == 200) {
        std::ofstream ofs(output_path, std::ios::binary);
        ofs.write(res->body.c_str(), res->body.size());
        ofs.close();
        std::cout << "Downloaded model to " << output_path << std::endl;
        return true;
    } else {
        std::cerr << "Failed to download model from " << url << ". Status: " << (res ? res->status : -1) << std::endl;
        return false;
    }
}

int main() {
    // Model URL and path
    const std::string model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx";
    const std::string model_path = "resnet50.onnx";

    // Download model
    if (!download_model(model_url, model_path)) {
        std::cerr << "Model download failed." << std::endl;
        return -1;
    }

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "simple_onnx_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(nullptr);
    try {
        session = Ort::Session(env, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return -1;
    }

    // Prepare dummy input (batch=1, channels=3, height=224, width=224)
    std::vector<float> input_data(1 * 3 * 224 * 224, 0.0f); // Zero-filled dummy image
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    // Define input and output names
    std::vector<const char*> input_names = {"data"};
    std::vector<const char*> output_names = {"resnetv24_dense0_fwd"};

    // Run inference
    try {
        auto output_tensors = session.Run(Ort::RunOptions{}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        // Print output shape and first value
        std::cout << "Inference output shape: ";
        for (auto dim : output_shape) std::cout << dim << " ";
        std::cout << std::endl;
        std::cout << "First output value: " << output_data[0] << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}