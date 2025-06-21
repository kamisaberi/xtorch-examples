#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <xtorch/xtorch.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace fs = std::filesystem;

// Helper function to create a dummy image tensor
torch::Tensor create_dummy_input_tensor(int batch_size, int channels, int height, int width, torch::Device device)
{
    return torch::randn({batch_size, channels, height, width}, device);
}

int main(int argc, char* argv[])
{
    std::string hf_model_to_convert = "microsoft/resnet-18";
    fs::path base_output_dir = fs::current_path() / "example_outputs";
    fs::create_directories(base_output_dir);
    fs::path torchscript_output_path = base_output_dir / "converted_resnet18.pt";

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available())
    {
        device = torch::kCUDA;
    }
    std::cout << device << std::endl;

    if (xt::convert_hf_model_to_torchscript_from_lib(hf_model_to_convert, torchscript_output_path, 1, 224, 3,
                                                     xt::VerboseType::EVERYTHING))
    {
        std::cout << "  Model conversion succeed" << std::endl;
    }
    else
    {
        std::cerr << "  Model conversion failed" << std::endl;
        return 1;
    }

    if (fs::exists(torchscript_output_path))
    {
        torch::jit::script::Module model;
        try
        {
            model = torch::jit::load(torchscript_output_path.string(), device); // Load to selected device
            model.eval(); // Set to evaluation mode

            int batch = 1;
            int channels = 3;
            int img_size = 224; // Common for ResNet-18 traced with default convert_hf_model.py

            torch::Tensor dummy_input = create_dummy_input_tensor(batch, channels, img_size, img_size, device);
            std::cout << "  Created dummy input tensor with shape: " << dummy_input.sizes() << " on device: " <<
                dummy_input.device() << std::endl;

            // Perform inference
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(dummy_input);

            at::Tensor output_tensor = model.forward(inputs).toTensor(); // The .toTensor() is important


            // Get predicted class (argmax on logits)
            torch::Tensor predicted_probs = torch::softmax(output_tensor, /*dim=*/1);
            torch::Tensor predicted_classes = torch::argmax(predicted_probs, /*dim=*/1);
            std::cout << "  Predicted class index for first item in batch: " << predicted_classes[0].item<int64_t>() <<
                std::endl;
        }
        catch (const c10::Error& e)
        {
            std::cerr << "  Error loading/running TorchScript model: " << e.what() << std::endl;
        } catch (const std::exception& e)
        {
            std::cerr << "  An unexpected C++ error occurred: " << e.what() << std::endl;
        }
    }
    else
    {
        std::cout << "  Converted model file not found at " << torchscript_output_path
            << ". Skipping inference." << std::endl;
    }

    return 0;
}
