#include <iostream>
#include <string>
#include <filesystem> // For fs::current_path, fs::exists
#include <vector>

// Include the utility header from the installed xTorch library
#include <xtorch/xtorch.h> // Assumes headers are installed in an 'xtorch' subdirectory

// === NEW: Include LibTorch headers for inference ===
#include <torch/script.h> // LibTorch's main header for TorchScript
#include <torch/torch.h>  // For torch::Tensor, torch::kCPU etc.
// Potentially for image loading/preprocessing if you add that:
// #include <opencv2/opencv.hpp> // If using OpenCV
// ====================================================

namespace fs = std::filesystem;

// Helper function to create a dummy image tensor
torch::Tensor create_dummy_input_tensor(int batch_size, int channels, int height, int width, torch::Device device) {
    return torch::randn({batch_size, channels, height, width}, device);
}

int main(int argc, char* argv[]) {
    std::cout << "--- External xTorch Full Example App ---" << std::endl;

    // --- Configuration ---
    std::string hf_model_to_convert = "microsoft/resnet-18";
    fs::path base_output_dir = fs::current_path() / "example_outputs";
    fs::create_directories(base_output_dir); // Ensure base output directory exists

    fs::path download_destination = base_output_dir / "downloaded_mnist_example.py";
    fs::path torchscript_output_path = base_output_dir / "converted_resnet18.pt";

    // LibTorch device selection
    torch::Device device = torch::kCPU; // Default to CPU
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using CUDA for LibTorch inference." << std::endl;
        device = torch::kCUDA;
    } else {
        std::cout << "CUDA not available. Using CPU for LibTorch inference." << std::endl;
    }


    // --- 1. Test File Download (using xTorch library) ---
    std::string file_url = "https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py";
    std::cout << "\n[Step 1] Attempting file download via xTorch::download_file..." << std::endl;
    if (xt::download_file(file_url, download_destination)) {
        std::cout << "  Download successful: " << download_destination << std::endl;
    } else {
        std::cout << "  Download failed." << std::endl;
    }

    // --- 2. Test Model Conversion (using xTorch library) ---
    std::cout << "\n[Step 2] Attempting model conversion via xTorch library..." << std::endl;
    std::cout << "  Model: " << hf_model_to_convert << std::endl;
    std::cout << "  Output: " << torchscript_output_path << std::endl;

    if (xt::convert_hf_model_to_torchscript_from_lib(hf_model_to_convert, torchscript_output_path)) {
        std::cout << "  Model conversion initiated by xTorch library. Check Python script output." << std::endl;
    } else {
        std::cout << "  Model conversion initiation failed via xTorch library." << std::endl;
        // Optionally, exit if conversion is critical for next steps
        // return 1;
    }

    // --- 3. Load Converted TorchScript Model and Perform Inference ---
    std::cout << "\n[Step 3] Loading converted TorchScript model and performing inference..." << std::endl;
    if (fs::exists(torchscript_output_path)) {
        torch::jit::script::Module model;
        try {
            // Load the TorchScript model
            std::cout << "  Loading TorchScript model from: " << torchscript_output_path << std::endl;
            model = torch::jit::load(torchscript_output_path.string(), device); // Load to selected device
            model.eval(); // Set to evaluation mode
            std::cout << "  TorchScript model loaded successfully." << std::endl;

            // Prepare a dummy input tensor for inference
            // Dimensions should match what the model expects (and what was used for tracing)
            int batch = 1;
            int channels = 3;
            int img_size = 224; // Common for ResNet-18 traced with default convert_hf_model.py

            torch::Tensor dummy_input = create_dummy_input_tensor(batch, channels, img_size, img_size, device);
            std::cout << "  Created dummy input tensor with shape: " << dummy_input.sizes()
                      << " on device: " << dummy_input.device() << std::endl;

            // Perform inference
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(dummy_input);

            std::cout << "  Performing inference..." << std::endl;
            at::Tensor output_tensor = model.forward(inputs).toTensor(); // The .toTensor() is important

            std::cout << "  Inference successful!" << std::endl;
            std::cout << "  Output tensor shape: " << output_tensor.sizes() << std::endl;
            std::cout << "  Output tensor device: " << output_tensor.device() << std::endl;
            // For a classification model, output_tensor usually contains logits
            // For ResNet-18 (1000 classes), shape would be [batch_size, 1000]
            // You can print some values, but it's raw logits.
            // std::cout << "  Output tensor (first 5 logits of first batch item): "
            //           << output_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5).slice(/*dim=*/0, /*start=*/0, /*end=*/1)
            //           << std::endl;

            // Get predicted class (argmax on logits)
            torch::Tensor predicted_probs = torch::softmax(output_tensor, /*dim=*/1);
            torch::Tensor predicted_classes = torch::argmax(predicted_probs, /*dim=*/1);
            std::cout << "  Predicted class index for first item in batch: " << predicted_classes[0].item<int64_t>() << std::endl;


        } catch (const c10::Error& e) {
            std::cerr << "  Error loading/running TorchScript model: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  An unexpected C++ error occurred: " << e.what() << std::endl;
        }
    } else {
        std::cout << "  Converted model file not found at " << torchscript_output_path
                  << ". Skipping inference." << std::endl;
    }

    // --- 4. Placeholder for Model Evaluation ---
    std::cout << "\n[Step 4] Model Evaluation Placeholder..." << std::endl;
    std::cout << "  To evaluate the model, you would typically:" << std::endl;
    std::cout << "  1. Load a validation dataset (images and labels)." << std::endl;
    std::cout << "  2. Preprocess images similar to how the model was trained/traced." << std::endl;
    std::cout << "  3. Iterate through the dataset, perform inference." << std::endl;
    std::cout << "  4. Compare predictions with true labels to calculate metrics (e.g., accuracy)." << std::endl;
    // This would involve a lot more code for data loading and processing.

    // --- 5. Optional: Show discovered paths (from xTorch library) ---
    std::cout << "\n[Step 5] xTorch internal paths (for debugging/info):" << std::endl;
    if (auto paths_opt = xt::get_internal_library_paths()) {
        const auto& paths = *paths_opt;
        std::cout << "  Library Location: " << paths.library_path << std::endl;
        // ... (print other paths if needed) ...
    } else {
        std::cout << "  Could not retrieve xTorch internal paths." << std::endl;
    }

    std::cout << "\n--- External xTorch Full Example App Finished ---" << std::endl;
    return 0;
}