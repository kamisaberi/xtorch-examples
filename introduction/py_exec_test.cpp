#include <iostream>
#include <string>
#include <filesystem> // For fs::current_path

// Include the new utility header from the installed xTorch library
#include <xtorch/xtorch.h> // Assumes headers are installed in an 'xtorch' subdirectory

namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    std::cout << "--- External xTorch Example App ---" << std::endl;

    // --- 1. Test File Download ---
    std::string file_url = "https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py"; // A small text file
    fs::path download_destination = fs::current_path() / "downloaded_mnist_example.py";

    std::cout << "\nAttempting to download a file using xTorch::download_file..." << std::endl;
    if (xt::download_file(file_url, download_destination))
    {
        std::cout << "Download test successful! File saved to: " << download_destination << std::endl;
    }
    else
    {
        std::cout << "Download test failed." << std::endl;
    }

    // --- 2. Test Model Conversion ---
    std::string hf_model = "microsoft/resnet-18"; // Use a small, common model
    // Ensure the output directory exists or the python script can create it
    fs::path model_output_dir = fs::current_path() / "converted_models";
    fs::create_directories(model_output_dir); // Ensure output directory exists
    fs::path torchscript_output = model_output_dir / "resnet18_from_external_app.pt";

    std::cout << "\nAttempting to convert Hugging Face model using xTorch library..." << std::endl;
    std::cout << "Model: " << hf_model << std::endl;
    std::cout << "Output: " << torchscript_output << std::endl;

    if (xt::convert_hf_model_to_torchscript_from_lib(hf_model, torchscript_output))
    {
        std::cout << "Model conversion initiated successfully via xTorch library!" << std::endl;
        std::cout << "Check console output from Python script for detailed status." << std::endl;
        if (fs::exists(torchscript_output))
        {
            std::cout << "Converted model file found at: " << torchscript_output << std::endl;
        }
        else
        {
            std::cout << "Converted model file NOT found. Python script might have had issues." << std::endl;
        }
    }
    else
    {
        std::cout << "Model conversion initiation failed via xTorch library." << std::endl;
    }

    // --- 3. Optional: Show discovered paths ---
    std::cout << "\nAttempting to get xTorch internal paths (for debugging/info):" << std::endl;
    if (auto paths_opt = xtorch::get_internal_library_paths())
    {
        const auto& paths = *paths_opt;
        std::cout << "  xTorch Library Location: " << paths.library_path << std::endl;
        std::cout << "  xTorch Install Prefix:   " << paths.install_prefix << std::endl;
        std::cout << "  xTorch Share Directory:  " << paths.share_dir << std::endl;
        std::cout << "  xTorch Venv Directory:   " << paths.venv_dir << std::endl;
        std::cout << "  xTorch Python Exec:      " << paths.python_executable << std::endl;
        std::cout << "  xTorch Conversion Script:" << paths.conversion_script << std::endl;
    }
    else
    {
        std::cout << "  Could not retrieve xTorch internal paths." << std::endl;
    }


    std::cout << "\n--- External Example App Finished ---" << std::endl;
    return 0;
}
