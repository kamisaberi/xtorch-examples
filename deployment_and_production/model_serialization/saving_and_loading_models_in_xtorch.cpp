#include <torch/torch.h>
#include <iostream>
#include <vector>

// Define a simple neural network
struct SimpleNet : torch::nn::Module {
    SimpleNet() {
        fc1 = register_module("fc1", torch::nn::Linear(10, 20));
        fc2 = register_module("fc2", torch::nn::Linear(20, 5));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
    // Set random seed for reproducibility
    torch::manual_seed(42);

    // --- Training Phase ---
    std::cout << "Starting training phase...\n";

    // Instantiate the model for training
    SimpleNet model;
    model.train(); // Set to training mode

    // Create a synthetic dataset: 100 samples, input dim 10, output dim 5
    int num_samples = 100;
    torch::Tensor inputs = torch::randn({num_samples, 10});
    torch::Tensor targets = torch::randn({num_samples, 5});

    // Define optimizer and loss function
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01) /* learning rate */);
    auto loss_fn = torch::nn::MSELoss();

    // Training loop
    int num_epochs = 100;
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        torch::Tensor outputs = model.forward(inputs);
        torch::Tensor loss = loss_fn(outputs, targets);

        // Backward pass and optimize
        loss.backward();
        optimizer.step();

        // Print loss every 10 epochs
        if (epoch % 10 == 0) {
            std::cout << "Epoch [" << epoch << "/" << num_epochs << "], Loss: " << loss.item<float>() << std::endl;
        }
    }

    // Save the model parameters
    try {
        torch::save(model, "model_params.pt");
        std::cout << "Trained model parameters saved as 'model_params.pt'\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error saving the model: " << e.what() << std::endl;
        return -1;
    }

    // --- Loading and Inference Phase ---
    std::cout << "\nStarting loading and inference phase...\n";

    // Instantiate a new model for loading
    SimpleNet loaded_model;

    // Load the saved model parameters
    try {
        torch::load(loaded_model, "model_params.pt");
        std::cout << "Model parameters loaded from 'model_params.pt'\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    // Set model to evaluation mode
    loaded_model.eval();

    // Create a sample input tensor for inference
    torch::Tensor input = torch::randn({1, 10});

    // Run inference
    try {
        torch::NoGradGuard no_grad; // Disable gradient computation
        torch::Tensor output = loaded_model.forward(input);
        std::cout << "Inference output: " << output << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}