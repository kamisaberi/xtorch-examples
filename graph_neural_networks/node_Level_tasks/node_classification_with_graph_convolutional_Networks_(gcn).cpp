#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip> // For std::fixed, std::setprecision

// --- Configuration ---
const int64_t NUM_NODES = 10;          // Number of nodes in our synthetic graph
const int64_t NUM_FEATURES = 5;       // Number of input features per node
const int64_t NUM_CLASSES = 3;        // Number of output classes for node classification
const int64_t HIDDEN_DIM = 16;        // Hidden dimension for GCN layers
const int64_t NUM_EPOCHS = 200;
const double LEARNING_RATE = 0.01;
const double WEIGHT_DECAY = 5e-4;     // L2 regularization
const int64_t LOG_INTERVAL = 20;

// --- GCN Layer Implementation ---
struct GCNLayerImpl : torch::nn::Module {
    torch::nn::Linear linear{nullptr}; // Equivalent to H * W

    GCNLayerImpl(int64_t in_features, int64_t out_features) {
        linear = register_module("linear", torch::nn::Linear(in_features, out_features));
    }

    // adj_norm: D̃^(-1/2) Ã D̃^(-1/2)
    // features: H (node features from previous layer or input)
    torch::Tensor forward(const torch::Tensor& adj_norm, const torch::Tensor& features) {
        // Support = D̃^(-1/2) Ã D̃^(-1/2) H  (or ÃH for simpler GCN variants)
        // For sparse adj_norm, use torch::sparse::mm
        torch::Tensor support = torch::matmul(adj_norm, features); // [N, N] @ [N, in_features] -> [N, in_features]
        torch::Tensor output = linear(support); // [N, in_features] @ [in_features, out_features] (implicitly) -> [N, out_features]
        return output;
    }
};
TORCH_MODULE(GCNLayer);

// --- GCN Model (Stack of GCN Layers) ---
struct GCNModelImpl : torch::nn::Module {
    GCNLayer gcn1{nullptr};
    GCNLayer gcn2{nullptr};
    torch::nn::Dropout dropout{nullptr};

    GCNModelImpl(int64_t in_features, int64_t hidden_dim, int64_t num_classes, double dropout_prob = 0.5) {
        gcn1 = register_module("gcn1", GCNLayer(in_features, hidden_dim));
        gcn2 = register_module("gcn2", GCNLayer(hidden_dim, num_classes));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_prob));
    }

    // adj_norm: Normalized adjacency matrix
    // features: Input node features
    torch::Tensor forward(const torch::Tensor& adj_norm, torch::Tensor features) {
        features = gcn1(adj_norm, features);
        features = torch::relu(features);
        features = dropout(features);
        features = gcn2(adj_norm, features);
        // For CrossEntropyLoss, raw logits are expected. Softmax is included in the loss.
        // return torch::log_softmax(features, /*dim=*/1); // If using NLLLoss
        return features; // Raw logits for CrossEntropyLoss
    }
};
TORCH_MODULE(GCNModel);

// --- Graph Data and Preprocessing ---
// In a real scenario, load from Cora, Citeseer, etc.
// Here, we create a synthetic graph.
struct GraphData {
    torch::Tensor adj;        // Adjacency matrix (dense for this example)
    torch::Tensor features;   // Node features
    torch::Tensor labels;     // Node labels
    torch::Tensor train_mask; // Mask for training nodes
    torch::Tensor val_mask;   // Mask for validation nodes (optional)
    torch::Tensor test_mask;  // Mask for test nodes (optional)

    torch::Tensor adj_norm;   // Normalized adjacency matrix D̃^(-1/2) Ã D̃^(-1/2)

    GraphData(int60_t num_nodes, int64_t num_features, int64_t num_classes, torch::Device device) {
        // Create a simple synthetic adjacency matrix (e.g., a few connected components or random)
        adj = torch::zeros({num_nodes, num_nodes}, device);
        // Example: a line graph 0-1, 1-2, ..., (N-2)-(N-1)
        for (int64_t i = 0; i < num_nodes - 1; ++i) {
            adj[i][i+1] = 1;
            adj[i+1][i] = 1; // Symmetric
        }
        // Add some more random connections for density
        // adj.index_put_({torch::tensor({0}), torch::tensor({3})}, 1); adj.index_put_({torch::tensor({3}), torch::tensor({0})}, 1);
        // adj.index_put_({torch::tensor({2}), torch::tensor({5})}, 1); adj.index_put_({torch::tensor({5}), torch::tensor({2})}, 1);
        // adj.index_put_({torch::tensor({7}), torch::tensor({9})}, 1); adj.index_put_({torch::tensor({9}), torch::tensor({7})}, 1);
        adj.index_put_({0, 3}, 1); adj.index_put_({3, 0}, 1);
        adj.index_put_({2, 5}, 1); adj.index_put_({5, 2}, 1);
        adj.index_put_({7, 9}, 1); adj.index_put_({9, 7}, 1);


        // Random node features
        features = torch::randn({num_nodes, num_features}, device);

        // Random node labels
        labels = torch::randint(0, num_classes, {num_nodes}, torch::TensorOptions().dtype(torch::kLong).device(device));

        // Create masks (e.g., 60% train, 20% val, 20% test)
        train_mask = torch::zeros({num_nodes}, torch::TensorOptions().dtype(torch::kBool).device(device));
        val_mask = torch::zeros({num_nodes}, torch::TensorOptions().dtype(torch::kBool).device(device));
        // test_mask = torch::zeros({num_nodes}, torch::TensorOptions().dtype(torch::kBool).device(device));

        // For simplicity, let's mark first few nodes for training
        for(int64_t i=0; i<num_nodes * 0.6; ++i) train_mask[i] = true;
        for(int64_t i=num_nodes * 0.6; i<num_nodes * 0.8; ++i) val_mask[i] = true;
        // The rest could be test, but we'll only use train/val here.

        // Normalize adjacency matrix
        normalize_adjacency();
    }

    void normalize_adjacency() {
        torch::Device device = adj.device();
        // Ã = A + I
        torch::Tensor adj_tilde = adj + torch::eye(adj.size(0), device);
        // D̃_ii = Σ_j Ã_ij
        torch::Tensor degree_tilde = torch::sum(adj_tilde, /*dim=*/1);
        // D̃^(-1/2)
        // Add a small epsilon to degree_tilde_inv_sqrt to avoid division by zero for isolated nodes
        torch::Tensor degree_tilde_inv_sqrt = torch::pow(degree_tilde + 1e-12, -0.5);
        degree_tilde_inv_sqrt.masked_fill_(degree_tilde_inv_sqrt == std::numeric_limits<float>::infinity(), 0); // Handle infs

        // D̃^(-1/2) matrix (diagonal)
        torch::Tensor D_inv_sqrt_matrix = torch::diag(degree_tilde_inv_sqrt);

        // adj_norm = D̃^(-1/2) Ã D̃^(-1/2)
        // For dense matrices:
        adj_norm = torch::matmul(torch::matmul(D_inv_sqrt_matrix, adj_tilde), D_inv_sqrt_matrix);
        // For sparse matrices, this would involve sparse matrix multiplications.
    }
};


int main() {
    std::cout << "GCN for Node Classification (LibTorch C++)" << std::endl;
    torch::manual_seed(1);
    std::cout << std::fixed << std::setprecision(4);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    // --- Load/Create Graph Data ---
    GraphData graph(NUM_NODES, NUM_FEATURES, NUM_CLASSES, device);
    std::cout << "Synthetic graph created." << std::endl;
    // std::cout << "Normalized Adjacency Matrix:\n" << graph.adj_norm << std::endl;

    // --- Model ---
    GCNModel model(NUM_FEATURES, HIDDEN_DIM, NUM_CLASSES);
    model->to(device);
    std::cout << "GCN model created." << std::endl;

    // --- Optimizer ---
    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(LEARNING_RATE).weight_decay(WEIGHT_DECAY)
    );
    std::cout << "Optimizer created." << std::endl;

    // --- Loss Function ---
    // CrossEntropyLoss for multi-class classification
    torch::nn::CrossEntropyLoss criterion;
    std::cout << "Loss function (CrossEntropyLoss) created." << std::endl;

    // --- Training Loop ---
    std::cout << "\nStarting Training..." << std::endl;
    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        model->train(); // Set model to training mode
        optimizer.zero_grad();

        // Forward pass
        // Pass the entire graph (adj_norm and features) to the model
        torch::Tensor output_logits = model->forward(graph.adj_norm, graph.features);

        // Calculate loss only on training nodes
        // output_logits[graph.train_mask] selects rows corresponding to training nodes
        // graph.labels[graph.train_mask] selects labels for training nodes
        torch::Tensor loss = criterion(output_logits.index_select(0, graph.train_mask.nonzero().squeeze()),
                                       graph.labels.index_select(0, graph.train_mask.nonzero().squeeze()));

        loss.backward();
        optimizer.step();

        // Evaluation (on validation set, for example)
        if (epoch % LOG_INTERVAL == 0 || epoch == NUM_EPOCHS) {
            model->eval(); // Set model to evaluation mode
            torch::NoGradGuard no_grad; // Disable gradient calculations

            torch::Tensor val_output_logits = model->forward(graph.adj_norm, graph.features);
            torch::Tensor val_loss = criterion(val_output_logits.index_select(0, graph.val_mask.nonzero().squeeze()),
                                               graph.labels.index_select(0, graph.val_mask.nonzero().squeeze()));

            // Calculate accuracy
            torch::Tensor val_predictions = torch::argmax(val_output_logits.index_select(0, graph.val_mask.nonzero().squeeze()), /*dim=*/1);
            torch::Tensor val_correct = (val_predictions == graph.labels.index_select(0, graph.val_mask.nonzero().squeeze()));
            double val_accuracy = static_cast<double>(val_correct.sum().item<int64_t>()) / val_correct.size(0);

            std::cout << "Epoch: " << std::setw(3) << epoch << "/" << NUM_EPOCHS
                      << " | Train Loss: " << loss.item<double>()
                      << " | Val Loss: " << val_loss.item<double>()
                      << " | Val Accuracy: " << val_accuracy
                      << std::endl;
        }
    }
    std::cout << "Training finished." << std::endl;

    // --- Final Evaluation on Test Set (if available) ---
    // model->eval();
    // torch::NoGradGuard no_grad;
    // torch::Tensor test_output_logits = model->forward(graph.adj_norm, graph.features);
    // ... calculate test loss and accuracy ...

    // torch::save(model, "gcn_model.pt");
    return 0;
}