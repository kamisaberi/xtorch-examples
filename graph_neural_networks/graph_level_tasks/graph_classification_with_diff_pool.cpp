#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random> // For graph generation
#include <algorithm> // For std::shuffle

// --- Helper: GraphData Struct ---
struct GraphData
{
    torch::Tensor node_features; // [N, F_in]
    torch::Tensor adj_matrix; // [N, N]
    torch::Tensor label; // [1] or scalar
};

// --- GCN Convolution Layer (Slightly enhanced) ---
struct GCNConvImpl : torch::nn::Module
{
    torch::nn::Linear fc{nullptr};
    bool _normalize;
    bool _bias;

    GCNConvImpl(int64_t in_features, int64_t out_features, bool normalize = true, bool bias = true)
        : _normalize(normalize), _bias(bias)
    {
        fc = register_module("fc", torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias)));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor adj)
    {
        // x: [N, in_features]
        // adj: [N, N] (adjacency matrix)

        torch::Tensor support = fc(x); // XW : [N, out_features]

        torch::Tensor norm_adj = adj;
        if (_normalize)
        {
            // A basic normalization: D^-1 * A (assumes A has self-loops or non-zero diagonal for D)
            // For a proper GCN, symmetric normalization (D^-0.5 * A * D^-0.5) is better.
            // Ensure adj has self-loops before this for degree calculation, or add identity to adj.
            // Here, we assume adj might already have self-loops from graph generation.
            torch::Tensor degree = adj.sum(/*dim=*/1, /*keepdim=*/true).clamp_min(1.0); // Avoid div by zero
            norm_adj = adj / degree; // Row-wise normalization by out-degree
        }

        torch::Tensor output = torch::matmul(norm_adj, support); // A_norm * XW : [N, out_features]
        return output;
    }
};

TORCH_MODULE(GCNConv);


// --- DiffPool Block ---
struct DiffPoolBlockImpl : torch::nn::Module
{
    GCNConv gnn_embed{nullptr};
    GCNConv gnn_assign{nullptr};
    int64_t num_clusters_target; // Target number of clusters

    DiffPoolBlockImpl(int64_t in_features, int64_t embed_features, int64_t target_clusters)
        : num_clusters_target(target_clusters)
    {
        gnn_embed = GCNConv(in_features, embed_features, /*normalize=*/true, /*bias=*/true);
        gnn_assign = GCNConv(in_features, target_clusters, /*normalize=*/true, /*bias=*/true);

        register_module("gnn_embed", gnn_embed);
        register_module("gnn_assign", gnn_assign);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor adj)
    {
        // x: [N, in_features]
        // adj: [N, N]
        int64_t N = x.size(0);

        // Determine actual number of clusters for this graph
        // If num_clusters_target is a ratio (e.g. 0.25), calculate:
        // int64_t current_num_clusters = std::max(1L, static_cast<int64_t>(N * pooling_ratio_));
        // For fixed target:
        int64_t current_num_clusters = num_clusters_target;
        if (N <= current_num_clusters)
        {
            // Not enough nodes to pool to target_clusters.
            // Option 1: Skip pooling (return x, adj, and zero losses)
            // Option 2: Pool to fewer clusters (e.g., N/2 or 1)
            // Option 3: For this example, we assume N > current_num_clusters from dataset generation.
            // If you hit this, your dataset might produce graphs smaller than num_clusters_target.
            std::cout << "Warning: Node count " << N << " is <= target clusters " << current_num_clusters
                << ". DiffPool might behave unexpectedly or error. Ensure graphs are large enough." << std::endl;
            // A simple bypass if needed (though gnn_assign output dim is fixed):
            // return {x, adj, torch::tensor(0.0, x.options()), torch::tensor(0.0, x.options())};
        }

        torch::Tensor z_embed = gnn_embed(x, adj);
        z_embed = torch::relu(z_embed);

        torch::Tensor s_logits = gnn_assign(x, adj); // [N, current_num_clusters]
        torch::Tensor s = torch::softmax(s_logits, /*dim=*/1); // Soft assignment matrix [N, current_num_clusters]

        torch::Tensor x_pooled = torch::matmul(s.transpose(0, 1), z_embed); // [current_num_clusters, embed_features]

        // A_pool = S^T * A * S
        torch::Tensor adj_s = torch::matmul(adj, s); // [N, current_num_clusters]
        torch::Tensor adj_pooled = torch::matmul(s.transpose(0, 1), adj_s);
        // [current_num_clusters, current_num_clusters]

        // Link prediction loss: L_lp = ||A - S*S^T||_F^2 (normalized)
        // This encourages nodes assigned to the same cluster to be connected.
        torch::Tensor s_st = torch::matmul(s, s.transpose(0, 1)); // [N, N]
        torch::Tensor link_loss = (adj - s_st).pow(2).mean(); // Mean squared error

        // Entropy loss: L_ent = - (1/N) * sum( S_ij * log(S_ij + eps) ) for each node's assignment
        // This encourages assignments to be confident (close to one-hot).
        torch::Tensor entropy_loss = (-s * torch::log(s + 1e-12)).sum(/*dim=*/1).mean();

        return {x_pooled, adj_pooled, link_loss, entropy_loss};
    }
};

TORCH_MODULE(DiffPoolBlock);


// --- Main DiffPool Network for Graph Classification ---
struct DiffPoolNetImpl : torch::nn::Module
{
    DiffPoolBlock diffpool1{nullptr};
    // GCNConv final_gcn{nullptr}; // Optional: GCN on the coarsest graph
    torch::nn::Linear classifier_fc1{nullptr};
    torch::nn::Linear classifier_fc2{nullptr};
    torch::nn::Dropout dropout{nullptr};

    double link_loss_coeff = 0.5; // Hyperparameter
    double entropy_loss_coeff = 0.1; // Hyperparameter

    DiffPoolNetImpl(int64_t input_node_features,
                    int64_t gcn_hidden_dim, // Hidden dim for GCNs inside DiffPool
                    int64_t num_clusters1, // Target clusters for the first pool layer
                    int64_t classifier_hidden_dim,
                    int64_t num_classes,
                    double dropout_rate = 0.5)
    {
        diffpool1 = DiffPoolBlock(input_node_features, gcn_hidden_dim, num_clusters1);
        register_module("diffpool1", diffpool1);

        // Example: Add a GCN layer on the pooled graph
        // final_gcn = GCNConv(gcn_hidden_dim, gcn_hidden_dim);
        // register_module("final_gcn", final_gcn);

        // Classifier input is features from pooled nodes after global pooling
        classifier_fc1 = torch::nn::Linear(gcn_hidden_dim, classifier_hidden_dim);
        classifier_fc2 = torch::nn::Linear(classifier_hidden_dim, num_classes);
        dropout = torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout_rate));

        register_module("classifier_fc1", classifier_fc1);
        register_module("dropout", dropout);
        register_module("classifier_fc2", classifier_fc2);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor adj)
    {
        auto [x_p1, adj_p1, ll1, el1] = diffpool1(x, adj);

        // Optional: Apply GCN on the pooled graph
        // x_p1 = final_gcn(x_p1, adj_p1);
        // x_p1 = torch::relu(x_p1);

        // Global mean pooling over the cluster nodes
        torch::Tensor graph_embedding = x_p1.mean(/*dim=*/0); // [gcn_hidden_dim]

        torch::Tensor logits = classifier_fc1(graph_embedding);
        logits = torch::relu(logits);
        logits = dropout(logits);
        logits = classifier_fc2(logits); // [num_classes]

        torch::Tensor total_aux_loss = link_loss_coeff * ll1 + entropy_loss_coeff * el1;

        return {logits.unsqueeze(0), total_aux_loss}; // unsqueeze for batch_size=1 compatibility
    }
};

TORCH_MODULE(DiffPoolNet);


// --- Synthetic Dataset Generation ---
GraphData generate_random_graph(int min_nodes, int max_nodes, int feature_dim, int class_id, torch::Device device)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib_nodes(min_nodes, max_nodes);

    int num_nodes = distrib_nodes(gen);
    torch::Tensor node_features = torch::randn({num_nodes, feature_dim}, device);

    // Create adjacency matrix
    torch::Tensor adj_matrix = torch::zeros({num_nodes, num_nodes}, device);
    double edge_prob = (class_id == 0) ? 0.2 : 0.4; // Class 0 sparser, Class 1 denser
    if (num_nodes <= 1) edge_prob = 0; // Avoid issues with single node graphs for density

    for (int i = 0; i < num_nodes; ++i)
    {
        for (int j = i + 1; j < num_nodes; ++j)
        {
            if (static_cast<double>(rand()) / RAND_MAX < edge_prob)
            {
                adj_matrix[i][j] = 1.0;
                adj_matrix[j][i] = 1.0; // Undirected
            }
        }
    }
    adj_matrix = adj_matrix + torch::eye(num_nodes, device); // Add self-loops

    torch::Tensor label = torch::tensor({class_id}, torch::TensorOptions().dtype(torch::kLong).device(device));

    return {node_features, adj_matrix, label};
}

std::vector<GraphData> create_synthetic_dataset(int num_graphs, int min_nodes, int max_nodes,
                                                int feature_dim, torch::Device device)
{
    std::vector<GraphData> dataset;
    for (int i = 0; i < num_graphs; ++i)
    {
        int class_id = i % 2; // Alternate classes
        dataset.push_back(generate_random_graph(min_nodes, max_nodes, feature_dim, class_id, device));
    }
    return dataset;
}


int main()
{
    torch::manual_seed(123); // For reproducibility
    srand(123); // For C random functions if used (like in generate_random_graph)

    // --- Hyperparameters ---
    int64_t input_node_features = 16;
    int64_t gcn_hidden_dim = 32; // Hidden dim for GCNs in DiffPool and for pooled features
    int64_t num_clusters1 = 8; // Target clusters for DiffPool. Must be < min_nodes in dataset.
    int64_t classifier_hidden_dim = 64;
    int64_t num_classes = 2;
    double dropout_rate = 0.3;

    int num_graphs_train = 100;
    int num_graphs_test = 20;
    int min_graph_nodes = num_clusters1 + 5; // Ensure graphs are poolable
    int max_graph_nodes = 30;

    int num_epochs = 20;
    double learning_rate = 1e-3;

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
    }

    // --- Create Datasets ---
    std::vector<GraphData> train_dataset = create_synthetic_dataset(num_graphs_train, min_graph_nodes, max_graph_nodes,
                                                                    input_node_features, device);
    std::vector<GraphData> test_dataset = create_synthetic_dataset(num_graphs_test, min_graph_nodes, max_graph_nodes,
                                                                   input_node_features, device);
    std::cout << "Datasets created. Train: " << train_dataset.size() << ", Test: " << test_dataset.size() << std::endl;
    std::cout << "Min nodes: " << min_graph_nodes << ", Max nodes: " << max_graph_nodes << ", Target clusters: " <<
        num_clusters1 << std::endl;


    // --- Instantiate Model and Optimizer ---
    DiffPoolNet model(input_node_features, gcn_hidden_dim, num_clusters1, classifier_hidden_dim, num_classes,
                      dropout_rate);
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    std::cout << "\nModel and optimizer created. Starting training..." << std::endl;

    // --- Training Loop ---
    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        model->train(); // Set model to training mode
        double epoch_loss = 0.0;
        int64_t correct_predictions = 0;
        int64_t total_predictions = 0;

        // Shuffle training data
        std::random_device rd_shuffle;
        std::mt19937 g_shuffle(rd_shuffle());
        std::shuffle(train_dataset.begin(), train_dataset.end(), g_shuffle);

        for (const auto& graph_data : train_dataset)
        {
            optimizer.zero_grad();

            auto [logits, aux_loss] = model->forward(graph_data.node_features, graph_data.adj_matrix);

            // Classification loss
            // NLLLoss expects log_softmax input. CrossEntropyLoss handles softmax internally.
            // Since logits are raw, use CrossEntropyLoss or apply log_softmax first for NLLLoss.
            torch::Tensor classification_loss = torch::cross_entropy_loss(logits, graph_data.label);

            torch::Tensor total_loss = classification_loss + aux_loss;

            total_loss.backward();
            optimizer.step();

            epoch_loss += total_loss.item<double>();

            // Calculate training accuracy for this batch (graph)
            torch::Tensor predicted_class = logits.argmax(1);
            correct_predictions += (predicted_class == graph_data.label).sum().item<int64_t>();
            total_predictions++;
        }

        double avg_epoch_loss = epoch_loss / train_dataset.size();
        double train_accuracy = static_cast<double>(correct_predictions) / total_predictions;

        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
            << " | Avg Loss: " << avg_epoch_loss
            << " | Train Acc: " << train_accuracy << std::endl;

        // --- Evaluation on Test Set (optional, at end of each epoch) ---
        if ((epoch + 1) % 5 == 0 || epoch == num_epochs - 1)
        {
            // Evaluate every 5 epochs or last epoch
            model->eval(); // Set model to evaluation mode
            torch::NoGradGuard no_grad; // Disable gradient calculations

            int64_t test_correct_predictions = 0;
            int64_t test_total_predictions = 0;
            double test_total_loss = 0.0;

            for (const auto& graph_data : test_dataset)
            {
                auto [logits, aux_loss] = model->forward(graph_data.node_features, graph_data.adj_matrix);
                torch::Tensor classification_loss = torch::cross_entropy_loss(logits, graph_data.label);
                test_total_loss += (classification_loss + aux_loss).item<double>();

                torch::Tensor predicted_class = logits.argmax(1);
                test_correct_predictions += (predicted_class == graph_data.label).sum().item<int64_t>();
                test_total_predictions++;
            }
            double avg_test_loss = test_total_loss / test_dataset.size();
            double test_accuracy = static_cast<double>(test_correct_predictions) / test_total_predictions;
            std::cout << "  Test Results - Avg Loss: " << avg_test_loss
                << " | Test Acc: " << test_accuracy << std::endl;
        }
    }

    std::cout << "\nTraining finished." << std::endl;

    // You could save the model here if needed
    // torch::save(model, "diffpool_model.pt");

    return 0;
}
