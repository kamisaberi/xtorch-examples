#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm> // For std::shuffle
#include <map>       // For adjacency list
// --- Configuration ---
const int64_t NUM_NODES = 20; // Number of nodes in our synthetic graph
const int64_t INPUT_FEATURE_DIM = 10; // Input feature dimension per node
const int64_t EMBEDDING_DIM = 32; // Output embedding dimension from GraphSAGE layers
const int64_t NUM_CLASSES = 3; // For a conceptual downstream classification task
const int NUM_SAGE_LAYERS = 2; // Number of GraphSAGE layers (hops)
const int NUM_SAMPLES_PER_HOP = 5; // Number of neighbors to sample at each hop
const int64_t BATCH_SIZE_NODES = 4; // Number of target nodes to process in a batch
const int64_t NUM_EPOCHS = 100;
const double LEARNING_RATE = 0.01;
const double DROPOUT_PROB = 0.5;
const int64_t LOG_INTERVAL = 10;


// --- Graph Representation (Adjacency List) ---
struct Graph
{
    int64_t num_nodes;
    torch::Tensor node_features; // [NUM_NODES, INPUT_FEATURE_DIM]
    std::vector<std::vector<int64_t>> adj_list; // Adjacency list
    torch::Tensor labels; // For conceptual supervised task [NUM_NODES]
    torch::Tensor train_mask; // Conceptual

    Graph(int64_t n_nodes, int64_t n_features, int64_t n_classes, torch::Device device)
        : num_nodes(n_nodes)
    {
        node_features = torch::randn({num_nodes, n_features}, device);
        adj_list.resize(num_nodes);

        // Create a simple synthetic graph (e.g., random connections)
        std::mt19937 rng(0); // For reproducibility
        for (int64_t i = 0; i < num_nodes; ++i)
        {
            int num_edges = std::uniform_int_distribution<int>(1, 4)(rng); // Each node has 1-4 edges
            for (int k = 0; k < num_edges; ++k)
            {
                int64_t neighbor = std::uniform_int_distribution<int64_t>(0, num_nodes - 1)(rng);
                if (neighbor != i)
                {
                    // Avoid duplicate edges for simplicity here
                    if (std::find(adj_list[i].begin(), adj_list[i].end(), neighbor) == adj_list[i].end())
                    {
                        adj_list[i].push_back(neighbor);
                    }
                    if (std::find(adj_list[neighbor].begin(), adj_list[neighbor].end(), i) == adj_list[neighbor].end())
                    {
                        adj_list[neighbor].push_back(i); // Make it undirected
                    }
                }
            }
            // Ensure no node is isolated (for sampling)
            if (adj_list[i].empty() && num_nodes > 1)
            {
                int64_t random_neighbor = (i + 1) % num_nodes;
                adj_list[i].push_back(random_neighbor);
                adj_list[random_neighbor].push_back(i);
            }
        }
        labels = torch::randint(0, n_classes, {num_nodes}, torch::TensorOptions().dtype(torch::kLong).device(device));
        train_mask = torch::zeros({num_nodes}, torch::kBool).to(device);
        for (int i = 0; i < num_nodes * 0.6; ++i) train_mask[i] = true; // Simple train mask
    }

    // Sample neighbors for a given set of nodes
    // Returns a vector where each element is a list of sampled neighbor indices for a node.
    std::vector<std::vector<int64_t>> sample_neighbors(const std::vector<int64_t>& nodes_batch, int num_samples) const
    {
        std::vector<std::vector<int64_t>> sampled_neighbors_batch;
        sampled_neighbors_batch.reserve(nodes_batch.size());
        std::mt19937 rng(std::random_device{}()); // Fresh RNG for each call for diverse samples

        for (int64_t node_idx : nodes_batch)
        {
            std::vector<int64_t> neighbors = adj_list[node_idx];
            std::vector<int64_t> sampled_neighbors_for_node;

            if (neighbors.empty())
            {
                // Handle nodes with no neighbors (or only self-loops if allowed)
                // Option 1: Sample itself (padding with self)
                // Option 2: Sample from global distribution (not done here)
                // Option 3: Pad with a placeholder (e.g., a zero vector, requires handling in aggregation)
                for (int i = 0; i < num_samples; ++i) sampled_neighbors_for_node.push_back(node_idx); // Sample self
            }
            else
            {
                if (neighbors.size() <= num_samples)
                {
                    // If fewer neighbors than num_samples, take all and pad by resampling with replacement
                    sampled_neighbors_for_node = neighbors;
                    while (sampled_neighbors_for_node.size() < num_samples)
                    {
                        sampled_neighbors_for_node.push_back(
                            neighbors[std::uniform_int_distribution<size_t>(0, neighbors.size() - 1)(rng)]);
                    }
                }
                else
                {
                    // If more neighbors, sample without replacement (or with, depending on SAGE variant)
                    std::shuffle(neighbors.begin(), neighbors.end(), rng); // Shuffle for random sampling
                    for (int i = 0; i < num_samples; ++i)
                    {
                        sampled_neighbors_for_node.push_back(neighbors[i]);
                    }
                }
            }
            sampled_neighbors_batch.push_back(sampled_neighbors_for_node);
        }
        return sampled_neighbors_batch;
    }
};


// --- GraphSAGE Layer ---
struct GraphSAGELayerImpl : torch::nn::Module
{
    torch::nn::Linear W_self{nullptr}; // Weight matrix for self features
    torch::nn::Linear W_neigh{nullptr}; // Weight matrix for aggregated neighbor features
    // For other aggregators like LSTM, more parameters would be needed.

    GraphSAGELayerImpl(int64_t in_dim, int64_t out_dim)
    {
        W_self = register_module("W_self", torch::nn::Linear(in_dim, out_dim));
        W_neigh = register_module("W_neigh", torch::nn::Linear(in_dim, out_dim));
        // Assuming mean aggregator, input dim for neigh is same
    }

    // self_features: [batch_size_nodes, in_dim] features of the target nodes
    // neighbor_features_aggregated: [batch_size_nodes, in_dim] MEAN aggregated features of their sampled neighbors
    torch::Tensor forward(torch::Tensor self_features, torch::Tensor neighbor_features_aggregated)
    {
        torch::Tensor self_transformed = W_self(self_features);
        torch::Tensor neigh_transformed = W_neigh(neighbor_features_aggregated);

        // Concatenation variant: torch::cat({self_transformed, neigh_transformed}, 1) then another Linear layer
        // Sum variant (simpler):
        torch::Tensor combined = self_transformed + neigh_transformed; // Element-wise sum
        return torch::relu(combined); // Or other activation
        // Note: Paper uses: CONCAT(W_self * h_v, W_neigh * h_N(v)) -> Linear -> ReLU
        // For simplicity, we use separate W_self, W_neigh and sum then ReLU.
        // A more canonical implementation would concat then have a single W_agg.
    }
};

TORCH_MODULE(GraphSAGELayer);


// --- GraphSAGE Model ---
// This model will compute embeddings for a given batch of target nodes
struct GraphSAGEModelImpl : torch::nn::Module
{
    std::vector<GraphSAGELayer> sage_layers;
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear classifier{nullptr}; // Optional: for a supervised task
    int num_layers;

    GraphSAGEModelImpl(int64_t input_dim, int64_t hidden_dim, int64_t output_dim_or_classes, int n_layers,
                       double dropout_p)
        : num_layers(n_layers)
    {
        // Input layer
        sage_layers.push_back(GraphSAGELayer(input_dim, hidden_dim));
        // Hidden layers
        for (int i = 1; i < num_layers; ++i)
        {
            sage_layers.push_back(GraphSAGELayer(hidden_dim, hidden_dim));
        }
        // Register layers
        for (size_t i = 0; i < sage_layers.size(); ++i)
        {
            register_module("sage_layer_" + std::to_string(i), sage_layers[i]);
        }

        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
        classifier = register_module("classifier", torch::nn::Linear(hidden_dim, output_dim_or_classes));
    }

    // This forward pass is for a batch of *target nodes*.
    // It requires the full graph structure (for sampling) and all node features.
    torch::Tensor forward(const std::vector<int64_t>& target_nodes_indices,
                          const Graph& full_graph,
                          int num_samples_per_hop)
    {
        // `nodes_at_hop[k]` stores the unique node indices needed at hop `k`.
        // `nodes_at_hop[0]` are the initial `target_nodes_indices`.
        std::vector<std::vector<int64_t>> nodes_to_fetch_per_hop(num_layers + 1);
        nodes_to_fetch_per_hop[0] = target_nodes_indices;

        // 1. Neighborhood Sampling (from outside to inside, i.e., from furthest hop to target nodes)
        // This collects all unique nodes whose features are needed.
        std::vector<int64_t> current_layer_nodes = target_nodes_indices;
        for (int k = 0; k < num_layers; ++k)
        {
            // For each SAGE layer (hop)
            // Sample neighbors for nodes in `current_layer_nodes`
            std::vector<std::vector<int64_t>> sampled_neighbor_lists =
                full_graph.sample_neighbors(current_layer_nodes, num_samples_per_hop);

            std::vector<int64_t> next_hop_nodes_flat; // All neighbors collected at this hop
            for (const auto& neighbors_for_node : sampled_neighbor_lists)
            {
                next_hop_nodes_flat.insert(next_hop_nodes_flat.end(), neighbors_for_node.begin(),
                                           neighbors_for_node.end());
            }
            // Add the nodes themselves (current_layer_nodes) to the list of nodes needed for the *next* hop's aggregation
            next_hop_nodes_flat.insert(next_hop_nodes_flat.end(), current_layer_nodes.begin(),
                                       current_layer_nodes.end());

            // Get unique nodes for the next hop (k+1 depth from target)
            std::sort(next_hop_nodes_flat.begin(), next_hop_nodes_flat.end());
            next_hop_nodes_flat.erase(std::unique(next_hop_nodes_flat.begin(), next_hop_nodes_flat.end()),
                                      next_hop_nodes_flat.end());
            nodes_to_fetch_per_hop[k + 1] = next_hop_nodes_flat;
            current_layer_nodes = next_hop_nodes_flat;
            // These are the nodes whose features we need to compute the *previous* layer's input
        }

        // 2. Feature Aggregation (from inside to outside, i.e., from raw features to final embeddings)
        // `h[k]` will store embeddings of `nodes_to_fetch_per_hop[k]` after SAGE layer `num_layers - 1 - k`
        // h_k_minus_1 is input, h_k is output for a SAGE layer
        torch::Tensor current_features; // Holds features for nodes in nodes_to_fetch_per_hop[num_layers - k]
        // Initially, for the furthest hop (k=num_layers), these are raw input features.

        // Get initial features for the furthest hop (nodes_to_fetch_per_hop[num_layers])
        // Efficiently fetch these features. For now, simple index_select.
        torch::Tensor furthest_hop_nodes_tensor = torch::tensor(nodes_to_fetch_per_hop[num_layers], torch::kLong).to(
            full_graph.node_features.device());
        current_features = full_graph.node_features.index_select(0, furthest_hop_nodes_tensor);

        // Map from global node index to index within current_features tensor
        auto create_local_idx_map = [&](const std::vector<int64_t>& global_indices)
        {
            std::map<int64_t, int64_t> local_map;
            for (size_t i = 0; i < global_indices.size(); ++i) local_map[global_indices[i]] = i;
            return local_map;
        };

        // Iterate from layer k = num_layers-1 down to 0 (innermost SAGE layer to outermost)
        for (int k_sage_layer_idx = 0; k_sage_layer_idx < num_layers; ++k_sage_layer_idx)
        {
            // Nodes whose embeddings we are computing in THIS SAGE layer pass:
            // These are `nodes_to_fetch_per_hop[num_layers - 1 - k_sage_layer_idx]`
            const std::vector<int64_t>& target_nodes_for_this_sage_pass = nodes_to_fetch_per_hop[num_layers - 1 -
                k_sage_layer_idx];

            // Features for these target nodes are derived from `current_features` (which holds embeddings for the *next deeper* hop)
            // And their neighbors' features also come from `current_features`.
            std::map<int64_t, int64_t> source_nodes_local_map = create_local_idx_map(
                nodes_to_fetch_per_hop[num_layers - k_sage_layer_idx]);

            std::vector<torch::Tensor> self_feature_slices;
            std::vector<torch::Tensor> aggregated_neighbor_feature_slices;

            for (int64_t target_node_global_idx : target_nodes_for_this_sage_pass)
            {
                // Get self feature from `current_features`
                self_feature_slices.push_back(current_features.index({source_nodes_local_map[target_node_global_idx]}));

                // Sample direct neighbors of `target_node_global_idx` (these are the "1-hop" neighbors for this SAGE layer pass)
                // Their features are already in `current_features` (as they were part of the deeper hop)
                std::vector<int64_t> neighbors_of_target = full_graph.adj_list[target_node_global_idx];
                // Real neighbors
                std::vector<int64_t> sampled_direct_neighbors; // Sampled from real neighbors

                // Simple sampling for direct neighbors (similar to graph.sample_neighbors but for this specific context)
                if (neighbors_of_target.empty())
                {
                    for (int s = 0; s < num_samples_per_hop; ++s) sampled_direct_neighbors.push_back(
                        target_node_global_idx); // Use self if no neighbors
                }
                else
                {
                    std::mt19937 rng_inner(std::random_device{}());
                    if (neighbors_of_target.size() <= num_samples_per_hop)
                    {
                        sampled_direct_neighbors = neighbors_of_target;
                        while (sampled_direct_neighbors.size() < num_samples_per_hop)
                        {
                            sampled_direct_neighbors.push_back(
                                neighbors_of_target[std::uniform_int_distribution<size_t>(
                                    0, neighbors_of_target.size() - 1)(rng_inner)]);
                        }
                    }
                    else
                    {
                        std::shuffle(neighbors_of_target.begin(), neighbors_of_target.end(), rng_inner);
                        for (int s = 0; s < num_samples_per_hop; ++s) sampled_direct_neighbors.push_back(
                            neighbors_of_target[s]);
                    }
                }

                std::vector<torch::Tensor> neighbor_feature_tensors;
                for (int64_t neighbor_global_idx : sampled_direct_neighbors)
                {
                    // Check if this neighbor was actually part of the source nodes for features (it should be)
                    if (source_nodes_local_map.count(neighbor_global_idx))
                    {
                        neighbor_feature_tensors.push_back(current_features.index({
                            source_nodes_local_map[neighbor_global_idx]
                        }));
                    }
                    else
                    {
                        // This case implies a neighbor was sampled that wasn't part of the required nodes from the deeper hop.
                        // This can happen if sampling logic doesn't perfectly align with nodes_to_fetch.
                        // For robustness, one might fetch its raw feature or use a zero vector.
                        // Here, we'll assume they are found or use self features as a fallback.
                        // std::cerr << "Warning: Sampled neighbor " << neighbor_global_idx << " not in source feature map. Using self." << std::endl;
                        neighbor_feature_tensors.push_back(current_features.index({
                            source_nodes_local_map[target_node_global_idx]
                        }));
                    }
                }
                torch::Tensor mean_aggregated_neighbors = torch::stack(neighbor_feature_tensors).mean(0);
                aggregated_neighbor_feature_slices.push_back(mean_aggregated_neighbors);
            }

            torch::Tensor batch_self_features = torch::stack(self_feature_slices);
            torch::Tensor batch_aggregated_neighbor_features = torch::stack(aggregated_neighbor_feature_slices);

            // Pass through SAGE layer k_sage_layer_idx
            current_features = sage_layers[k_sage_layer_idx]->forward(batch_self_features,
                                                                      batch_aggregated_neighbor_features);
            if (k_sage_layer_idx < num_layers - 1)
            {
                // Apply dropout to hidden layers
                current_features = dropout(current_features);
            }
        }

        // After all SAGE layers, `current_features` contains the embeddings for `target_nodes_indices`
        torch::Tensor final_embeddings = current_features;

        // Optional: Pass through a classifier for a supervised task
        torch::Tensor logits = classifier(final_embeddings);
        return logits; // Or return final_embeddings if only doing embedding generation
    }
};

TORCH_MODULE(GraphSAGEModel);


int main()
{
    std::cout << "GraphSAGE Node Embedding (Conceptual - LibTorch C++)" << std::endl;
    torch::manual_seed(1);
    std::cout << std::fixed << std::setprecision(4);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    Graph graph(NUM_NODES, INPUT_FEATURE_DIM, NUM_CLASSES, device);
    std::cout << "Synthetic graph created." << std::endl;

    GraphSAGEModel model(INPUT_FEATURE_DIM, EMBEDDING_DIM, NUM_CLASSES, NUM_SAGE_LAYERS, DROPOUT_PROB);
    model->to(device);
    std::cout << "GraphSAGE model created." << std::endl;

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    torch::nn::CrossEntropyLoss criterion;
    std::cout << "Optimizer and Loss created." << std::endl;

    // --- Training Loop (Conceptual Supervised Task) ---
    // In a real unsupervised GraphSAGE, you'd have a different loss (e.g., based on random walks)
    std::cout << "\nStarting Training (Conceptual Supervised Task)..." << std::endl;

    // Create a list of all node indices for batching
    std::vector<int64_t> all_node_indices(NUM_NODES);
    std::iota(all_node_indices.begin(), all_node_indices.end(), 0);


    for (int64_t epoch = 1; epoch <= NUM_EPOCHS; ++epoch)
    {
        model->train();
        double epoch_loss = 0.0;
        int num_batches = 0;

        // Shuffle training nodes for batching
        std::vector<int64_t> training_nodes_epoch;
        for (int64_t i = 0; i < graph.num_nodes; ++i)
        {
            if (graph.train_mask[i].item<bool>()) training_nodes_epoch.push_back(i);
        }
        std::shuffle(training_nodes_epoch.begin(), training_nodes_epoch.end(), std::mt19937(std::random_device{}()));


        for (size_t i = 0; i < training_nodes_epoch.size(); i += BATCH_SIZE_NODES)
        {
            optimizer.zero_grad();

            std::vector<int64_t> batch_node_indices;
            for (size_t j = i; j < std::min(i + BATCH_SIZE_NODES, training_nodes_epoch.size()); ++j)
            {
                batch_node_indices.push_back(training_nodes_epoch[j]);
            }
            if (batch_node_indices.empty()) continue;

            torch::Tensor output_logits = model->forward(batch_node_indices, graph, NUM_SAMPLES_PER_HOP);

            // Get labels for the batch nodes
            std::vector<long> batch_labels_vec;
            for (int64_t node_idx : batch_node_indices) batch_labels_vec.push_back(graph.labels[node_idx].item<long>());
            torch::Tensor batch_labels = torch::tensor(batch_labels_vec, torch::kLong).to(device);

            torch::Tensor loss = criterion(output_logits, batch_labels);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
            num_batches++;
        }

        if (epoch % LOG_INTERVAL == 0 || epoch == NUM_EPOCHS)
        {
            model->eval();
            torch::NoGradGuard no_grad;
            // Evaluate on a fixed batch (or all training nodes if small) for simplicity
            std::vector<int64_t> eval_batch_indices = training_nodes_epoch; // Use all training nodes for eval
            if (eval_batch_indices.size() > BATCH_SIZE_NODES * 2)
            {
                // Cap eval batch size
                eval_batch_indices.resize(BATCH_SIZE_NODES * 2);
            }
            if (!eval_batch_indices.empty())
            {
                torch::Tensor eval_logits = model->forward(eval_batch_indices, graph, NUM_SAMPLES_PER_HOP);
                std::vector<long> eval_labels_vec;
                for (int64_t node_idx : eval_batch_indices) eval_labels_vec.push_back(
                    graph.labels[node_idx].item<long>());
                torch::Tensor eval_labels = torch::tensor(eval_labels_vec, torch::kLong).to(device);

                torch::Tensor eval_predictions = torch::argmax(eval_logits, /*dim=*/1);
                double accuracy = (eval_predictions == eval_labels).sum().item<double>() / eval_labels.size(0);
                std::cout << "Epoch: " << std::setw(3) << epoch << "/" << NUM_EPOCHS
                    << " | Avg Train Loss: " << (epoch_loss / std::max(1, num_batches))
                    << " | Eval Accuracy (on some train nodes): " << accuracy
                    << std::endl;
            }
            else
            {
                std::cout << "Epoch: " << std::setw(3) << epoch << "/" << NUM_EPOCHS
                    << " | Avg Train Loss: " << (epoch_loss / std::max(1, num_batches))
                    << " | No nodes for evaluation."
                    << std::endl;
            }
        }
    }
    std::cout << "Training finished." << std::endl;

    // --- Generate Embeddings for all nodes (Example) ---
    model->eval();
    torch::NoGradGuard no_grad;
    std::cout << "\nGenerating final embeddings (conceptually)..." << std::endl;
    // For GraphSAGE, the "embedding" is often the output of the SAGE layers *before* the final classifier.
    // To get this, you might need to modify the model's forward or have a separate method.
    // For this example, `model->forward` returns logits.
    // If you wanted embeddings:
    // 1. Remove the classifier from the forward pass or create `model->embed(...)`
    // 2. The output of the last SAGE layer (after dropout) would be the embeddings.

    // Let's call forward for all nodes to get "final representations" (logits in this case)
    torch::Tensor all_nodes_output = model->forward(all_node_indices, graph, NUM_SAMPLES_PER_HOP);
    std::cout << "Output shape for all nodes (logits): " << all_nodes_output.sizes() << std::endl;
    // This `all_nodes_output` would be your embeddings if the classifier was removed.

    // torch::save(model, "graphsage_model.pt");
    return 0;
}
