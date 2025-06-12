#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm> // For std::shuffle

// --- Data Structures for a Molecule ---
struct MolecularGraph {
    torch::Tensor atom_features;  // [num_atoms, atom_feature_dim]
    torch::Tensor edge_index;     // [2, num_bonds], edge_index[0] = src, edge_index[1] = dst
    torch::Tensor edge_features;  // [num_bonds, bond_feature_dim]
    torch::Tensor target_property; // Scalar or [1]
};

// --- MPNN Layer ---
// Implements one step of message passing and update
struct MPNNLayerImpl : torch::nn::Module {
    torch::nn::Linear message_mlp{nullptr};
    torch::nn::Linear update_mlp{nullptr};
    int64_t _hidden_dim;

    MPNNLayerImpl(int64_t atom_feature_dim, int64_t bond_feature_dim, int64_t hidden_dim)
        : _hidden_dim(hidden_dim) {
        // Message MLP: takes [h_v, h_w, e_vw] -> message_dim (which is hidden_dim)
        // h_v and h_w have atom_feature_dim (or hidden_dim from previous layer)
        // e_vw has bond_feature_dim
        // Input to message_mlp: atom_feature_dim*2 + bond_feature_dim
        message_mlp = register_module("message_mlp",
                                      torch::nn::Linear(atom_feature_dim * 2 + bond_feature_dim, hidden_dim));

        // Update MLP: takes [h_v_old, aggregated_message] -> hidden_dim (new h_v)
        // h_v_old has atom_feature_dim, aggregated_message has hidden_dim
        update_mlp = register_module("update_mlp",
                                     torch::nn::Linear(atom_feature_dim + hidden_dim, hidden_dim));
    }

    // Overload for subsequent layers where input atom features are already hidden_dim
    MPNNLayerImpl(int64_t hidden_dim, int64_t bond_feature_dim) : _hidden_dim(hidden_dim) {
         message_mlp = register_module("message_mlp",
                                      torch::nn::Linear(hidden_dim * 2 + bond_feature_dim, hidden_dim));
         update_mlp = register_module("update_mlp",
                                     torch::nn::Linear(hidden_dim + hidden_dim, hidden_dim));
    }


    // h_atoms: [num_atoms, atom_feature_dim_current_layer]
    // edge_index: [2, num_bonds]
    // edge_features: [num_bonds, bond_feature_dim]
    // Returns: new_h_atoms [num_atoms, hidden_dim]
    torch::Tensor forward(torch::Tensor h_atoms, torch::Tensor edge_index, torch::Tensor edge_features) {
        int64_t num_atoms = h_atoms.size(0);
        int64_t num_bonds = edge_index.size(1);

        // --- Message Computation ---
        // For each bond (v, w), create input for message_mlp: [h_v, h_w, e_vw]
        torch::Tensor src_nodes = edge_index.index({0, "..."}); // Shape: [num_bonds]
        torch::Tensor dst_nodes = edge_index.index({1, "..."}); // Shape: [num_bonds]

        // Gather features for source and destination nodes of each bond
        torch::Tensor h_src = h_atoms.index_select(0, src_nodes); // [num_bonds, atom_feat_dim]
        torch::Tensor h_dst = h_atoms.index_select(0, dst_nodes); // [num_bonds, atom_feat_dim]

        // Concatenate: [h_src, h_dst, edge_features]
        torch::Tensor message_inputs = torch::cat({h_src, h_dst, edge_features}, /*dim=*/1);
        // message_inputs shape: [num_bonds, atom_feat_dim*2 + bond_feat_dim]

        torch::Tensor messages = message_mlp(message_inputs); // [num_bonds, hidden_dim]
        messages = torch::relu(messages);

        // --- Message Aggregation (Summing messages for each destination node) ---
        torch::Tensor aggregated_messages = torch::zeros({num_atoms, _hidden_dim}, h_atoms.options());
        // scatter_add_(dim, index, src)
        // index must be same size as src for dim other than `dim`.
        // Here, dst_nodes is [num_bonds], we need to expand it to match messages [num_bonds, hidden_dim] for scatter.
        torch::Tensor dst_nodes_expanded = dst_nodes.unsqueeze(1).expand_as(messages);
        aggregated_messages.scatter_add_(0, dst_nodes_expanded, messages);

        // --- Update Step ---
        // Input to update_mlp: [h_atoms_old, aggregated_messages]
        torch::Tensor update_inputs = torch::cat({h_atoms, aggregated_messages}, /*dim=*/1);
        // update_inputs shape: [num_atoms, atom_feat_dim + hidden_dim]

        torch::Tensor new_h_atoms = update_mlp(update_inputs);
        new_h_atoms = torch::relu(new_h_atoms); // Or another activation like Tanh

        return new_h_atoms;
    }
};
TORCH_MODULE(MPNNLayer);


// --- Full MPNN Model for Molecular Property Prediction ---
struct MoleculeMPNNImpl : torch::nn::Module {
    torch::nn::Linear atom_embed_in{nullptr}; // Optional: if initial atom features need projection
    torch::nn::Linear bond_embed_in{nullptr}; // Optional: if initial bond features need projection

    std::vector<MPNNLayer> mpnn_layers;
    torch::nn::Linear readout_mlp1{nullptr};
    torch::nn::Linear readout_mlp2{nullptr};
    int _num_mpnn_steps;

    MoleculeMPNNImpl(int64_t initial_atom_dim, int64_t initial_bond_dim,
                       int64_t hidden_dim, int64_t num_mpnn_steps,
                       int64_t readout_hidden_dim, int64_t output_dim = 1,
                       bool embed_initial = false) // Set true if initial features are e.g. one-hot indices
        : _num_mpnn_steps(num_mpnn_steps) {

        if (embed_initial) {
            // This assumes initial_atom_dim and initial_bond_dim are vocabulary sizes
            // For simplicity, we'll use Linear for projection, not Embedding layers
            atom_embed_in = register_module("atom_embed_in", torch::nn::Linear(initial_atom_dim, hidden_dim));
            bond_embed_in = register_module("bond_embed_in", torch::nn::Linear(initial_bond_dim, hidden_dim)); // Embed bonds to hidden_dim too
        } else {
            // If features are already dense, project them to hidden_dim if different
            // Or ensure they are already hidden_dim
            if (initial_atom_dim != hidden_dim) {
                 atom_embed_in = register_module("atom_embed_in", torch::nn::Linear(initial_atom_dim, hidden_dim));
            }
             if (initial_bond_dim != hidden_dim && bond_embed_in) { // Check if bond_embed_in is needed
                // If initial_bond_dim is used directly in MPNNLayer, it might be different from hidden_dim
                // The MPNNLayer is set up to handle this initially, but let's assume we want bond features also in hidden_dim for consistency
                bond_embed_in = register_module("bond_embed_in", torch::nn::Linear(initial_bond_dim, hidden_dim));
            }
        }


        // Create MPNN layers
        // First layer takes (projected) initial atom features and (projected) bond features
        // Subsequent layers take hidden_dim atom features and (projected) bond features
        int64_t current_atom_dim = (atom_embed_in) ? hidden_dim : initial_atom_dim;
        int64_t current_bond_dim = (bond_embed_in) ? hidden_dim : initial_bond_dim;

        mpnn_layers.push_back(MPNNLayer(current_atom_dim, current_bond_dim, hidden_dim));
        register_module("mpnn_layer_0", mpnn_layers.back());

        for (int i = 1; i < _num_mpnn_steps; ++i) {
            mpnn_layers.push_back(MPNNLayer(hidden_dim, current_bond_dim, hidden_dim)); // Subsequent layers use hidden_dim for atoms
            register_module("mpnn_layer_" + std::to_string(i), mpnn_layers.back());
        }

        // Readout MLP
        readout_mlp1 = register_module("readout_mlp1", torch::nn::Linear(hidden_dim, readout_hidden_dim));
        readout_mlp2 = register_module("readout_mlp2", torch::nn::Linear(readout_hidden_dim, output_dim)); // output_dim = 1 for single property
    }

    torch::Tensor forward(torch::Tensor atom_features, torch::Tensor edge_index, torch::Tensor edge_features) {
        torch::Tensor h_atoms = atom_features;
        torch::Tensor e_features = edge_features;

        if (atom_embed_in) {
            h_atoms = torch::relu(atom_embed_in(h_atoms));
        }
        if (bond_embed_in) {
            e_features = torch::relu(bond_embed_in(e_features));
        }

        // Message Passing
        for (int i = 0; i < _num_mpnn_steps; ++i) {
            h_atoms = mpnn_layers[i]->forward(h_atoms, edge_index, e_features);
        }

        // Readout (Global Sum Pooling)
        torch::Tensor graph_embedding = h_atoms.sum(/*dim=*/0); // Sum over all atoms: [hidden_dim]

        // Prediction
        torch::Tensor prediction = readout_mlp1(graph_embedding);
        prediction = torch::relu(prediction);
        prediction = readout_mlp2(prediction); // No activation if it's regression output

        return prediction.unsqueeze(0); // [1, output_dim] for consistency (batch_size=1)
    }
};
TORCH_MODULE(MoleculeMPNN);


// --- Synthetic Dataset Generation ---
MolecularGraph generate_random_molecule(int min_atoms, int max_atoms,
                                        int atom_feat_dim, int bond_feat_dim,
                                        torch::Device device) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib_atoms(min_atoms, max_atoms);

    int num_atoms = distrib_atoms(gen);
    torch::Tensor atom_feats = torch::randn({num_atoms, atom_feat_dim}, device);

    // Generate random bonds (edge_index and edge_features)
    // For simplicity, connect each atom to a few random other atoms
    std::vector<std::pair<int64_t, int64_t>> edges;
    std::vector<torch::Tensor> bond_feature_list;

    if (num_atoms > 1) { // Only add bonds if more than 1 atom
        std::uniform_int_distribution<> distrib_bonds(1, std::min(3, num_atoms -1)); // Each atom has 1 to 3 bonds
        for (int i = 0; i < num_atoms; ++i) {
            int num_bonds_for_atom = distrib_bonds(gen);
            for (int k=0; k < num_bonds_for_atom; ++k) {
                int neighbor = (i + distrib_atoms(gen)) % num_atoms; // Simple way to get a different atom
                if (neighbor == i) neighbor = (i + 1) % num_atoms; // Ensure not self-loop unless num_atoms=1
                if (neighbor == i && num_atoms == 1) continue; // Skip if only one atom and tries to bond to self

                // Avoid duplicate edges in one direction for now (simple graph)
                bool exists = false;
                for(const auto& edge : edges) {
                    if ((edge.first == i && edge.second == neighbor)) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    edges.push_back({i, neighbor});
                    // Also add reverse edge for undirected graph representation in MPNN
                    edges.push_back({neighbor, i});

                    torch::Tensor bf = torch::randn({1, bond_feat_dim}, device);
                    bond_feature_list.push_back(bf);
                    bond_feature_list.push_back(bf); // Same feature for reverse edge
                }
            }
        }
    }


    torch::Tensor edge_idx_tensor;
    torch::Tensor bond_feats_tensor;

    if (!edges.empty()) {
        std::vector<int64_t> src_nodes_vec, dst_nodes_vec;
        for(const auto& edge : edges) {
            src_nodes_vec.push_back(edge.first);
            dst_nodes_vec.push_back(edge.second);
        }
        torch::Tensor src_tensor = torch::tensor(src_nodes_vec, torch::kLong).to(device);
        torch::Tensor dst_tensor = torch::tensor(dst_nodes_vec, torch::kLong).to(device);
        edge_idx_tensor = torch::stack({src_tensor, dst_tensor}, 0);
        bond_feats_tensor = torch::cat(bond_feature_list, 0);
    } else {
        edge_idx_tensor = torch::empty({2,0}, torch::TensorOptions().dtype(torch::kLong).device(device));
        bond_feats_tensor = torch::empty({0, bond_feat_dim}, torch::TensorOptions().device(device));
    }


    // Dummy target property: sum of first feature of all atoms + avg first bond feature
    torch::Tensor target_prop = atom_feats.index({"...", 0}).sum();
    if (bond_feats_tensor.size(0) > 0) {
        target_prop += bond_feats_tensor.index({"...", 0}).mean();
    }
    target_prop = target_prop.unsqueeze(0).to(device); // [1]

    return {atom_feats, edge_idx_tensor, bond_feats_tensor, target_prop};
}

std::vector<MolecularGraph> create_synthetic_molecular_dataset(int num_molecules, int min_atoms, int max_atoms,
                                                               int atom_feat_dim, int bond_feat_dim,
                                                               torch::Device device) {
    std::vector<MolecularGraph> dataset;
    for (int i = 0; i < num_molecules; ++i) {
        dataset.push_back(generate_random_molecule(min_atoms, max_atoms, atom_feat_dim, bond_feat_dim, device));
    }
    return dataset;
}


int main() {
    torch::manual_seed(0);
    srand(0);

    // --- Hyperparameters ---
    int64_t initial_atom_dim = 10;   // Raw feature dim for atoms
    int64_t initial_bond_dim = 5;    // Raw feature dim for bonds
    int64_t hidden_dim = 32;         // Hidden dimension for MPNN layers and embeddings
    int64_t num_mpnn_steps = 3;      // Number of message passing iterations
    int64_t readout_hidden_dim = 64; // Hidden dim for the prediction MLP
    int64_t output_dim = 1;          // Predicting a single scalar property

    int num_molecules_train = 200;
    int num_molecules_test = 50;
    int min_atoms_per_molecule = 3;  // Min atoms for graph generation
    int max_atoms_per_molecule = 15; // Max atoms

    int num_epochs = 30;
    double learning_rate = 1e-3;
    bool embed_initial_features = false; // Set true if initial_atom_dim/initial_bond_dim are vocab sizes for embedding

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
    }

    // --- Create Datasets ---
    std::vector<MolecularGraph> train_dataset = create_synthetic_molecular_dataset(
        num_molecules_train, min_atoms_per_molecule, max_atoms_per_molecule,
        initial_atom_dim, initial_bond_dim, device);
    std::vector<MolecularGraph> test_dataset = create_synthetic_molecular_dataset(
        num_molecules_test, min_atoms_per_molecule, max_atoms_per_molecule,
        initial_atom_dim, initial_bond_dim, device);
    std::cout << "Datasets created. Train: " << train_dataset.size() << ", Test: " << test_dataset.size() << std::endl;


    // --- Instantiate Model and Optimizer ---
    MoleculeMPNN model(initial_atom_dim, initial_bond_dim, hidden_dim, num_mpnn_steps,
                       readout_hidden_dim, output_dim, embed_initial_features);
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    std::cout << "\nModel and optimizer created. Starting training..." << std::endl;

    // --- Training Loop ---
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        double epoch_loss = 0.0;

        std::random_device rd_shuffle;
        std::mt19937 g_shuffle(rd_shuffle());
        std::shuffle(train_dataset.begin(), train_dataset.end(), g_shuffle);

        for (const auto& mol_data : train_dataset) {
            // Skip if no atoms or no bonds for simplicity in this example
            // A robust implementation would handle these (e.g. graph with 1 atom, no bonds)
            if (mol_data.atom_features.size(0) == 0) continue;
            // The MPNNLayer's scatter_add handles empty edge_index correctly (adds nothing)

            optimizer.zero_grad();

            torch::Tensor prediction = model->forward(mol_data.atom_features,
                                                     mol_data.edge_index,
                                                     mol_data.edge_features);

            torch::Tensor loss = torch::mse_loss(prediction, mol_data.target_property);

            loss.backward();
            optimizer.step();
            epoch_loss += loss.item<double>();
        }

        double avg_epoch_loss = epoch_loss / train_dataset.size();
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                  << " | Avg Train MSE: " << avg_epoch_loss << std::endl;

        // --- Evaluation on Test Set ---
        if ((epoch + 1) % 5 == 0 || epoch == num_epochs - 1) {
            model->eval();
            torch::NoGradGuard no_grad;
            double test_loss = 0.0;
            for (const auto& mol_data : test_dataset) {
                if (mol_data.atom_features.size(0) == 0) continue;

                torch::Tensor prediction = model->forward(mol_data.atom_features,
                                                         mol_data.edge_index,
                                                         mol_data.edge_features);
                test_loss += torch::mse_loss(prediction, mol_data.target_property).item<double>();
            }
            double avg_test_loss = test_loss / test_dataset.size();
            std::cout << "  Test Results - Avg MSE: " << avg_test_loss << std::endl;
        }
    }

    std::cout << "\nTraining finished." << std::endl;
    return 0;
}