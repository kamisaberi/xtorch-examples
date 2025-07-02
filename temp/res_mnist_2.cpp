#include <onnxruntime/onnxruntime_cxx_api.h>
#include <curl/curl.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm> // For std::min
#include <filesystem> // For checking file existence (C++17)

// --- Constants ---
const std::string MODEL_URL =
    "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx";
const std::string MODEL_FILENAME = "resnet18-v2-7.onnx";
// IMPORTANT: Download t10k-images-idx3-ubyte from http://yann.lecun.com/exdb/mnist/
// and place it in the same directory as the executable, or provide the correct path.
const std::string MNIST_IMAGE_DATA_PATH = "t10k-images-idx3-ubyte";
const int MNIST_IMG_DIM = 28;
const int MODEL_INPUT_IMG_DIM = 224;
const int MNIST_HEADER_SIZE = 16; // Offset in bytes to the first image in MNIST image file
const int NUM_MNIST_CLASSES = 10;
const int RESNET_OUTPUT_FEATURES = 1000; // ResNet18 typically outputs 1000 classes for ImageNet
const char* ONNX_INPUT_NAME = "data"; // Common input name for ResNet models

// --- libcurl Helper ---

// RAII wrapper for curl_global_init/cleanup
struct CurlGlobalInitializer
{
    CurlGlobalInitializer()
    {
        CURLcode res = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (res != CURLE_OK)
        {
            std::cerr << "Failed to initialize libcurl globally: " << curl_easy_strerror(res) << std::endl;
            // Optionally throw an exception or exit
            exit(-1);
        }
    }

    ~CurlGlobalInitializer()
    {
        curl_global_cleanup();
    }
};

// Callback function for libcurl to write data to file
size_t write_data_to_file(void* ptr, size_t size, size_t nmemb, void* stream)
{
    size_t written = fwrite(ptr, size, nmemb, (FILE*)stream);
    return written;
}

// Function to download model using libcurl
bool download_model_if_needed(const std::string& url, const std::string& output_path)
{
    // Check if file already exists
    if (std::filesystem::exists(output_path))
    {
        std::cout << "Model " << output_path << " already exists. Skipping download." << std::endl;
        return true;
    }

    std::cout << "Downloading model from " << url << " to " << output_path << "..." << std::endl;
    FILE* fp = fopen(output_path.c_str(), "wb");
    if (!fp)
    {
        std::cerr << "Failed to open file for writing: " << output_path << std::endl;
        return false;
    }

    CURL* curl_handle = curl_easy_init();
    if (!curl_handle)
    {
        std::cerr << "Failed to create curl handle." << std::endl;
        fclose(fp);
        return false;
    }

    curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, write_data_to_file);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects
    curl_easy_setopt(curl_handle, CURLOPT_FAILONERROR, 1L); // Fail on HTTP errors >= 400

    CURLcode res = curl_easy_perform(curl_handle);
    bool success = true;

    if (res != CURLE_OK)
    {
        std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
        success = false;
    }
    else
    {
        long http_code = 0;
        curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &http_code);
        // This check is somewhat redundant due to CURLOPT_FAILONERROR, but good for verbosity
        if (http_code != 200)
        {
            std::cerr << "Download failed with HTTP status: " << http_code << std::endl;
            success = false;
        }
    }

    fclose(fp);
    curl_easy_cleanup(curl_handle);

    if (success)
    {
        std::cout << "Successfully downloaded model to " << output_path << std::endl;
    }
    else
    {
        std::remove(output_path.c_str()); // Clean up partially downloaded file
    }
    return success;
}

// --- Image Processing ---

// Simple resize function (bilinear interpolation)
std::vector<float> resize_image_bilinear(const std::vector<float>& input_img,
                                         int input_h, int input_w,
                                         int output_h, int output_w)
{
    std::vector<float> output_img(output_h * output_w);
    const float scale_x = static_cast<float>(input_w) / output_w;
    const float scale_y = static_cast<float>(input_h) / output_h;

    for (int y_out = 0; y_out < output_h; ++y_out)
    {
        for (int x_out = 0; x_out < output_w; ++x_out)
        {
            float src_x = (x_out + 0.5f) * scale_x - 0.5f;
            float src_y = (y_out + 0.5f) * scale_y - 0.5f;

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, input_w - 1);
            int y1 = std::min(y0 + 1, input_h - 1);

            // Ensure x0, y0 are within bounds (can happen for x_out/y_out = 0 if src_x/src_y < 0)
            x0 = std::max(0, x0);
            y0 = std::max(0, y0);

            float dx = src_x - x0;
            float dy = src_y - y0;

            float val = (1.0f - dx) * (1.0f - dy) * input_img[y0 * input_w + x0] +
                dx * (1.0f - dy) * input_img[y0 * input_w + x1] +
                (1.0f - dx) * dy * input_img[y1 * input_w + x0] +
                dx * dy * input_img[y1 * input_w + x1];
            output_img[y_out * output_w + x_out] = val;
        }
    }
    return output_img;
}

// Reads a single MNIST image from the IDX3-UBYTE file
std::vector<float> read_mnist_image(const std::string& path, int image_index)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open MNIST image file: " << path << std::endl;
        std::cerr << "Please download t10k-images-idx3-ubyte from http://yann.lecun.com/exdb/mnist/" << std::endl;
        return {};
    }

    // Skip header and previous images
    file.seekg(MNIST_HEADER_SIZE + image_index * MNIST_IMG_DIM * MNIST_IMG_DIM);
    if (file.fail() || file.eof())
    {
        std::cerr << "Error: Could not seek to image index " << image_index << " in " << path << std::endl;
        return {};
    }

    std::vector<float> img_data(MNIST_IMG_DIM * MNIST_IMG_DIM);
    for (int i = 0; i < MNIST_IMG_DIM * MNIST_IMG_DIM; ++i)
    {
        unsigned char pixel;
        file.read(reinterpret_cast<char*>(&pixel), 1);
        if (file.fail())
        {
            std::cerr << "Error: Failed to read pixel data for image " << image_index << std::endl;
            return {};
        }
        img_data[i] = static_cast<float>(pixel) / 255.0f; // Normalize to [0, 1]
    }
    return img_data;
}

// --- Model Adaptation ---

// Adapts final layer weights from ResNet's output features to MNIST classes
// This is a very naive random projection and not a proper transfer learning fine-tuning.
// It's for demonstration purposes to get *some* output in the shape of 10 classes.
std::vector<float> adapt_resnet_output_to_mnist(const float* resnet_output_data,
                                                size_t resnet_features, // Should be RESNET_OUTPUT_FEATURES
                                                size_t mnist_classes)
{
    // Should be NUM_MNIST_CLASSES
    if (resnet_features != RESNET_OUTPUT_FEATURES || mnist_classes != NUM_MNIST_CLASSES)
    {
        std::cerr << "Warning: Mismatch in feature/class counts for adaptation." << std::endl;
    }

    std::vector<float> mnist_adapted_output(mnist_classes, 0.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    // Small weights for random projection to avoid large sum
    std::normal_distribution<float> dist(0.0f, 0.01f);

    // Simple projection: for each MNIST class, sum a random subset of ResNet outputs
    // This is arbitrary; a real scenario would involve retraining the last layer.
    for (size_t i = 0; i < mnist_classes; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < resnet_features; ++j)
        {
            // A slightly more structured (but still naive) projection:
            // Each MNIST class j "listens" to a block of resnet_features.
            // This is still arbitrary. A better naive projection might be
            // to just take the first `mnist_classes` outputs from resnet_output_data
            // or average blocks of `resnet_features / mnist_classes`.
            // The original code did `resnet_output_data[j % resnet_features] * dist(gen)`
            // which doesn't make sense if resnet_output_data is a flat vector of one prediction.
            // It should be `resnet_output_data[j] * dist(gen)`.
            // Let's use a similar random weighting as the original:
            if (j % (resnet_features / mnist_classes + 1) == 0) // just to make it sparse
                sum += resnet_output_data[j] * dist(gen);
        }
        mnist_adapted_output[i] = sum;
        // The original effectively summed up all features `resnet_features` times with different random weights for each of the 10 classes.
        // Let's replicate that structure but on the actual output tensor
    }

    // Replicating the original's random projection more closely:
    // The original code had weights `new_weights[in_features * 10 + j] = weights[in_features * 1000 + j % 1000] * dist(gen);`
    // This implies the `weights` parameter was a weight matrix, not an activation vector.
    // If `resnet_output_data` is the activation vector (shape [1, 1000]), then we are effectively creating
    // a new random weight matrix [1000, 10] and multiplying: Output_mnist = Activation_resnet * W_random
    mnist_adapted_output.assign(mnist_classes, 0.0f); // Reset
    for (size_t mnist_idx = 0; mnist_idx < mnist_classes; ++mnist_idx)
    {
        for (size_t resnet_idx = 0; resnet_idx < resnet_features; ++resnet_idx)
        {
            // Each MNIST logit is a weighted sum of all ResNet output features
            mnist_adapted_output[mnist_idx] += resnet_output_data[resnet_idx] * dist(gen);
        }
    }
    return mnist_adapted_output;
}


// --- Main ---
int main()
{
    CurlGlobalInitializer curl_init; // Handles curl_global_init/cleanup

    // --- 1. Download Model ---
    if (!download_model_if_needed(MODEL_URL, MODEL_FILENAME))
    {
        std::cerr << "Failed to obtain the ONNX model." << std::endl;
        return -1;
    }

    // --- 2. Initialize ONNX Runtime ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mnist_resnet_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // Consider enabling optimizations:
    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(nullptr);
    try
    {
#ifdef _WIN32
            std::wstring model_path_wstr = std::filesystem::path(MODEL_FILENAME).wstring();
            session = Ort::Session(env, model_path_wstr.c_str(), session_options);
#else
        session = Ort::Session(env, MODEL_FILENAME.c_str(), session_options);
#endif
        std::cout << "ONNX Runtime session created successfully for model: " << MODEL_FILENAME << std::endl;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "Failed to create ONNX Runtime session: " << e.what() << std::endl;
        return -1;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // --- 3. Get Model Input/Output Details ---
    if (session.GetInputCount() == 0)
    {
        std::cerr << "Error: Model has no inputs." << std::endl;
        return -1;
    }
    // Assuming the first input is the image data
    Ort::AllocatedStringPtr input_name_alloc = session.GetInputNameAllocated(0, allocator);
    std::string actual_input_name = input_name_alloc.get();
    if (actual_input_name != ONNX_INPUT_NAME)
    {
        std::cout << "Warning: Model input name is '" << actual_input_name
            << "', expected '" << ONNX_INPUT_NAME << "'. Using actual name." << std::endl;
    }
    std::vector<const char*> input_node_names = {actual_input_name.c_str()};


    if (session.GetOutputCount() == 0)
    {
        std::cerr << "Error: Model has no outputs." << std::endl;
        return -1;
    }
    Ort::AllocatedStringPtr output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> output_node_names = {output_name_alloc.get()};
    std::cout << "Using model input name: " << input_node_names[0] << std::endl;
    std::cout << "Using model output name: " << output_node_names[0] << std::endl;


    // --- 4. Prepare Input Data ---
    int image_to_test_idx = 0; // Use the first image from the MNIST test set
    std::vector<float> mnist_img_28x28 = read_mnist_image(MNIST_IMAGE_DATA_PATH, image_to_test_idx);
    if (mnist_img_28x28.empty())
    {
        std::cerr << "Failed to read MNIST image." << std::endl;
        return -1;
    }

    std::vector<float> resized_img_224x224 = resize_image_bilinear(mnist_img_28x28,
                                                                   MNIST_IMG_DIM, MNIST_IMG_DIM,
                                                                   MODEL_INPUT_IMG_DIM, MODEL_INPUT_IMG_DIM);

    // ResNet expects NCHW format: [batch_size, channels, height, width]
    // For MNIST, it's grayscale, so we replicate the single channel to 3 channels (R=G=B)
    std::vector<float> model_input_data(1 * 3 * MODEL_INPUT_IMG_DIM * MODEL_INPUT_IMG_DIM);
    const int H = MODEL_INPUT_IMG_DIM;
    const int W = MODEL_INPUT_IMG_DIM;
    for (int i = 0; i < H * W; ++i)
    {
        model_input_data[i] = resized_img_224x224[i]; // R channel
        model_input_data[i + H * W] = resized_img_224x224[i]; // G channel
        model_input_data[i + 2 * H * W] = resized_img_224x224[i]; // B channel
    }

    std::vector<int64_t> input_tensor_shape = {1, 3, MODEL_INPUT_IMG_DIM, MODEL_INPUT_IMG_DIM};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              model_input_data.data(), model_input_data.size(),
                                                              input_tensor_shape.data(), input_tensor_shape.size());

    // --- 5. Run Inference ---
    std::cout << "Running inference..." << std::endl;
    try
    {
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_node_names.data(), &input_tensor, 1,
                                          output_node_names.data(), 1);

        if (output_tensors.empty() || !output_tensors[0].IsTensor())
        {
            std::cerr << "Error: Inference did not return a valid tensor." << std::endl;
            return -1;
        }

        float* resnet_output_ptr = output_tensors[0].GetTensorMutableData<float>();
        Ort::TensorTypeAndShapeInfo output_shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> resnet_output_shape = output_shape_info.GetShape();

        std::cout << "ResNet output shape: ";
        for (int64_t dim : resnet_output_shape) std::cout << dim << " ";
        std::cout << std::endl;

        if (resnet_output_shape.size() != 2 || resnet_output_shape[0] != 1 || resnet_output_shape[1] !=
            RESNET_OUTPUT_FEATURES)
        {
            std::cerr << "Error: Unexpected ResNet output shape. Expected [1, " << RESNET_OUTPUT_FEATURES << "]." <<
                std::endl;
            return -1;
        }

        // --- 6. Post-process: Adapt to MNIST classes and find prediction ---
        std::vector<float> mnist_predictions = adapt_resnet_output_to_mnist(resnet_output_ptr,
                                                                            resnet_output_shape[1],
                                                                            NUM_MNIST_CLASSES);

        int predicted_class = 0;
        float max_score = mnist_predictions[0];
        std::cout << "MNIST class scores (adapted):" << std::endl;
        for (int i = 0; i < NUM_MNIST_CLASSES; ++i)
        {
            std::cout << "  Class " << i << ": " << mnist_predictions[i] << std::endl;
            if (mnist_predictions[i] > max_score)
            {
                max_score = mnist_predictions[i];
                predicted_class = i;
            }
        }
        std::cout << "Predicted MNIST class for image " << image_to_test_idx << ": " << predicted_class << std::endl;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Execution finished successfully." << std::endl;
    return 0;
}
