#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>

using namespace std::chrono;

// ========== DEBUG MODE ==========
// Set to true to print intermediate values after each layer
constexpr bool DEBUG_MODE = true;

/**
 * @brief MNIST LeNet-5 Network Architecture (Cleartext - No Encryption)
 *
 * Network: Conv1 -> ReLU -> AvgPool1 -> Conv2 -> ReLU -> AvgPool2 -> Dense1 -> ReLU -> Dense2 -> ReLU -> Dense3
 * - Input: 28x28 MNIST image (1 channel)
 * - Conv1: 5x5 kernel, 6 output channels, stride=1, padding=0 -> 24x24x6
 * - ReLU: Standard activation max(0, x)
 * - AvgPool1: 2x2 kernel, stride=2 -> 12x12x6
 * - Conv2: 5x5 kernel, 16 output channels, stride=1, padding=0 -> 8x8x16
 * - ReLU: Standard activation
 * - AvgPool2: 2x2 kernel, stride=2 -> 4x4x16 = 256
 * - Dense1: 256 -> 120 neurons
 * - ReLU: Standard activation
 * - Dense2: 120 -> 84 neurons
 * - ReLU: Standard activation
 * - Dense3: 84 -> 10 neurons (output classes)
 */

/**
 * @brief Print min/max bounds of a vector
 */
void PrintBounds(const std::vector<double>& vec, const std::string& name) {
    auto minVal = *std::min_element(vec.begin(), vec.end());
    auto maxVal = *std::max_element(vec.begin(), vec.end());
    std::cout << "  " << name << " bounds: [" << std::fixed << std::setprecision(6)
              << minVal << ", " << maxVal << "]" << std::endl;
}

/**
 * @brief Print first N values for debugging
 */
void PrintDebugValues(const std::vector<double>& vec, const std::string& name, size_t numValues = 10) {
    if (!DEBUG_MODE) return;

    std::cout << "  [DEBUG] " << name << " (first " << std::min(numValues, vec.size()) << " values):" << std::endl;
    std::cout << "    ";
    for (size_t i = 0; i < std::min(numValues, vec.size()); i++) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < std::min(numValues, vec.size()) - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

/**
 * @brief Print first N values of conv kernel for debugging
 */
void PrintKernelDebug(const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel, const std::string& name, size_t numValues = 10) {
    if (!DEBUG_MODE) return;

    std::cout << "  [DEBUG WEIGHTS] " << name << " first channel [0][0] (first " << numValues << " values, flattened):" << std::endl;
    std::cout << "    ";
    size_t count = 0;
    for (size_t i = 0; i < kernel[0][0].size() && count < numValues; i++) {
        for (size_t j = 0; j < kernel[0][0][i].size() && count < numValues; j++) {
            std::cout << std::fixed << std::setprecision(4) << kernel[0][0][i][j];
            if (count < numValues - 1) std::cout << ", ";
            count++;
        }
    }
    std::cout << std::endl;
}

/**
 * @brief Print first N values of dense weights for debugging
 */
void PrintWeightsDebug(const std::vector<std::vector<double>>& weights, const std::string& name, size_t numValues = 10) {
    if (!DEBUG_MODE) return;

    std::cout << "  [DEBUG WEIGHTS] " << name << " first row [0] (first " << std::min(numValues, weights[0].size()) << " values):" << std::endl;
    std::cout << "    ";
    for (size_t i = 0; i < std::min(numValues, weights[0].size()); i++) {
        std::cout << std::fixed << std::setprecision(4) << weights[0][i];
        if (i < std::min(numValues, weights[0].size()) - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

/**
 * @brief Apply ReLU activation function
 */
std::vector<double> ReLU(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::max(0.0, input[i]);
    }
    return output;
}

/**
 * @brief General cleartext 2D convolution
 * @param input 3D input tensor (in_channels, height, width)
 * @param kernel 4D kernel tensor (out_channels, in_channels, kernel_height, kernel_width)
 * @param stride Convolution stride
 * @param padding Zero padding size
 * @param dilation Kernel dilation
 * @return 3D output tensor (out_channels, output_height, output_width)
 */
std::vector<std::vector<std::vector<double>>> Conv2D(
    const std::vector<std::vector<std::vector<double>>>& input,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
    uint32_t stride = 1,
    uint32_t padding = 0,
    uint32_t dilation = 1
) {
    uint32_t in_channels = input.size();
    uint32_t input_height = input[0].size();
    uint32_t input_width = input[0][0].size();
    uint32_t out_channels = kernel.size();
    uint32_t kernel_height = kernel[0][0].size();
    uint32_t kernel_width = kernel[0][0][0].size();

    uint32_t output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    uint32_t output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    std::vector<std::vector<std::vector<double>>> output(
        out_channels,
        std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0))
    );

    for (uint32_t oc = 0; oc < out_channels; ++oc) {
        for (uint32_t oh = 0; oh < output_height; ++oh) {
            for (uint32_t ow = 0; ow < output_width; ++ow) {
                double sum = 0.0;
                for (uint32_t ic = 0; ic < in_channels; ++ic) {
                    for (uint32_t kh = 0; kh < kernel_height; ++kh) {
                        for (uint32_t kw = 0; kw < kernel_width; ++kw) {
                            int32_t ih = oh * stride - padding + kh * dilation;
                            int32_t iw = ow * stride - padding + kw * dilation;
                            if (ih >= 0 && ih < (int32_t)input_height &&
                                iw >= 0 && iw < (int32_t)input_width) {
                                sum += input[ic][ih][iw] * kernel[oc][ic][kh][kw];
                            }
                        }
                    }
                }
                output[oc][oh][ow] = sum;
            }
        }
    }
    return output;
}

/**
 * @brief Average pooling operation
 * @param input 3D input tensor (channels, height, width)
 * @param kernel_size Size of pooling kernel
 * @param stride Pooling stride
 * @return 3D output tensor (channels, output_height, output_width)
 */
std::vector<std::vector<std::vector<double>>> AvgPool2D(
    const std::vector<std::vector<std::vector<double>>>& input,
    uint32_t kernel_size,
    uint32_t stride
) {
    uint32_t channels = input.size();
    uint32_t input_height = input[0].size();
    uint32_t input_width = input[0][0].size();

    uint32_t output_height = (input_height - kernel_size) / stride + 1;
    uint32_t output_width = (input_width - kernel_size) / stride + 1;

    std::vector<std::vector<std::vector<double>>> output(
        channels,
        std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0))
    );

    double pool_size = kernel_size * kernel_size;

    for (uint32_t c = 0; c < channels; ++c) {
        for (uint32_t oh = 0; oh < output_height; ++oh) {
            for (uint32_t ow = 0; ow < output_width; ++ow) {
                double sum = 0.0;
                for (uint32_t kh = 0; kh < kernel_size; ++kh) {
                    for (uint32_t kw = 0; kw < kernel_size; ++kw) {
                        uint32_t ih = oh * stride + kh;
                        uint32_t iw = ow * stride + kw;
                        sum += input[c][ih][iw];
                    }
                }
                output[c][oh][ow] = sum / pool_size;
            }
        }
    }
    return output;
}

/**
 * @brief Flatten 3D tensor to 1D vector
 */
std::vector<double> Flatten(const std::vector<std::vector<std::vector<double>>>& input) {
    std::vector<double> output;
    for (const auto& channel : input) {
        for (const auto& row : channel) {
            for (double val : row) {
                output.push_back(val);
            }
        }
    }
    return output;
}

/**
 * @brief Perform dense (fully connected) layer
 * @param input Input vector
 * @param weights Weight matrix [output_size][input_size]
 * @return Output vector
 */
std::vector<double> Dense(
    const std::vector<double>& input,
    const std::vector<std::vector<double>>& weights
) {
    uint32_t outputSize = weights.size();
    uint32_t inputSize = input.size();

    std::vector<double> output(outputSize, 0.0);

    for (uint32_t i = 0; i < outputSize; i++) {
        double sum = 0.0;
        for (uint32_t j = 0; j < inputSize; j++) {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }

    return output;
}

void MNISTLeNet5Cleartext() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  MNIST LeNet-5 Network Inference (Cleartext - No Encryption)" << std::endl;
    std::cout << "  Architecture: Conv1->ReLU->Pool1->Conv2->ReLU->Pool2->FC1->ReLU->FC2->ReLU->FC3" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // ========== Set Fixed Random Seed for Reproducibility ==========
    srand(42);  // Fixed seed ensures same weights across all implementations
    std::cout << "Random seed: 42 (for reproducible weights)" << std::endl << std::endl;

    // ========== Network Parameters ==========
    std::cout << "LeNet-5 Architecture:" << std::endl;
    std::cout << "  Input: 28x28 MNIST image (1 channel)" << std::endl;
    std::cout << "  Conv1: 5x5 kernel, 6 output channels, stride=1 -> 24x24x6" << std::endl;
    std::cout << "  ReLU: Standard activation" << std::endl;
    std::cout << "  AvgPool1: 2x2 kernel, stride=2 -> 12x12x6" << std::endl;
    std::cout << "  Conv2: 5x5 kernel, 16 output channels, stride=1 -> 8x8x16" << std::endl;
    std::cout << "  ReLU: Standard activation" << std::endl;
    std::cout << "  AvgPool2: 2x2 kernel, stride=2 -> 4x4x16 = 256" << std::endl;
    std::cout << "  Dense1: 256 -> 120 neurons" << std::endl;
    std::cout << "  ReLU: Standard activation" << std::endl;
    std::cout << "  Dense2: 120 -> 84 neurons" << std::endl;
    std::cout << "  ReLU: Standard activation" << std::endl;
    std::cout << "  Dense3: 84 -> 10 neurons (output)" << std::endl << std::endl;

    // ========== Sample MNIST Input ==========
    std::cout << "Creating sample MNIST input..." << std::endl;
    std::vector<std::vector<std::vector<double>>> mnistInput(1,
        std::vector<std::vector<double>>(28, std::vector<double>(28, 0.0)));

    mnistInput[0] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    std::cout << "Sample input created (1 channel, 28x28)" << std::endl;

    // ========== Define Network Weights ==========
    std::cout << "\nInitializing network weights..." << std::endl;

    // Conv1: 1 -> 6 channels, 5x5 kernel, stride=1
    std::vector<std::vector<std::vector<std::vector<double>>>> conv1Kernel(6);
    for (int oc = 0; oc < 6; oc++) {
        conv1Kernel[oc].resize(1);
        conv1Kernel[oc][0].resize(5, std::vector<double>(5, 0.0));
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                conv1Kernel[oc][0][i][j] = (rand() % 200 - 100) / 100.0;
            }
        }
    }

    uint32_t conv1OutputHeight = 24;  // (28 - 5) / 1 + 1
    uint32_t conv1OutputWidth = 24;
    uint32_t conv1OutputChannels = 6;
    uint32_t conv1FlatSize = conv1OutputHeight * conv1OutputWidth * conv1OutputChannels;  // 3456
    std::cout << "  Conv1 output: " << conv1OutputChannels << " channels, "
              << conv1OutputHeight << "x" << conv1OutputWidth << " = " << conv1FlatSize << std::endl;
    PrintKernelDebug(conv1Kernel, "Conv1 kernel");

    // Pool1: 2x2, stride=2
    uint32_t pool1OutputHeight = 12;  // 24 / 2
    uint32_t pool1OutputWidth = 12;
    uint32_t pool1OutputChannels = 6;
    uint32_t pool1FlatSize = pool1OutputHeight * pool1OutputWidth * pool1OutputChannels;  // 864
    std::cout << "  AvgPool1 output: " << pool1OutputChannels << " channels, "
              << pool1OutputHeight << "x" << pool1OutputWidth << " = " << pool1FlatSize << std::endl;

    // Conv2: 6 -> 16 channels, 5x5 kernel, stride=1
    std::vector<std::vector<std::vector<std::vector<double>>>> conv2Kernel(16);
    for (int oc = 0; oc < 16; oc++) {
        conv2Kernel[oc].resize(6);
        for (int ic = 0; ic < 6; ic++) {
            conv2Kernel[oc][ic].resize(5, std::vector<double>(5, 0.0));
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    conv2Kernel[oc][ic][i][j] = (rand() % 200 - 100) / 100.0;
                }
            }
        }
    }
    uint32_t conv2OutputHeight = 8;  // (12 - 5) / 1 + 1
    uint32_t conv2OutputWidth = 8;
    uint32_t conv2OutputChannels = 16;
    uint32_t conv2FlatSize = conv2OutputHeight * conv2OutputWidth * conv2OutputChannels;  // 1024
    std::cout << "  Conv2 output: " << conv2OutputChannels << " channels, "
              << conv2OutputHeight << "x" << conv2OutputWidth << " = " << conv2FlatSize << std::endl;
    PrintKernelDebug(conv2Kernel, "Conv2 kernel");

    // Pool2: 2x2, stride=2
    uint32_t pool2OutputHeight = 4;  // 8 / 2
    uint32_t pool2OutputWidth = 4;
    uint32_t pool2OutputChannels = 16;
    uint32_t pool2FlatSize = pool2OutputHeight * pool2OutputWidth * pool2OutputChannels;  // 256
    std::cout << "  AvgPool2 output: " << pool2OutputChannels << " channels, "
              << pool2OutputHeight << "x" << pool2OutputWidth << " = " << pool2FlatSize << std::endl;

    // Dense layers
    uint32_t dense1Input = pool2FlatSize;  // 256
    uint32_t dense1Output = 120;
    std::vector<std::vector<double>> dense1Weights(dense1Output, std::vector<double>(dense1Input, 0.0));
    for (uint32_t i = 0; i < dense1Output; i++) {
        for (uint32_t j = 0; j < dense1Input; j++) {
            dense1Weights[i][j] = (rand() % 200 - 100) / 200.0;
        }
    }
    std::cout << "  Dense1: " << dense1Input << " -> " << dense1Output << std::endl;
    PrintWeightsDebug(dense1Weights, "Dense1 weights");

    uint32_t dense2Input = dense1Output;  // 120
    uint32_t dense2Output = 84;
    std::vector<std::vector<double>> dense2Weights(dense2Output, std::vector<double>(dense2Input, 0.0));
    for (uint32_t i = 0; i < dense2Output; i++) {
        for (uint32_t j = 0; j < dense2Input; j++) {
            dense2Weights[i][j] = (rand() % 200 - 100) / 200.0;
        }
    }
    std::cout << "  Dense2: " << dense2Input << " -> " << dense2Output << std::endl;
    PrintWeightsDebug(dense2Weights, "Dense2 weights");

    uint32_t dense3Input = dense2Output;  // 84
    uint32_t dense3Output = 10;
    std::vector<std::vector<double>> dense3Weights(dense3Output, std::vector<double>(dense3Input, 0.0));
    for (uint32_t i = 0; i < dense3Output; i++) {
        for (uint32_t j = 0; j < dense3Input; j++) {
            dense3Weights[i][j] = (rand() % 200 - 100) / 200.0;
        }
    }
    std::cout << "  Dense3: " << dense3Input << " -> " << dense3Output << std::endl;
    PrintWeightsDebug(dense3Weights, "Dense3 weights");

    // ========== Forward Pass ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Starting cleartext inference..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    auto totalStart = high_resolution_clock::now();

    // Layer 1: Conv1
    std::cout << "\n[Layer 1] Conv1 (28x28x1 -> 24x24x6)..." << std::endl;
    auto start = high_resolution_clock::now();
    auto conv1Out = Conv2D(mnistInput, conv1Kernel, 1, 0, 1);
    auto end = high_resolution_clock::now();
    double conv1Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << conv1Time << " ms" << std::endl;

    auto flatConv1Out = Flatten(conv1Out);
    std::cout << "  Flattened size: " << flatConv1Out.size() << std::endl;
    PrintBounds(flatConv1Out, "Conv1 output");
    std::cout << "  >>> Use these bounds for Chebyshev ReLU1 <<<" << std::endl;
    PrintDebugValues(flatConv1Out, "Conv1 output");

    // ========== REMOVE THIS DEBUG CODE LATER ==========
    // Print values at specific critical indices
    if (flatConv1Out.size() == 3456) {
        std::cout << "  Critical indices:" << std::endl;
        std::cout << "    [776]  (ch1) = " << std::fixed << std::setprecision(6) << flatConv1Out[776] << std::endl;
        std::cout << "    [904]  (ch1) = " << std::fixed << std::setprecision(6) << flatConv1Out[904] << std::endl;
        std::cout << "    [1880] (ch3) = " << std::fixed << std::setprecision(6) << flatConv1Out[1880] << std::endl;

        // Check one element from each channel
        std::cout << "  Sample from each channel:" << std::endl;
        std::cout << "    Ch0 [100]:  = " << flatConv1Out[100] << std::endl;
        std::cout << "    Ch1 [676]:  = " << flatConv1Out[676] << std::endl;
        std::cout << "    Ch2 [1252]: = " << flatConv1Out[1252] << std::endl;
        std::cout << "    Ch3 [1828]: = " << flatConv1Out[1828] << std::endl;
        std::cout << "    Ch4 [2404]: = " << flatConv1Out[2404] << std::endl;
        std::cout << "    Ch5 [2980]: = " << flatConv1Out[2980] << std::endl;
    }
    // ========== END DEBUG CODE TO REMOVE ==========

    // Layer 2: ReLU1
    std::cout << "\n[Layer 2] ReLU1..." << std::endl;
    start = high_resolution_clock::now();
    auto relu1Out = ReLU(flatConv1Out);
    end = high_resolution_clock::now();
    double relu1Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << relu1Time << " ms" << std::endl;
    PrintDebugValues(relu1Out, "ReLU1 output");

    // Reshape back to 3D for pooling
    std::vector<std::vector<std::vector<double>>> relu1Out3D(6,
        std::vector<std::vector<double>>(24, std::vector<double>(24, 0.0)));
    size_t idx = 0;
    for (size_t c = 0; c < 6; c++) {
        for (size_t h = 0; h < 24; h++) {
            for (size_t w = 0; w < 24; w++) {
                relu1Out3D[c][h][w] = relu1Out[idx++];
            }
        }
    }

    // Layer 3: AvgPool1
    std::cout << "\n[Layer 3] AvgPool1 (24x24x6 -> 12x12x6)..." << std::endl;
    start = high_resolution_clock::now();
    auto pool1Out = AvgPool2D(relu1Out3D, 2, 2);
    end = high_resolution_clock::now();
    double pool1Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << pool1Time << " ms" << std::endl;

    auto flatPool1Out = Flatten(pool1Out);
    std::cout << "  Flattened size: " << flatPool1Out.size() << std::endl;
    PrintBounds(flatPool1Out, "AvgPool1 output");
    PrintDebugValues(flatPool1Out, "AvgPool1 output");

    // Layer 4: Conv2
    std::cout << "\n[Layer 4] Conv2 (12x12x6 -> 8x8x16)..." << std::endl;
    start = high_resolution_clock::now();
    auto conv2Out = Conv2D(pool1Out, conv2Kernel, 1, 0, 1);
    end = high_resolution_clock::now();
    double conv2Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << conv2Time << " ms" << std::endl;

    auto flatConv2Out = Flatten(conv2Out);
    std::cout << "  Flattened size: " << flatConv2Out.size() << std::endl;
    PrintBounds(flatConv2Out, "Conv2 output");
    std::cout << "  >>> Use these bounds for Chebyshev ReLU2 <<<" << std::endl;
    PrintDebugValues(flatConv2Out, "Conv2 output");

    // Layer 5: ReLU2
    std::cout << "\n[Layer 5] ReLU2..." << std::endl;
    start = high_resolution_clock::now();
    auto relu2Out = ReLU(flatConv2Out);
    end = high_resolution_clock::now();
    double relu2Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << relu2Time << " ms" << std::endl;
    PrintDebugValues(relu2Out, "ReLU2 output");

    // Reshape back to 3D for pooling
    std::vector<std::vector<std::vector<double>>> relu2Out3D(16,
        std::vector<std::vector<double>>(8, std::vector<double>(8, 0.0)));
    idx = 0;
    for (size_t c = 0; c < 16; c++) {
        for (size_t h = 0; h < 8; h++) {
            for (size_t w = 0; w < 8; w++) {
                relu2Out3D[c][h][w] = relu2Out[idx++];
            }
        }
    }

    // Layer 6: AvgPool2
    std::cout << "\n[Layer 6] AvgPool2 (8x8x16 -> 4x4x16)..." << std::endl;
    start = high_resolution_clock::now();
    auto pool2Out = AvgPool2D(relu2Out3D, 2, 2);
    end = high_resolution_clock::now();
    double pool2Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << pool2Time << " ms" << std::endl;

    auto flatPool2Out = Flatten(pool2Out);
    std::cout << "  Flattened size: " << flatPool2Out.size() << std::endl;
    PrintBounds(flatPool2Out, "AvgPool2 output");
    PrintDebugValues(flatPool2Out, "AvgPool2 output");

    // Layer 7: Dense1
    std::cout << "\n[Layer 7] Dense1 (256 -> 120)..." << std::endl;
    start = high_resolution_clock::now();
    auto dense1Out = Dense(flatPool2Out, dense1Weights);
    end = high_resolution_clock::now();
    double dense1Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << dense1Time << " ms" << std::endl;
    PrintBounds(dense1Out, "Dense1 output");
    std::cout << "  >>> Use these bounds for Chebyshev ReLU3 <<<" << std::endl;
    PrintDebugValues(dense1Out, "Dense1 output");

    // Layer 8: ReLU3
    std::cout << "\n[Layer 8] ReLU3..." << std::endl;
    start = high_resolution_clock::now();
    auto relu3Out = ReLU(dense1Out);
    end = high_resolution_clock::now();
    double relu3Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << relu3Time << " ms" << std::endl;
    PrintDebugValues(relu3Out, "ReLU3 output");

    // Layer 9: Dense2
    std::cout << "\n[Layer 9] Dense2 (120 -> 84)..." << std::endl;
    start = high_resolution_clock::now();
    auto dense2Out = Dense(relu3Out, dense2Weights);
    end = high_resolution_clock::now();
    double dense2Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << dense2Time << " ms" << std::endl;
    PrintBounds(dense2Out, "Dense2 output");
    std::cout << "  >>> Use these bounds for Chebyshev ReLU4 <<<" << std::endl;
    PrintDebugValues(dense2Out, "Dense2 output");

    // Layer 10: ReLU4
    std::cout << "\n[Layer 10] ReLU4..." << std::endl;
    start = high_resolution_clock::now();
    auto relu4Out = ReLU(dense2Out);
    end = high_resolution_clock::now();
    double relu4Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << relu4Time << " ms" << std::endl;
    PrintDebugValues(relu4Out, "ReLU4 output");

    // Layer 11: Dense3
    std::cout << "\n[Layer 11] Dense3 (84 -> 10)..." << std::endl;
    start = high_resolution_clock::now();
    auto output = Dense(relu4Out, dense3Weights);
    end = high_resolution_clock::now();
    double dense3Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << dense3Time << " ms" << std::endl;
    PrintDebugValues(output, "Final output");

    auto totalEnd = high_resolution_clock::now();
    double totalInferenceTime = duration_cast<microseconds>(totalEnd - totalStart).count() / 1000.0;
    std::cout << "\nTotal inference time: " << totalInferenceTime << " ms" << std::endl;

    // ========== Display Results ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::cout << "\nOutput logits (10 classes):" << std::endl;
    for (uint32_t i = 0; i < dense3Output; i++) {
        std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(6)
                  << output[i] << std::endl;
    }

    // Find predicted class
    uint32_t predictedClass = 0;
    double maxLogit = output[0];
    for (uint32_t i = 1; i < dense3Output; i++) {
        if (output[i] > maxLogit) {
            maxLogit = output[i];
            predictedClass = i;
        }
    }

    std::cout << "\nPredicted class: " << predictedClass << std::endl;
    std::cout << "Confidence: " << maxLogit << std::endl;

    // ========== Performance Summary ==========
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Performance Summary (LeNet-5)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::left << std::setw(30) << "Layer" << "Time (ms)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Conv1 (28x28x1->24x24x6)" << conv1Time << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU1" << relu1Time << std::endl;
    std::cout << std::left << std::setw(30) << "AvgPool1 (24x24x6->12x12x6)" << pool1Time << std::endl;
    std::cout << std::left << std::setw(30) << "Conv2 (12x12x6->8x8x16)" << conv2Time << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU2" << relu2Time << std::endl;
    std::cout << std::left << std::setw(30) << "AvgPool2 (8x8x16->4x4x16)" << pool2Time << std::endl;
    std::cout << std::left << std::setw(30) << "Dense1 (256->120)" << dense1Time << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU3" << relu3Time << std::endl;
    std::cout << std::left << std::setw(30) << "Dense2 (120->84)" << dense2Time << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU4" << relu4Time << std::endl;
    std::cout << std::left << std::setw(30) << "Dense3 (84->10)" << dense3Time << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Total Inference" << totalInferenceTime << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\n✓ MNIST LeNet-5 Cleartext Inference Complete!" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        MNISTLeNet5Cleartext();
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
