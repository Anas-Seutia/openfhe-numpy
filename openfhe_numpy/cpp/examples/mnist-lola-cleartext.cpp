#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>

using namespace std::chrono;

/**
 * @brief MNIST LoLa Network Architecture (Cleartext - No Encryption)
 *
 * Network: Conv -> ReLU -> Dense -> ReLU -> Dense
 * - Input: 28x28 MNIST image (1 channel)
 * - Conv: 5x5 kernel, 5 output channels, stride=2, no padding -> 12x12x5
 * - ReLU: Standard activation max(0, x)
 * - Dense1: 12x12x5 = 720 -> 64 neurons
 * - ReLU: Standard activation
 * - Dense2: 64 -> 10 neurons (output classes)
 */

/**
 * @brief Print min/max bounds of a vector
 */
void PrintBounds(const std::vector<double>& vec, const std::string& name) {
    double minVal = *std::min_element(vec.begin(), vec.end());
    double maxVal = *std::max_element(vec.begin(), vec.end());
    std::cout << "  " << name << " bounds: [" << std::fixed << std::setprecision(6)
              << minVal << ", " << maxVal << "]" << std::endl;
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
 * @brief Perform 2D convolution
 * @param input Input image [height][width]
 * @param kernels Convolution kernels [out_channels][in_channels][kernel_h][kernel_w]
 * @param stride Stride value
 * @param padding Padding value
 * @return Output feature maps [out_channels][out_height][out_width]
 */
std::vector<std::vector<std::vector<double>>> Conv2D(
    const std::vector<std::vector<double>>& input,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernels,
    uint32_t stride,
    uint32_t padding
) {
    uint32_t inputHeight = input.size();
    uint32_t inputWidth = input[0].size();
    uint32_t kernelHeight = kernels[0][0].size();
    uint32_t kernelWidth = kernels[0][0][0].size();
    uint32_t outputChannels = kernels.size();

    uint32_t outputHeight = (inputHeight - kernelHeight + 2 * padding) / stride + 1;
    uint32_t outputWidth = (inputWidth - kernelWidth + 2 * padding) / stride + 1;

    // Initialize output
    std::vector<std::vector<std::vector<double>>> output(
        outputChannels,
        std::vector<std::vector<double>>(
            outputHeight,
            std::vector<double>(outputWidth, 0.0)
        )
    );

    // Perform convolution
    for (uint32_t oc = 0; oc < outputChannels; oc++) {
        for (uint32_t oh = 0; oh < outputHeight; oh++) {
            for (uint32_t ow = 0; ow < outputWidth; ow++) {
                double sum = 0.0;

                // Convolve over kernel
                for (uint32_t kh = 0; kh < kernelHeight; kh++) {
                    for (uint32_t kw = 0; kw < kernelWidth; kw++) {
                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;

                        // Check bounds (padding)
                        if (ih >= 0 && ih < (int)inputHeight && iw >= 0 && iw < (int)inputWidth) {
                            sum += input[ih][iw] * kernels[oc][0][kh][kw];  // Single input channel
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

void MNISTLoLaCleartext() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  MNIST LoLa Network Inference (Cleartext - No Encryption)" << std::endl;
    std::cout << "  Architecture: Conv -> ReLU -> Dense -> ReLU -> Dense" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // ========== Set Fixed Random Seed for Reproducibility ==========
    srand(42);  // Fixed seed ensures same weights across all implementations
    std::cout << "Random seed: 42 (for reproducible weights)" << std::endl << std::endl;

    // ========== Network Parameters ==========
    std::cout << "Network Architecture:" << std::endl;
    std::cout << "  Input: 28x28 MNIST image (1 channel)" << std::endl;
    std::cout << "  Conv: 5x5 kernel, 5 output channels, stride=2 -> 12x12x5" << std::endl;
    std::cout << "  ReLU: Standard activation" << std::endl;
    std::cout << "  Dense1: 720 -> 64 neurons" << std::endl;
    std::cout << "  ReLU: Standard activation" << std::endl;
    std::cout << "  Dense2: 64 -> 10 neurons (output)" << std::endl << std::endl;

    // ========== Sample MNIST Input (simplified) ==========
    std::cout << "Creating sample MNIST input..." << std::endl;
    std::vector<std::vector<double>> mnistInput(28, std::vector<double>(28, 0.0));

    // Create a simple vertical edge pattern in the center
    for (int i = 8; i < 20; i++) {
        for (int j = 12; j < 16; j++) {
            mnistInput[i][j] = (j < 14) ? 1.0 : 0.5;
        }
    }

    std::cout << "Sample input created (28x28)" << std::endl;

    // ========== Define Network Weights ==========
    std::cout << "\nInitializing network weights..." << std::endl;

    // Conv layer: 5 output channels, 1 input channel, 5x5 kernel
    // Using same random initialization as encrypted version
    std::vector<std::vector<std::vector<std::vector<double>>>> convKernel(5);
    for (int oc = 0; oc < 5; oc++) {
        convKernel[oc].resize(1);  // 1 input channel
        convKernel[oc][0].resize(5, std::vector<double>(5, 0.0));
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                convKernel[oc][0][i][j] = (rand() % 200 - 100) / 100.0;  // Range [-1.0, 1.0]
            }
        }
    }

    uint32_t convStride = 2;
    uint32_t convPadding = 0;
    uint32_t convOutputHeight = (28 - 5) / convStride + 1;  // 12
    uint32_t convOutputWidth = (28 - 5) / convStride + 1;   // 12
    uint32_t convOutputChannels = 5;
    uint32_t flattenedSize = convOutputHeight * convOutputWidth * convOutputChannels;  // 720

    std::cout << "  Conv output shape: " << convOutputChannels << " channels, "
              << convOutputHeight << "x" << convOutputWidth
              << " = " << flattenedSize << " total" << std::endl;

    // Dense layer 1: 720 -> 64
    // Using same pseudo-random initialization as encrypted version
    uint32_t dense1Input = flattenedSize;
    uint32_t dense1Output = 64;
    std::vector<std::vector<double>> dense1Weights(dense1Output, std::vector<double>(dense1Input, 0.0));
    for (uint32_t i = 0; i < dense1Output; i++) {
        for (uint32_t j = 0; j < dense1Input; j++) {
            dense1Weights[i][j] = (rand() % 200 - 100) / 200.0;  // Scaled to prevent explosion
        }
    }
    std::cout << "  Dense1 shape: " << dense1Input << " -> " << dense1Output << std::endl;

    // Dense layer 2: 64 -> 10
    // Using same random initialization as encrypted version WITH SCALING
    uint32_t dense2Input = dense1Output;
    uint32_t dense2Output = 10;
    double dense2Scale = 1.0 / std::sqrt(dense2Input);  // Xavier scaling
    std::vector<std::vector<double>> dense2Weights(dense2Output, std::vector<double>(dense2Input, 0.0));
    for (uint32_t i = 0; i < dense2Output; i++) {
        for (uint32_t j = 0; j < dense2Input; j++) {
            dense2Weights[i][j] = (rand() % 200 - 100) / 200.0;  // Scaled
        }
    }
    std::cout << "  Dense2 shape: " << dense2Input << " -> " << dense2Output << std::endl;

    // ========== Forward Pass ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Starting cleartext inference..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    auto totalStart = high_resolution_clock::now();

    // Layer 1: Convolution
    std::cout << "\n[Layer 1] Convolution (28x28x1 -> 12x12x5)..." << std::endl;
    auto start = high_resolution_clock::now();
    auto convOut = Conv2D(mnistInput, convKernel, convStride, convPadding);
    auto end = high_resolution_clock::now();
    double convTime = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << convTime << " ms" << std::endl;

    // Flatten for dense layers
    auto flatConvOut = Flatten(convOut);
    std::cout << "  Flattened size: " << flatConvOut.size() << std::endl;
    PrintBounds(flatConvOut, "Conv output");
    std::cout << "  >>> Use these bounds for Chebyshev ReLU1 <<<" << std::endl;

    // Layer 2: ReLU
    std::cout << "\n[Layer 2] ReLU..." << std::endl;
    start = high_resolution_clock::now();
    auto relu1Out = ReLU(flatConvOut);
    end = high_resolution_clock::now();
    double relu1Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << relu1Time << " ms" << std::endl;

    // Layer 3: Dense 1 (720 -> 64)
    std::cout << "\n[Layer 3] Dense1 (720 -> 64)..." << std::endl;
    start = high_resolution_clock::now();
    auto dense1Out = Dense(relu1Out, dense1Weights);
    end = high_resolution_clock::now();
    double dense1Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << dense1Time << " ms" << std::endl;
    PrintBounds(dense1Out, "Dense1 output");
    std::cout << "  >>> Use these bounds for Chebyshev ReLU2 <<<" << std::endl;

    // Layer 4: ReLU
    std::cout << "\n[Layer 4] ReLU..." << std::endl;
    start = high_resolution_clock::now();
    auto relu2Out = ReLU(dense1Out);
    end = high_resolution_clock::now();
    double relu2Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << relu2Time << " ms" << std::endl;

    // Layer 5: Dense 2 (64 -> 10)
    std::cout << "\n[Layer 5] Dense2 (64 -> 10)..." << std::endl;
    start = high_resolution_clock::now();
    auto output = Dense(relu2Out, dense2Weights);
    end = high_resolution_clock::now();
    double dense2Time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "  Time: " << dense2Time << " ms" << std::endl;

    auto totalEnd = high_resolution_clock::now();
    double totalInferenceTime = duration_cast<microseconds>(totalEnd - totalStart).count() / 1000.0;
    std::cout << "\nTotal inference time: " << totalInferenceTime << " ms" << std::endl;

    // ========== Display Results ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::cout << "\nOutput logits (10 classes):" << std::endl;
    for (uint32_t i = 0; i < dense2Output; i++) {
        std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(6)
                  << output[i] << std::endl;
    }

    // Find predicted class
    uint32_t predictedClass = 0;
    double maxLogit = output[0];
    for (uint32_t i = 1; i < dense2Output; i++) {
        if (output[i] > maxLogit) {
            maxLogit = output[i];
            predictedClass = i;
        }
    }

    std::cout << "\nPredicted class: " << predictedClass << std::endl;
    std::cout << "Confidence: " << maxLogit << std::endl;

    // ========== Performance Summary ==========
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Performance Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::left << std::setw(30) << "Layer" << "Time (ms)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Convolution" << convTime << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU 1" << relu1Time << std::endl;
    std::cout << std::left << std::setw(30) << "Dense 1" << dense1Time << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU 2" << relu2Time << std::endl;
    std::cout << std::left << std::setw(30) << "Dense 2" << dense2Time << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Total Inference" << totalInferenceTime << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\n✓ MNIST LoLa Cleartext Inference Complete!" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        MNISTLoLaCleartext();
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
