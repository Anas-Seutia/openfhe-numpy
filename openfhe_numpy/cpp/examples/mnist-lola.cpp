// Suppress false positive warning from OpenFHE headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "numpy_enc_matrix.h"
#include "openfhe.h"
#include "binfhecontext.h"
#pragma GCC diagnostic pop

#include "numpy_utils.h"
#include "numpy_helper_functions.h"
#include "conv_helper_function.h"
#include "relu_helper_function.h"
#include "weight_loader.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace openfhe_numpy;
using namespace lbcrypto;

// ========== ACTIVATION FUNCTION TYPE ==========
enum class ActivationType {
    SCHEME_SWITCH,  // CKKS-FHEW-CKKS scheme switching
    CHEBYSHEV,      // Chebyshev polynomial approximation
    SQUARE          // Square approximation (x^2 based)
};

/**
 * @brief MNIST LoLa Network Architecture (Scheme Switching for ReLU)
 *
 * Network: Conv -> ReLU -> Dense -> ReLU -> Dense
 * - Input: 28x28 MNIST image (1 channel)
 * - Conv: 5x5 kernel, 5 output channels, stride=2, no padding -> 12x12x5
 * - ReLU: Scheme switching (CKKS-FHEW-CKKS)
 * - Dense1: 12x12x5 = 720 -> 64 neurons
 * - ReLU: Scheme switching
 * - Dense2: 64 -> 10 neurons (output classes)
 */

/**
 * @brief Map Chebyshev polynomial degree to multiplicative depth
 * Based on: https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/FUNCTION_EVALUATION.md
 */
uint32_t GetChebyDepthFromDegree(uint32_t degree) {
    if (degree >= 3 && degree <= 5) return 4;
    if (degree >= 6 && degree <= 13) return 5;
    if (degree >= 14 && degree <= 27) return 6;
    if (degree >= 28 && degree <= 59) return 7;
    if (degree >= 60 && degree <= 119) return 8;
    if (degree >= 120 && degree <= 247) return 9;
    if (degree >= 248 && degree <= 495) return 10;
    if (degree >= 496 && degree <= 1007) return 11;
    if (degree >= 1008 && degree <= 2031) return 12;
    if (degree >= 2032 && degree <= 4031) return 13;
    if (degree >= 4032 && degree <= 8127) return 14;
    if (degree >= 8128 && degree <= 16255) return 15;
    if (degree >= 16256 && degree <= 32639) return 16;
    if (degree >= 32640 && degree <= 65279) return 17;
    if (degree >= 65280 && degree <= 130815) return 18;
    if (degree >= 130816 && degree <= 261631) return 19;

    throw std::runtime_error("Chebyshev degree out of supported range (3-261631)");
}

/**
 * @brief Helper function to print min/max bounds of a vector
 */
void PrintBounds(const std::vector<double>& vec, const std::string& name) {
    double minVal = *std::min_element(vec.begin(), vec.end());
    double maxVal = *std::max_element(vec.begin(), vec.end());
    std::cout << "  " << name << " bounds: [" << std::fixed << std::setprecision(6)
              << minVal << ", " << maxVal << "]" << std::endl;
}

/**
 * @brief Compare cleartext and encrypted results
 */
void CompareVectors(
    const std::vector<double>& cleartext,
    const std::vector<double>& encrypted,
    const std::string& layerName,
    double threshold = 1e-2
) {
    if (cleartext.size() != encrypted.size()) {
        std::cout << "  [VALIDATION ERROR] " << layerName << ": Size mismatch! "
                  << "Cleartext: " << cleartext.size() << ", Encrypted: " << encrypted.size() << std::endl;
        return;
    }

    PrintBounds(cleartext, "cleartext");
    PrintBounds(encrypted, "encrypted");

    double maxError = 0.0;
    double sumError = 0.0;

    for (size_t i = 0; i < cleartext.size(); ++i) {
        double error = std::abs(cleartext[i] - encrypted[i]);
        sumError += error;
        if (error > maxError) {
            maxError = error;
        }
    }

    double avgError = sumError / cleartext.size();

    std::cout << "    Max error: " << std::scientific << std::setprecision(6) << maxError << std::endl;
    std::cout << "    Avg error: " << std::scientific << std::setprecision(6) << avgError << std::endl;
}

/**
 * @brief Cleartext 2D convolution for validation
 * Input: 2D matrix (height, width) - single channel OR 3D (channels, height, width)
 * Kernel: 4D (out_channels, in_channels, kernel_height, kernel_width)
 * Returns: 3D (out_channels, output_height, output_width)
 */
std::vector<std::vector<std::vector<double>>> CleartextConv2D(
    const std::vector<std::vector<double>>& input,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
    uint32_t stride = 1,
    uint32_t padding = 0,
    const std::vector<double>* bias = nullptr
) {
    uint32_t input_height = input.size();
    uint32_t input_width = input[0].size();
    uint32_t out_channels = kernel.size();
    uint32_t in_channels = kernel[0].size();
    uint32_t kernel_height = kernel[0][0].size();
    uint32_t kernel_width = kernel[0][0][0].size();

    uint32_t output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    uint32_t output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

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
                            int32_t ih = oh * stride - padding + kh;
                            int32_t iw = ow * stride - padding + kw;
                            if (ih >= 0 && ih < (int32_t)input_height &&
                                iw >= 0 && iw < (int32_t)input_width) {
                                sum += input[ih][iw] * kernel[oc][ic][kh][kw];
                            }
                        }
                    }
                }
                output[oc][oh][ow] = sum;
                if (bias) {
                    output[oc][oh][ow] += (*bias)[oc];
                }
            }
        }
    }
    return output;
}

/**
 * @brief Flatten 3D tensor to 1D vector
 */
std::vector<double> CleartextFlatten(const std::vector<std::vector<std::vector<double>>>& input) {
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
 * @brief Cleartext dense layer
 */
std::vector<double> CleartextDense(
    const std::vector<double>& input,
    const std::vector<std::vector<double>>& weights,
    const std::vector<double>* bias = nullptr
) {
    uint32_t output_size = weights.size();
    std::vector<double> output(output_size, 0.0);

    for (uint32_t i = 0; i < output_size; ++i) {
        for (uint32_t j = 0; j < input.size(); ++j) {
            output[i] += input[j] * weights[i][j];
        }
        if (bias) {
            output[i] += (*bias)[i];
        }
    }
    return output;
}

/**
 * @brief Cleartext ReLU activation
 */
std::vector<double> CleartextReLU(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0, input[i]);
    }
    return output;
}

/**
 * @brief Cleartext Square activation for validation
 */
std::vector<double> CleartextSquare(const std::vector<double>& input, double scaleFactor = 1.0) {
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = input[i] * input[i] * scaleFactor;
    }
    return output;
}

/**
 * @brief Cleartext activation function dispatcher
 */
std::vector<double> CleartextActivation(const std::vector<double>& input, ActivationType activationType, double scaleFactor = 1.0) {
    switch (activationType) {
        case ActivationType::SCHEME_SWITCH:
        case ActivationType::CHEBYSHEV:
            return CleartextReLU(input);  // Both use ReLU
        case ActivationType::SQUARE:
            return CleartextSquare(input, scaleFactor);
        default:
            throw std::runtime_error("Unknown activation type");
    }
}

// ========== CLEARTEXT VALIDATION FUNCTIONS (END) ==========

/**
 * @brief Helper function to prepare bias vector for addition to ciphertext
 * @param bias Bias vector (one value per output channel/neuron)
 * @param outputSize Total size of output (for dense layers, or multiplexed size for conv)
 * @param channels Number of channels (for conv layers)
 * @param spatialSize Height * width (for conv layers, set to 1 for dense)
 * @param outputGap Multiplexing gap (default 1 = no multiplexing)
 * @param outputHeight Output height before multiplexing (for conv with multiplexing)
 * @param outputWidth Output width before multiplexing (for conv with multiplexing)
 */
std::vector<double> PrepareBiasVector(
    const std::vector<double>& bias,
    uint32_t outputSize,
    uint32_t channels = 1,
    uint32_t spatialSize = 1,
    uint32_t outputGap = 1,
    uint32_t outputHeight = 0,
    uint32_t outputWidth = 0
) {
    if (spatialSize == 1) {
        // Dense layer: bias[i] goes to position i
        std::vector<double> biasVec(outputSize, 0.0);
        for (size_t i = 0; i < bias.size() && i < outputSize; i++) {
            biasVec[i] = bias[i];
        }
        return biasVec;
    } else if (outputGap > 1) {
        // Conv layer with multiplexing
        uint32_t outputGapSquared = outputGap * outputGap;
        uint32_t multiplexedHeight = outputHeight * outputGap;
        uint32_t multiplexedWidth = outputWidth * outputGap;

        std::vector<double> biasVec(outputSize, 0.0);
        for (uint32_t co = 0; co < channels; ++co) {
            for (uint32_t h = 0; h < outputHeight; ++h) {
                for (uint32_t w = 0; w < outputWidth; ++w) {
                    uint32_t superCh = co / outputGapSquared;
                    uint32_t inBlock = co % outputGapSquared;
                    uint32_t blockH = inBlock / outputGap;
                    uint32_t blockW = inBlock % outputGap;
                    uint32_t finalH = h * outputGap + blockH;
                    uint32_t finalW = w * outputGap + blockW;
                    uint32_t idx = superCh * multiplexedHeight * multiplexedWidth +
                                   finalH * multiplexedWidth + finalW;
                    biasVec[idx] = bias[co];
                }
            }
        }
        return biasVec;
    } else {
        // Conv layer without multiplexing: bias[c] is replicated across all spatial positions of channel c
        std::vector<double> biasVec(outputSize, 0.0);
        for (uint32_t c = 0; c < channels; c++) {
            for (uint32_t s = 0; s < spatialSize; s++) {
                biasVec[c * spatialSize + s] = bias[c];
            }
        }
        return biasVec;
    }
}

/**
 * @brief Helper function to perform ReLU using Chebyshev approximation
 */
Ciphertext<DCRTPoly> EvalReLUChebyshev(
    CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct,
    uint32_t polyDegree = 63,
    double lowerBound = -10.0,
    double upperBound = 10.0
) {
    // Use Chebyshev approximation for ReLU function
    auto reluResult = cc->EvalChebyshevFunction(
        [](double x) -> double { return std::max(0.0, x); },
        ct,
        lowerBound,
        upperBound,
        polyDegree
    );
    return reluResult;
}

/**
 * @brief Helper function to perform Square activation: f(x) = x^2
 */
Ciphertext<DCRTPoly> EvalSquareActivation(
    CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct
) {
    // Simply square the input
    auto squareResult = cc->EvalMult(ct, ct);
    return squareResult;
}

/**
 * @brief Helper function to perform ReLU using scheme switching
 */
Ciphertext<DCRTPoly> EvalReLUSchemeSwitching(
    CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct,
    const PublicKey<DCRTPoly>& publicKey,
    uint32_t numSlots,
    uint32_t totalSlots,
    double scaleSign = 8.0
) {
    // Create zero ciphertext for comparison
    std::vector<double> zeros(totalSlots, 0.0);
    Plaintext ptxtZero = cc->MakeCKKSPackedPlaintext(zeros, 1, 0, nullptr, totalSlots);
    auto ctZero = cc->Encrypt(publicKey, ptxtZero);

    // ReLU(x) = x * (x > 0)
    // Step 1: Compute comparison result (x > 0)
    auto ctComparison = cc->EvalCompareSchemeSwitching(ct, ctZero, NextPow2(numSlots), totalSlots, 0, scaleSign);

    // Step 2: Multiply input by comparison result to get ReLU
    // The comparison returns 1 if x < 0, 0 otherwise
    // We need to invert: (1 - comparison) to get mask
    auto ctReLU = cc->EvalMult(ct, cc->EvalSub(1, ctComparison));

    return ctReLU;
}

/**
 * @brief Unified activation function that dispatches to the appropriate method
 */
Ciphertext<DCRTPoly> EvalActivation(
    CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct,
    ActivationType activationType,
    const PublicKey<DCRTPoly>& publicKey = nullptr,
    uint32_t numSlots = 0,
    uint32_t totalSlots = 0,
    double scaleSign = 8.0,
    uint32_t chebyDegree = 63,
    double chebyLower = -10.0,
    double chebyUpper = 10.0,
    double scaleFactor = 1.0
) {
    switch (activationType) {
        case ActivationType::SCHEME_SWITCH:
            return EvalReLUSchemeSwitching(cc, ct, publicKey, numSlots, totalSlots, scaleSign);

        case ActivationType::CHEBYSHEV:
            return EvalReLUChebyshev(cc, ct, chebyDegree, chebyLower, chebyUpper);

        case ActivationType::SQUARE: {
            auto result = EvalSquareActivation(cc, ct);
            // Apply scale factor if not 1.0
            if (std::abs(scaleFactor - 1.0) > 1e-9) {
                result = cc->EvalMult(result, scaleFactor);
            }
            return result;
        }

        default:
            throw std::runtime_error("Unknown activation type");
    }
}

void MNISTLoLaInference(int sampleIndex = 8, ActivationType activationType = ActivationType::SCHEME_SWITCH, uint32_t ChebyDegree = 119, uint32_t ChebyMultDepth = 8, bool useOptimized = false, bool enableValidation = true) {
    std::cout << "\n" << std::string(80, '=') << std::endl;

    // Print activation type
    std::string activationName;
    switch (activationType) {
        case ActivationType::SCHEME_SWITCH:
            activationName = "Scheme Switching";
            break;
        case ActivationType::CHEBYSHEV:
            activationName = "Chebyshev Approximation";
            break;
        case ActivationType::SQUARE:
            activationName = "Square Activation (x²)";
            break;
    }

    std::cout << "  MNIST LoLa Network Inference (" << activationName << ")" << std::endl;
    std::cout << "  Architecture: Conv -> ReLU -> Dense -> ReLU -> Dense" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // ========== Network Parameters ==========
    std::cout << "Network Architecture:" << std::endl;
    std::cout << "  Input: 28x28 MNIST image (1 channel)" << std::endl;
    std::cout << "  Conv: 5x5 kernel, 5 output channels, stride=2 -> 12x12x5" << std::endl;
    std::cout << "  Activation: " << activationName << std::endl;
    std::cout << "  Dense1: 720 -> 64 neurons" << std::endl;
    std::cout << "  Activation: " << activationName << std::endl;
    std::cout << "  Dense2: 64 -> 10 neurons (output)" << std::endl << std::endl;

    // ========== Load MNIST Input ==========
    std::cout << "Loading MNIST test sample #" << sampleIndex << "..." << std::endl;

    std::string mnistDataDir = "../openfhe_numpy/cpp/data/mnist";
    std::string actualFile = "";
    int trueLabel = -1;

    // Search for the sample file (we don't know the label beforehand)
    for (int label = 0; label < 10; label++) {
        std::stringstream testPath;
        testPath << mnistDataDir << "/mnist_" << sampleIndex << "_label_" << label << ".bin";
        std::ifstream testFile(testPath.str());
        if (testFile.good()) {
            actualFile = testPath.str();
            trueLabel = label;
            break;
        }
    }

    std::vector<std::vector<double>> mnistInput;
    if (actualFile.empty()) {
        std::cout << "Warning: Could not find MNIST sample #" << sampleIndex << " in " << mnistDataDir << std::endl;
        std::cout << "Using fallback hardcoded sample (digit 5)..." << std::endl;

        // Fallback: Use hardcoded sample
        mnistInput = {
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
        trueLabel = 5;
    } else {
        mnistInput = LoadMNISTImage(actualFile);
        std::cout << "  Loaded sample with true label: " << trueLabel << std::endl;
    }

    // ========== Setup Crypto Context ==========
    std::cout << "\nSetting up crypto context..." << std::endl;

    ScalingTechnique scTech = FLEXIBLEAUTO;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;

    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;
    uint32_t ringDim = 32768;
    std::vector<uint32_t> levelBudget = {3, 3};
    std::vector<uint32_t> bsgsDim = {0, 0};
    SecurityLevel sl = HEStd_NotSet;
    BINFHE_PARAMSET slBin = STD128;
    uint32_t logQ_ccLWE = 27;
    uint32_t slots = 4096;
    uint32_t batchSize = slots;

    // Bootstrapping parameters
    uint32_t levelsAvailableAfterBootstrap = ChebyMultDepth + 1;
    uint32_t approxBootstrapDepth = FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    uint32_t multDepth = 1;
    bool enableBootstrapping = false;  // Track if bootstrapping is needed

    if (activationType == ActivationType::CHEBYSHEV) {
        // conv + act + fc + act + fc
        uint32_t option1 = 1 + ChebyMultDepth + 1 + ChebyMultDepth + 1;
        uint32_t option2 = std::max({1U,ChebyMultDepth,1U,ChebyMultDepth,1U}) + approxBootstrapDepth + 1;
        // Prefer option1 unless option2 saves 4+ layers (threshold of 3)
        enableBootstrapping = (option2 + 3 < option1);
        multDepth = enableBootstrapping ? option2 : option1;
    } else if (activationType == ActivationType::SQUARE) {
        // conv + act + fc + act + fc
        uint32_t option1 = 1 + 2 + 1 + 2 + 1;
        uint32_t option2 = std::max({1U,2U,1U,2U,1U}) + approxBootstrapDepth + 1;
        // Prefer option1 unless option2 saves 4+ layers (threshold of 3)
        enableBootstrapping = (option2 + 3 < option1);
        multDepth = enableBootstrapping ? option2 : option1;
    } else if (activationType == ActivationType::SCHEME_SWITCH) {
        // conv + act + fc + act + fc
        uint32_t option1 = 1 + 13 + 1 + 1 + 1;
        uint32_t option2 = std::max({1U,13U,1U,1U,1U}) + approxBootstrapDepth + 1;
        // Prefer option1 unless option2 saves 4+ layers (threshold of 3)
        enableBootstrapping = (option2 + 3 < option1);
        multDepth = enableBootstrapping ? option2 : option1;
    }

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetScalingTechnique(scTech);
    parameters.SetSecurityLevel(sl);
    parameters.SetRingDim(ringDim);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecretKeyDist(secretKeyDist);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    if (enableBootstrapping) cc->Enable(FHE);  // Enable bootstrapping only if needed

    // Enable scheme switching if using that activation type
    if (activationType == ActivationType::SCHEME_SWITCH) {
        cc->Enable(SCHEMESWITCH);
    }

    std::cout << "CKKS scheme using ring dimension " << cc->GetRingDimension() << std::endl;
    std::cout << "  Number of slots: " << slots << std::endl;
    std::cout << "  Multiplicative depth: " << multDepth << std::endl;
    std::cout << "  Activation function: " << activationName << std::endl;

    // ========== Key Generation ==========
    std::cout << "\nGenerating keys..." << std::endl;
    TimeVar t;
    TIC(t);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    if (enableBootstrapping) {
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, slots);
        cc->EvalBootstrapKeyGen(keys.secretKey, slots);
    }

    // Setup scheme switching parameters if needed
    double scaleSignFHEW = 1.0;
    if (activationType == ActivationType::SCHEME_SWITCH) {
        SchSwchParams params;
        params.SetSecurityLevelCKKS(sl);
        params.SetSecurityLevelFHEW(slBin);
        params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE);
        params.SetNumSlotsCKKS(slots);
        params.SetNumValues(720);

        auto privateKeyFHEW = cc->EvalSchemeSwitchingSetup(params);
        auto ccLWE = cc->GetBinCCForSchemeSwitch();
        ccLWE->BTKeyGen(privateKeyFHEW);
        cc->EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW);

        auto modulus_LWE = 1 << logQ_ccLWE;
        auto beta = ccLWE->GetBeta().ConvertToInt();
        auto pLWE = modulus_LWE / (2 * beta);
        cc->EvalCompareSwitchPrecompute(pLWE, scaleSignFHEW);

        std::cout << "FHEW scheme is using lattice parameter " << ccLWE->GetParams()->GetLWEParams()->Getn();
        std::cout << ", logQ " << logQ_ccLWE;
        std::cout << ", modulus q " << ccLWE->GetParams()->GetLWEParams()->Getq() << std::endl << std::endl;
    }

    std::cout << "  Key generation time: " << TOC(t) << " ms" << std::endl;

    // ========== Load Network Weights ==========
    std::cout << "\nLoading network weights from trained model..." << std::endl;

    // Select model based on activation type
    std::string weightsDir;
    if (activationType == ActivationType::SQUARE) {
        weightsDir = "../openfhe_numpy/cpp/models/lola_weights_square";
    } else {
        weightsDir = "../openfhe_numpy/cpp/models/lola_weights_relu";
    }

    LoLaWeights trainedWeights = LoadLoLaWeights(weightsDir);

    // Use loaded weights
    auto convKernel = trainedWeights.conv1_weight;
    auto dense1Weights = trainedWeights.fc1_weight;
    auto dense2Weights = trainedWeights.fc2_weight;

    // Network dimensions
    uint32_t convStride = 2;
    uint32_t convPadding = 0;
    uint32_t convOutputHeight = (28 - 5) / convStride + 1;  // 12
    uint32_t convOutputWidth = (28 - 5) / convStride + 1;   // 12
    uint32_t convOutputChannels = 5;
    uint32_t flattenedSize = convOutputHeight * convOutputWidth * convOutputChannels;  // 720

    // Multiplexing parameters (optimized vs unoptimized)
    uint32_t input_gap = 1;
    uint32_t output_gap = useOptimized ? (input_gap * convStride) : 1;

    std::cout << "Running in " << (useOptimized ? "optimized" : "unoptimized") << " mode" << std::endl;
    std::cout << "  Conv: input_gap=" << input_gap << ", output_gap=" << output_gap << std::endl;

    uint32_t dense1Output = 64;
    uint32_t dense2Output = 10;

    // Use loaded biases
    auto convBias = trainedWeights.conv1_bias;
    auto dense1Bias = trainedWeights.fc1_bias;
    auto dense2Bias = trainedWeights.fc2_bias;

    // Scale factors for square activation (loaded from model if available)
    double scale1 = trainedWeights.scale1;
    double scale2 = trainedWeights.scale2;

    // ========== Prepare Encrypted Network Weights ==========
    std::cout << "\nPreparing encrypted network weights..." << std::endl;

    // Convolution layer
    TIC(t);
    auto toeplitzConv = ConstructConv2DToeplitz(convKernel, 28, 28, convStride, convPadding, 1, input_gap, output_gap);
    std::vector<std::vector<double>> convDiagonals = PackMatDiagWise(toeplitzConv, batchSize);
    std::size_t convCols = convDiagonals.size();
    std::vector<bool> convNonZeros(convCols);
    std::vector<int32_t> convRotations = getOptimalRots(convDiagonals, &convNonZeros, useOptimized);
    std::cout << "  Conv Toeplitz: " << toeplitzConv.size() << " rows (multiplexed), "
              << toeplitzConv[0].size() << " cols, "
              << convRotations.size() << " rotation keys needed" << std::endl;

    // Dense layer 1 - unmultiplex the conv output from 1152 -> 720
    auto dense1 = MultiplexDenseMatrix(dense1Weights, convOutputHeight, convOutputWidth, output_gap);
    std::vector<std::vector<double>> dense1Diagonals = PackMatDiagWise(dense1, batchSize);
    std::size_t dense1Cols = dense1Diagonals.size();
    std::vector<bool> dense1NonZeros(dense1Cols);
    std::vector<int32_t> dense1Rotations = getOptimalRots(dense1Diagonals, &dense1NonZeros, useOptimized);
    std::cout << "  Dense1: " << dense1.size() << " rows, "
              << dense1[0].size() << " cols (multiplexed input), "
              << dense1Rotations.size() << " rotation keys needed" << std::endl;

    // Dense layer 2
    std::vector<std::vector<double>> dense2Diagonals = PackMatDiagWise(dense2Weights, batchSize);
    std::size_t dense2Cols = dense2Diagonals.size();
    std::vector<bool> dense2NonZeros(dense2Cols);
    std::vector<int32_t> dense2Rotations = getOptimalRots(dense2Diagonals, &dense2NonZeros, useOptimized);
    std::cout << "  Dense2: " << dense2Cols << " rows, "
              << dense2Rotations.size() << " rotation keys needed" << std::endl;

    // Collect all rotation indices (one key per index for faster inference)
    std::vector<int32_t> allRotations;
    allRotations.insert(allRotations.end(), convRotations.begin(), convRotations.end());
    allRotations.insert(allRotations.end(), dense1Rotations.begin(), dense1Rotations.end());
    allRotations.insert(allRotations.end(), dense2Rotations.begin(), dense2Rotations.end());

    // Remove duplicates
    std::sort(allRotations.begin(), allRotations.end());
    allRotations.erase(std::unique(allRotations.begin(), allRotations.end()), allRotations.end());

    std::cout << "  Total unique rotation keys needed: " << allRotations.size() << std::endl;
    std::cout << "  Generating rotation keys..." << std::endl;

    cc->EvalRotateKeyGen(keys.secretKey, allRotations);

    // Encode weight diagonals as plaintexts (not encrypted)
    // Standard approach for neural network inference: encrypted input + plaintext weights
    // TESTING: Use raw diagonals directly instead of encoding
    // auto ptConvDiags = MakeCKKSPackedPlaintextVectors(cc, convDiagonals);
    // auto ptDense1Diags = MakeCKKSPackedPlaintextVectors(cc, dense1Diagonals);
    // auto ptDense2Diags = MakeCKKSPackedPlaintextVectors(cc, dense2Diagonals);

    std::cout << "Weight preparation time: " << TOC(t) << " ms" << std::endl;

    // ========== Encrypt Input ==========
    std::cout << "\nEncrypting input..." << std::endl;
    TIC(t);
    std::vector<double> flatInput = EncodeMatrix(mnistInput, 784*2);
    auto ptInput = cc->MakeCKKSPackedPlaintext(flatInput);
    auto ctInput = cc->Encrypt(keys.publicKey, ptInput);
    std::cout << "Input encryption time: " << TOC(t) << " ms" << std::endl;

    // ========== CLEARTEXT FORWARD PASS FOR VALIDATION ==========
    std::vector<double> clearConv, clearAct1, clearDense1, clearAct2, clearDense2;

    if (enableValidation) {
        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "Computing cleartext reference values..." << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        // Cleartext Conv
        auto clearConv_3D = CleartextConv2D(mnistInput, convKernel, convStride, convPadding, &convBias);
        clearConv = CleartextFlatten(clearConv_3D);
        std::cout << "  Cleartext Conv output size: " << clearConv.size() << std::endl;

        // Cleartext Activation1
        clearAct1 = CleartextActivation(clearConv, activationType, scale1);
        std::cout << "  Cleartext Activation1 output size: " << clearAct1.size() << std::endl;

        // Cleartext Dense1
        clearDense1 = CleartextDense(clearAct1, dense1Weights, &dense1Bias);
        std::cout << "  Cleartext Dense1 output size: " << clearDense1.size() << std::endl;

        // Cleartext Activation2
        clearAct2 = CleartextActivation(clearDense1, activationType, scale2);
        std::cout << "  Cleartext Activation2 output size: " << clearAct2.size() << std::endl;

        // Cleartext Dense2 (final output)
        clearDense2 = CleartextDense(clearAct2, dense2Weights, &dense2Bias);
        std::cout << "  Cleartext Dense2 (final) output size: " << clearDense2.size() << std::endl;

        std::cout << "Cleartext reference computation complete!" << std::endl;
    }

    // ========== Encrypted Inference ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Starting encrypted inference..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Layer 1: Convolution
    std::cout << "\n[Layer 1] Convolution (28x28x1 -> 12x12x5";
    if (output_gap > 1) {
        uint32_t superChannels = (convOutputChannels + output_gap * output_gap - 1) / (output_gap * output_gap);
        std::cout << " multiplexed to " << superChannels << "x" << (convOutputHeight * output_gap) << "x" << (convOutputWidth * output_gap);
    }
    std::cout << ")..." << std::endl;

    // Prepare bias vector
    uint32_t multiplexedFlattenedSize = ((convOutputChannels + output_gap * output_gap - 1) / (output_gap * output_gap)) *
                                         (convOutputHeight * output_gap) * (convOutputWidth * output_gap);
    auto convBiasVec = PrepareBiasVector(convBias, multiplexedFlattenedSize, convOutputChannels,
                                         convOutputHeight * convOutputWidth, output_gap,
                                         convOutputHeight, convOutputWidth);
    auto ptConvBias = cc->MakeCKKSPackedPlaintext(convBiasVec);

    TIC(t);
    // TESTING: Pass raw diagonals instead of encoded plaintexts
    uint32_t hoistingMode = useOptimized ? 2 : 1;
    auto ctConvOut = EvalMultMatVecDiag(ctInput, convDiagonals, hoistingMode, convRotations, 0, &convNonZeros);
    ctConvOut = cc->EvalAdd(ctConvOut, ptConvBias);

    double convTime = TOC(t);
    std::cout << "  Time: " << convTime << " ms" << std::endl;
    std::cout << "  Level: " << ctConvOut->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Conv output
    if (enableValidation) {
        Plaintext ptConvResult;
        cc->Decrypt(keys.secretKey, ctConvOut, &ptConvResult);
        ptConvResult->SetLength(flattenedSize);
        std::vector<double> encConv = ptConvResult->GetRealPackedValue();
        CompareVectors(clearConv, encConv, "Conv", 1e-1);
    }

    // Layer 2: Activation1
    std::cout << "\n[Layer 2] Activation1 (" << activationName << ")..." << std::endl;
    TIC(t);
    auto ctAct1 = EvalActivation(cc, ctConvOut, activationType, keys.publicKey, multiplexedFlattenedSize, slots, scaleSignFHEW, ChebyDegree, std::floor(*std::min_element(clearConv.begin(), clearConv.end())), std::ceil(*std::max_element(clearConv.begin(), clearConv.end())), scale1);
    double act1Time = TOC(t);
    std::cout << "  Time: " << act1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctAct1->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Activation1 output
    if (enableValidation) {
        Plaintext ptAct1Result;
        cc->Decrypt(keys.secretKey, ctAct1, &ptAct1Result);
        ptAct1Result->SetLength(flattenedSize);
        std::vector<double> encAct1 = ptAct1Result->GetRealPackedValue();
        CompareVectors(clearAct1, encAct1, "Activation1", 1e-1);
    }

    // Bootstrap if needed (when levels are low)
    double bootstrap1Time = 0.0;
    uint32_t levelsRemaining1 = multDepth - ctAct1->GetLevel();
    std::cout << "\n[Bootstrap Check] " << levelsRemaining1 << " levels remaining after Activation1" << std::endl;
    if (enableBootstrapping && activationType == ActivationType::CHEBYSHEV && levelsRemaining1 <= levelsAvailableAfterBootstrap+1) {
        TIC(t);
        ctAct1 = cc->EvalBootstrap(ctAct1);
        bootstrap1Time = TOC(t);
        std::cout << "  Time: " << bootstrap1Time << " ms" << std::endl;
        std::cout << "  Levels after bootstrap: " << (multDepth - ctAct1->GetLevel()) << std::endl;
    } else {
        std::cout << "  Skipping bootstrap (sufficient levels or disabled)" << std::endl;
    }

    // Layer 3: Dense 1 (unmultiplex -> 720 -> 64)
    std::cout << "\n[Layer 3] Dense1 (unmultiplex " << multiplexedFlattenedSize << " -> " << flattenedSize << " -> " << dense1Output << ")..." << std::endl;
    auto dense1BiasVec = PrepareBiasVector(dense1Bias, dense1Output);
    auto ptDense1Bias = cc->MakeCKKSPackedPlaintext(dense1BiasVec);

    TIC(t);
    cc->EvalAddInPlace(ctAct1, cc->EvalRotate(ctAct1, -dense1Cols));
    // TESTING: Pass raw diagonals instead of encoded plaintexts
    auto ctDense1Out = EvalMultMatVecDiag(ctAct1, dense1Diagonals, hoistingMode, dense1Rotations, 0, &dense1NonZeros);
    ctDense1Out = cc->EvalAdd(ctDense1Out, ptDense1Bias);

    double dense1Time = TOC(t);
    std::cout << "  Time: " << dense1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctDense1Out->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Dense1 output
    std::vector<double> encDense1;
    if (enableValidation) {
        Plaintext ptDense1Result;
        cc->Decrypt(keys.secretKey, ctDense1Out, &ptDense1Result);
        ptDense1Result->SetLength(dense1Output);
        encDense1 = ptDense1Result->GetRealPackedValue();
        CompareVectors(clearDense1, encDense1, "Dense1", 1e-1);
    }

    // Layer 4: Activation2
    std::cout << "\n[Layer 4] Activation2 (" << activationName << ")..." << std::endl;
    TIC(t);
    auto ctAct2 = EvalActivation(cc, ctDense1Out, activationType, keys.publicKey, 64, slots, scaleSignFHEW, ChebyDegree, std::floor(*std::min_element(encDense1.begin(), encDense1.end())), std::ceil(*std::max_element(encDense1.begin(), encDense1.end())), scale2);
    double act2Time = TOC(t);
    std::cout << "  Time: " << act2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctAct2->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Activation2 output
    if (enableValidation) {
        Plaintext ptAct2Result;
        cc->Decrypt(keys.secretKey, ctAct2, &ptAct2Result);
        ptAct2Result->SetLength(dense1Output);
        std::vector<double> encAct2 = ptAct2Result->GetRealPackedValue();
        CompareVectors(clearAct2, encAct2, "Activation2", 1e-1);
    }

    // Bootstrap if needed (when levels are low)
    double bootstrap2Time = 0.0;
    uint32_t levelsRemaining2 = multDepth - ctAct2->GetLevel();
    std::cout << "\n[Bootstrap Check] " << levelsRemaining2 << " levels remaining after Activation2" << std::endl;
    if (enableBootstrapping && activationType == ActivationType::CHEBYSHEV && levelsRemaining2 <= levelsAvailableAfterBootstrap+1) {
        TIC(t);
        ctAct2 = cc->EvalBootstrap(ctAct2);
        bootstrap2Time = TOC(t);
        std::cout << "  Time: " << bootstrap2Time << " ms" << std::endl;
        std::cout << "  Levels after bootstrap: " << (multDepth - ctAct2->GetLevel()) << std::endl;
    } else {
        std::cout << "  Skipping bootstrap (sufficient levels or disabled)" << std::endl;
    }

    // Layer 5: Dense 2 (64 -> 10)
    std::cout << "\n[Layer 5] Dense2 (64 -> 10)..." << std::endl;
    auto dense2BiasVec = PrepareBiasVector(dense2Bias, dense2Output);
    auto ptDense2Bias = cc->MakeCKKSPackedPlaintext(dense2BiasVec);

    TIC(t);
    cc->EvalAddInPlace(ctAct2, cc->EvalRotate(ctAct2, -dense2Cols));
    // TESTING: Pass raw diagonals instead of encoded plaintexts
    auto ctOutput = EvalMultMatVecDiag(ctAct2, dense2Diagonals, hoistingMode, dense2Rotations, 0, &dense2NonZeros);
    ctOutput = cc->EvalAdd(ctOutput, ptDense2Bias);

    double dense2Time = TOC(t);
    std::cout << "  Time: " << dense2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctOutput->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Dense2 (final) output
    if (enableValidation) {
        Plaintext ptDense2Result;
        cc->Decrypt(keys.secretKey, ctOutput, &ptDense2Result);
        ptDense2Result->SetLength(dense2Output);
        std::vector<double> encDense2 = ptDense2Result->GetRealPackedValue();
        CompareVectors(clearDense2, encDense2, "Dense2 (Final)", 1e-1);
    }

    double totalInferenceTime = convTime + act1Time + bootstrap1Time + dense1Time + act2Time + bootstrap2Time + dense2Time;
    double totalBootstrapTime = bootstrap1Time + bootstrap2Time;
    std::cout << "\nTotal inference time: " << totalInferenceTime << " ms";
    if (totalBootstrapTime > 0) {
        std::cout << " (includes " << totalBootstrapTime << " ms bootstrapping)";
    }
    std::cout << std::endl;

    // ========== Decrypt and Display Results ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    TIC(t);
    Plaintext ptOutput;
    cc->Decrypt(keys.secretKey, ctOutput, &ptOutput);
    ptOutput->SetLength(dense2Output);
    std::vector<double> outputVector = ptOutput->GetRealPackedValue();
    std::cout << "Output decryption time: " << TOC(t) << " ms" << std::endl;

    std::cout << "\nOutput logits (10 classes):" << std::endl;
    for (uint32_t i = 0; i < dense2Output; i++) {
        std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(6)
                  << outputVector[i] << std::endl;
    }

    // Find predicted class
    uint32_t predictedClass = 0;
    double maxLogit = outputVector[0];
    for (uint32_t i = 1; i < dense2Output; i++) {
        if (outputVector[i] > maxLogit) {
            maxLogit = outputVector[i];
            predictedClass = i;
        }
    }

    std::cout << "\nPredicted class: " << predictedClass;
    if (trueLabel >= 0) {
        std::cout << " (True label: " << trueLabel << ")";
        if (predictedClass == static_cast<uint32_t>(trueLabel)) {
            std::cout << " ✓ CORRECT";
        } else {
            std::cout << " ✗ INCORRECT";
        }
    }
    std::cout << std::endl;
    std::cout << "Confidence: " << maxLogit << std::endl;

    // ========== Performance Summary ==========
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Performance Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::left << std::setw(32) << "Layer" << std::setw(15) << "Time (ms)" << "Level" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(32) << "Conv (28x28 -> 12x12x5)" << std::setw(15) << convTime << ctConvOut->GetLevel() << std::endl;
    std::cout << std::left << std::setw(32) << "Activation1" << std::setw(15) << act1Time << (bootstrap1Time > 0 ? "N/A" : std::to_string(ctAct1->GetLevel())) << std::endl;
    if (bootstrap1Time > 0) {
        std::cout << std::left << std::setw(32) << "  Bootstrap1" << std::setw(15) << bootstrap1Time << ctAct1->GetLevel() << std::endl;
    }
    std::cout << std::left << std::setw(32) << "Dense1 (720 -> 64)" << std::setw(15) << dense1Time << ctDense1Out->GetLevel() << std::endl;
    std::cout << std::left << std::setw(32) << "Activation2" << std::setw(15) << act2Time << (bootstrap2Time > 0 ? "N/A" : std::to_string(ctAct2->GetLevel())) << std::endl;
    if (bootstrap2Time > 0) {
        std::cout << std::left << std::setw(32) << "  Bootstrap2" << std::setw(15) << bootstrap2Time << ctAct2->GetLevel() << std::endl;
    }
    std::cout << std::left << std::setw(32) << "Dense2 (64 -> 10)" << std::setw(15) << dense2Time << ctOutput->GetLevel() << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(32) << "Total Inference" << std::setw(15) << totalInferenceTime << std::endl;
    if (totalBootstrapTime > 0) {
        std::cout << std::left << std::setw(32) << "  (Bootstrapping only)" << std::setw(15) << totalBootstrapTime << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\n✓ MNIST LoLa Inference Complete (" << activationName << ")!" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        int sampleIndex = 8;
        ActivationType activationType = ActivationType::SCHEME_SWITCH;
        uint32_t chebyDegree = 119;
        uint32_t chebyMultDepth = 8;
        bool useOptimized = false;
        bool enableValidation = true;

        // Parse command line arguments
        if (argc > 1) {
            sampleIndex = std::atoi(argv[1]);
            if (sampleIndex < 0 || sampleIndex > 9999) {
                std::cerr << "Error: Sample index must be between 0 and 9999" << std::endl;
                std::cerr << "Usage: " << argv[0] << " [sample_index] [activation_type] [cheby_degree] [optimize]" << std::endl;
                std::cerr << "  or:    " << argv[0] << " [sample_index] [activation_type] [optimize]  (for non-cheby)" << std::endl;
                std::cerr << "  activation_type: scheme (default), cheby, square" << std::endl;
                std::cerr << "  cheby_degree: Chebyshev degree (3-261631, default: 119, only for 'cheby' activation)" << std::endl;
                std::cerr << "  optimize: 1 to enable optimization (output_gap, hoisting mode 2), 0 to disable (default)" << std::endl;
                return 1;
            }
        }

        if (argc > 2) {
            std::string activationStr = argv[2];
            if (activationStr == "scheme") {
                activationType = ActivationType::SCHEME_SWITCH;
            } else if (activationStr == "cheby") {
                activationType = ActivationType::CHEBYSHEV;
            } else if (activationStr == "square") {
                activationType = ActivationType::SQUARE;
            } else {
                std::cerr << "Error: Unknown activation type '" << activationStr << "'" << std::endl;
                std::cerr << "Valid options: scheme, cheby, square" << std::endl;
                return 1;
            }
        }

        if (argc > 3) {
            if (activationType == ActivationType::CHEBYSHEV) {
                // For cheby: argv[3] is degree, argv[4] is optimize
                chebyDegree = std::atoi(argv[3]);
                if (chebyDegree < 3 || chebyDegree > 261631) {
                    std::cerr << "Error: Chebyshev degree must be between 3 and 261631" << std::endl;
                    return 1;
                }
                chebyMultDepth = GetChebyDepthFromDegree(chebyDegree);

                if (argc > 4) {
                    useOptimized = (std::atoi(argv[4]) != 0);
                }
            } else {
                // For non-cheby: argv[3] is optimize
                useOptimized = (std::atoi(argv[3]) != 0);
            }
        }

        MNISTLoLaInference(sampleIndex, activationType, chebyDegree, chebyMultDepth, useOptimized, enableValidation);
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
