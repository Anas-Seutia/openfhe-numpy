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

// ========== DEBUG MODE ==========
// Set to true to decrypt and print intermediate values after each layer
constexpr bool DEBUG_MODE = true;

// ========== ACTIVATION FUNCTION TYPE ==========
enum class ActivationType {
    SCHEME_SWITCH,  // CKKS-FHEW-CKKS scheme switching
    CHEBYSHEV,      // Chebyshev polynomial approximation
    SQUARE          // Square approximation (x^2 based)
};

/**
 * @brief MNIST LeNet-5 Network Architecture (Scheme Switching for ReLU)
 *
 * Network: Conv1 -> ReLU -> AvgPool1 -> Conv2 -> ReLU -> AvgPool2 -> Dense1 -> ReLU -> Dense2 -> ReLU -> Dense3
 * - Input: 28x28 MNIST image (1 channel)
 * - Conv1: 5x5 kernel, 6 output channels, stride=1, padding=0 -> 24x24x6
 * - ReLU: Scheme switching (CKKS-FHEW-CKKS)
 * - AvgPool1: 2x2 kernel, stride=2 (as Conv 6->6, 2x2, stride=2) -> 12x12x6
 * - Conv2: 5x5 kernel, 16 output channels, stride=1, padding=0 -> 8x8x16
 * - ReLU: Scheme switching
 * - AvgPool2: 2x2 kernel, stride=2 (as Conv 16->16, 2x2, stride=2) -> 4x4x16 = 256
 * - Dense1: 256 -> 120 neurons
 * - ReLU: Scheme switching
 * - Dense2: 120 -> 84 neurons
 * - ReLU: Scheme switching
 * - Dense3: 84 -> 10 neurons (output classes)
 */

/**
 * @brief Print min/max bounds of decrypted vector
 */
void PrintBounds(const std::vector<double>& vec, const std::string& name) {
    double minVal = *std::min_element(vec.begin(), vec.end());
    double maxVal = *std::max_element(vec.begin(), vec.end());
    std::cout << "  " << name << " bounds: [" << std::fixed << std::setprecision(6)
              << minVal << ", " << maxVal << "]" << std::endl;
}

/**
 * @brief Decrypt and print first N values for debugging
 */
void PrintDebugValues(
    CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct,
    const PrivateKey<DCRTPoly>& secretKey,
    const std::string& name,
    size_t numValues = 10,
    size_t totalElements = 0
) {
    if (!DEBUG_MODE) return;

    Plaintext ptxt;
    cc->Decrypt(secretKey, ct, &ptxt);
    if (totalElements > 0) {
        ptxt->SetLength(totalElements);
    }
    std::vector<double> values = ptxt->GetRealPackedValue();

    std::cout << "  [DEBUG] " << name << " (first " << std::min(numValues, values.size()) << " values):" << std::endl;
    std::cout << "    ";
    for (size_t i = 0; i < std::min(numValues, values.size()); i++) {
        std::cout << std::fixed << std::setprecision(4) << values[i];
        if (i < std::min(numValues, values.size()) - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Also print bounds
    if (totalElements > 0 && values.size() > totalElements) {
        values.resize(totalElements);
    }
    PrintBounds(values, name);
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
 * @brief Helper function to prepare bias vector for addition to ciphertext
 * @param bias Bias vector (one value per output channel/neuron)
 * @param outputSize Total size of output (channels * height * width for conv, or neurons for dense)
 * @param channels Number of channels (for conv layers)
 * @param spatialSize Height * width (for conv layers, set to 1 for dense)
 */
std::vector<double> PrepareBiasVector(
    const std::vector<double>& bias,
    uint32_t outputSize,
    uint32_t channels = 1,
    uint32_t spatialSize = 1
) {
    std::vector<double> biasVec(outputSize, 0.0);

    if (spatialSize == 1) {
        // Dense layer: bias[i] goes to position i
        for (size_t i = 0; i < bias.size() && i < outputSize; i++) {
            biasVec[i] = bias[i];
        }
    } else {
        // Conv layer: bias[c] is replicated across all spatial positions of channel c
        for (uint32_t c = 0; c < channels; c++) {
            for (uint32_t s = 0; s < spatialSize; s++) {
                biasVec[c * spatialSize + s] = bias[c];
            }
        }
    }

    return biasVec;
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

    // ReLU(x) = -(x < 0) + 1
    // Step 1: Compute comparison result (x < 0)
    auto ctComparison = cc->EvalCompareSchemeSwitching(ct, ctZero, NextPow2(numSlots), totalSlots, 0, scaleSign);

    // Step 2: Multiply input by comparison result to get ReLU
    // The comparison returns 1 if x > 0, 0 otherwise
    // We need to invert: (1 - comparison) to get mask
    auto ctReLU = cc->EvalMult(ct, cc->EvalAdd(cc->EvalMult(ctComparison, -1), 1));

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

// ========== CLEARTEXT VALIDATION FUNCTIONS (REMOVE LATER) ==========

/**
 * @brief Cleartext 2D convolution for validation
 */
std::vector<std::vector<std::vector<double>>> CleartextConv2D(
    const std::vector<std::vector<std::vector<double>>>& input,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
    uint32_t stride = 1,
    uint32_t padding = 0,
    const std::vector<double>* bias = nullptr
) {
    uint32_t in_channels = input.size();
    uint32_t input_height = input[0].size();
    uint32_t input_width = input[0][0].size();
    uint32_t out_channels = kernel.size();
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
                                sum += input[ic][ih][iw] * kernel[oc][ic][kh][kw];
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
 * @brief Cleartext average pooling for validation
 */
std::vector<std::vector<std::vector<double>>> CleartextAvgPool2D(
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
                        sum += input[c][oh * stride + kh][ow * stride + kw];
                    }
                }
                output[c][oh][ow] = sum / pool_size;
            }
        }
    }
    return output;
}

/**
 * @brief Flatten 3D to 1D for validation
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
 * @brief Cleartext dense layer for validation
 */
std::vector<double> CleartextDense(
    const std::vector<double>& input,
    const std::vector<std::vector<double>>& weights,
    const std::vector<double>* bias = nullptr
) {
    std::vector<double> output(weights.size(), 0.0);
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < input.size(); j++) {
            output[i] += weights[i][j] * input[j];
        }
        if (bias) {
            output[i] += (*bias)[i];
        }
    }
    return output;
}

/**
 * @brief Cleartext ReLU for validation
 */
std::vector<double> CleartextReLU(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
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

/**
 * @brief Compare two vectors and report detailed statistics
 */
void CompareVectors(const std::vector<double>& cleartext, const std::vector<double>& encrypted,
                    const std::string& layerName, double threshold = 1e-2) {
    size_t len = std::min(cleartext.size(), encrypted.size());
    double maxError = 0.0;
    double sumError = 0.0;
    size_t errorCount = 0;

    for (size_t i = 0; i < len; i++) {
        double error = std::abs(cleartext[i] - encrypted[i]);
        sumError += error;
        if (error > maxError) maxError = error;
        if (error > threshold) errorCount++;
    }

    double avgError = sumError / len;
    std::cout << "  [VALIDATION] " << layerName << ":" << std::endl;
    std::cout << "    Max error: " << std::fixed << std::setprecision(6) << maxError << std::endl;
    std::cout << "    Avg error: " << avgError << std::endl;
    std::cout << "    Elements with error > " << threshold << ": " << errorCount << " / " << len;

    if (errorCount == 0) {
        std::cout << " ✓ PASS" << std::endl;
    } else {
        std::cout << " ✗ FAIL" << std::endl;
        // Print first few mismatches
        std::cout << "    First mismatches:" << std::endl;
        int printed = 0;
        for (size_t i = 0; i < len && printed < 5; i++) {
            double error = std::abs(cleartext[i] - encrypted[i]);
            if (error > threshold) {
                std::cout << "      [" << i << "] cleartext: " << cleartext[i]
                         << ", encrypted: " << encrypted[i] << ", error: " << error << std::endl;
                printed++;
            }
        }
    }
}

// ========== END CLEARTEXT VALIDATION FUNCTIONS ==========

void MNISTLeNet5Inference(int sampleIndex = 8, ActivationType activationType = ActivationType::SCHEME_SWITCH) {
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

    std::cout << "  MNIST LeNet-5 Network Inference (" << activationName << ")" << std::endl;
    std::cout << "  Architecture: Conv1->Act->Pool1->Conv2->Act->Pool2->FC1->Act->FC2->Act->FC3" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // Note: Using trained weights from PyTorch model (no random seed needed)

    // ========== Network Parameters ==========
    std::cout << "LeNet-5 Architecture:" << std::endl;
    std::cout << "  Input: 28x28 MNIST image (1 channel)" << std::endl;
    std::cout << "  Conv1: 5x5 kernel, 6 output channels, stride=1 -> 24x24x6" << std::endl;
    std::cout << "  ReLU: Scheme switching (CKKS-FHEW-CKKS)" << std::endl;
    std::cout << "  AvgPool1: 2x2 kernel, stride=2 -> 12x12x6" << std::endl;
    std::cout << "  Conv2: 5x5 kernel, 16 output channels, stride=1 -> 8x8x16" << std::endl;
    std::cout << "  ReLU: Scheme switching" << std::endl;
    std::cout << "  AvgPool2: 2x2 kernel, stride=2 -> 4x4x16 = 256" << std::endl;
    std::cout << "  Dense1: 256 -> 120 neurons" << std::endl;
    std::cout << "  ReLU: Scheme switching" << std::endl;
    std::cout << "  Dense2: 120 -> 84 neurons" << std::endl;
    std::cout << "  ReLU: Scheme switching" << std::endl;
    std::cout << "  Dense3: 84 -> 10 neurons (output)" << std::endl << std::endl;

    // ========== Load MNIST Input ==========
    std::cout << "Loading MNIST test sample #" << sampleIndex << "..." << std::endl;

    // Construct path to MNIST sample
    std::string mnistDataDir = "../openfhe_numpy/cpp/data/mnist";

    // Find the file for this sample index
    std::stringstream samplePath;
    samplePath << mnistDataDir << "/mnist_" << sampleIndex << "_label_";

    // We need to find the actual file (since we don't know the label yet)
    // Try to find files matching the pattern
    std::string basePattern = samplePath.str();
    std::string actualFile = "";
    int trueLabel = -1;

    // Try labels 0-9
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

    if (actualFile.empty()) {
        throw std::runtime_error("Could not find MNIST sample #" + std::to_string(sampleIndex) +
                                 " in " + mnistDataDir + "\nRun export_mnist_sample.py first!");
    }

    // Load the MNIST image
    auto mnistInput = LoadMNISTImage(actualFile);
    std::cout << "True label: " << trueLabel << std::endl;

    // ========== Setup Crypto Context ==========
    std::cout << "\nSetting up crypto context..." << std::endl;

    ScalingTechnique scTech = FLEXIBLEAUTO;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;

    uint32_t scaleModSize = 40;
    uint32_t firstModSize = 50;
    uint32_t ringDim = 32768;
    SecurityLevel sl = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE = 25;
    uint32_t slots = 4096;
    uint32_t batchSize = slots;

    // Bootstrapping parameters
    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};
    uint32_t levelsAvailableAfterBootstrap = 12;
    uint32_t approxBootstrapDepth = FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    uint32_t multDepth = levelsAvailableAfterBootstrap + approxBootstrapDepth;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetScalingTechnique(scTech);
    parameters.SetSecurityLevel(sl);
    parameters.SetRingDim(ringDim);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetNumLargeDigits(3);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);  // Enable bootstrapping

    // Only enable scheme switching if needed
    if (activationType == ActivationType::SCHEME_SWITCH) {
        cc->Enable(SCHEMESWITCH);
    }

    std::cout << "CKKS scheme using ring dimension " << cc->GetRingDimension() << std::endl;
    std::cout << "Number of slots: " << slots << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl;
    std::cout << "Activation function: " << activationName << std::endl;

    // ========== Bootstrapping Setup ==========
    std::cout << "\nSetting up bootstrapping..." << std::endl;
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, slots);

    // ========== Key Generation ==========
    std::cout << "\nGenerating keys..." << std::endl;
    TimeVar t;
    TIC(t);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    // Setup scheme switching only if needed
    double scaleSignFHEW = 4.0;
    if (activationType == ActivationType::SCHEME_SWITCH) {
        SchSwchParams params;
        params.SetSecurityLevelCKKS(sl);
        params.SetSecurityLevelFHEW(slBin);
        params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE);
        params.SetNumSlotsCKKS(slots);
        params.SetNumValues(3456);  // Max(24*24*6=3456, 12*12*6=864, 8*8*16=1024, 256, 120, 84)

        auto privateKeyFHEW = cc->EvalSchemeSwitchingSetup(params);
        auto ccLWE = cc->GetBinCCForSchemeSwitch();
        ccLWE->BTKeyGen(privateKeyFHEW);
        cc->EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW);

        auto modulus_LWE = 1 << logQ_ccLWE;
        auto beta = ccLWE->GetBeta().ConvertToInt();
        auto pLWE = modulus_LWE / (2 * beta);
        cc->EvalCompareSwitchPrecompute(pLWE, scaleSignFHEW);
    }

    std::cout << "Key generation time: " << TOC(t) << " ms" << std::endl;

    // ========== Load Network Weights ==========
    std::cout << "\nLoading network weights from trained model..." << std::endl;

    // Select model based on activation type
    std::string weightsDir;
    if (activationType == ActivationType::SQUARE) {
        weightsDir = "../openfhe_numpy/cpp/models/lenet5_weight_square";
        std::cout << "Using square activation model (lenet5square.pt)" << std::endl;
    } else {
        weightsDir = "../openfhe_numpy/cpp/models/lenet5_weight_relu";
        std::cout << "Using ReLU model (lenet5relu.pt)" << std::endl;
    }

    LeNet5Weights trainedWeights = LoadLeNet5Weights(weightsDir);

    // Use loaded weights
    auto conv1Kernel = trainedWeights.conv1_weight;
    auto conv2Kernel = trainedWeights.conv2_weight;
    auto dense1Weights = trainedWeights.fc1_weight;
    auto dense2Weights = trainedWeights.fc2_weight;
    auto dense3Weights = trainedWeights.fc3_weight;

    uint32_t conv1OutputHeight = 24;  // (28 - 5) / 1 + 1
    uint32_t conv1OutputWidth = 24;
    uint32_t conv1OutputChannels = 6;
    uint32_t conv1FlatSize = conv1OutputHeight * conv1OutputWidth * conv1OutputChannels;  // 3456
    std::cout << "  Conv1 output: " << conv1OutputChannels << " channels, "
              << conv1OutputHeight << "x" << conv1OutputWidth << " = " << conv1FlatSize << std::endl;

    // AvgPool1: 2x2, stride=2 (implemented as Conv 6->6, 2x2, stride=2)
    std::vector<std::vector<std::vector<std::vector<double>>>> avgpool1Kernel(6);
    for (int oc = 0; oc < 6; oc++) {
        avgpool1Kernel[oc].resize(6);
        for (int ic = 0; ic < 6; ic++) {
            avgpool1Kernel[oc][ic].resize(2, std::vector<double>(2, 0.0));
            if (oc == ic) {  // Identity mapping for each channel
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        avgpool1Kernel[oc][ic][i][j] = 0.25;  // Average pooling
                    }
                }
            }
        }
    }
    uint32_t pool1OutputHeight = 12;  // 24 / 2
    uint32_t pool1OutputWidth = 12;
    uint32_t pool1OutputChannels = 6;
    uint32_t pool1FlatSize = pool1OutputHeight * pool1OutputWidth * pool1OutputChannels;  // 864
    std::cout << "  AvgPool1 output: " << pool1OutputChannels << " channels, "
              << pool1OutputHeight << "x" << pool1OutputWidth << " = " << pool1FlatSize << std::endl;

    uint32_t conv2OutputHeight = 8;  // (12 - 5) / 1 + 1
    uint32_t conv2OutputWidth = 8;
    uint32_t conv2OutputChannels = 16;
    uint32_t conv2FlatSize = conv2OutputHeight * conv2OutputWidth * conv2OutputChannels;  // 1024
    std::cout << "  Conv2 output: " << conv2OutputChannels << " channels, "
              << conv2OutputHeight << "x" << conv2OutputWidth << " = " << conv2FlatSize << std::endl;

    // AvgPool2: 2x2, stride=2 (implemented as Conv 16->16, 2x2, stride=2)
    std::vector<std::vector<std::vector<std::vector<double>>>> avgpool2Kernel(16);
    for (int oc = 0; oc < 16; oc++) {
        avgpool2Kernel[oc].resize(16);
        for (int ic = 0; ic < 16; ic++) {
            avgpool2Kernel[oc][ic].resize(2, std::vector<double>(2, 0.0));
            if (oc == ic) {
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        avgpool2Kernel[oc][ic][i][j] = 0.25;  // Average pooling
                    }
                }
            }
        }
    }
    uint32_t pool2OutputHeight = 4;  // 8 / 2
    uint32_t pool2OutputWidth = 4;
    uint32_t pool2OutputChannels = 16;
    uint32_t pool2FlatSize = pool2OutputHeight * pool2OutputWidth * pool2OutputChannels;  // 256
    std::cout << "  AvgPool2 output: " << pool2OutputChannels << " channels, "
              << pool2OutputHeight << "x" << pool2OutputWidth << " = " << pool2FlatSize << std::endl;

    // Dense layer dimensions
    uint32_t dense1Input = pool2FlatSize;  // 256
    uint32_t dense1Output = 120;
    std::cout << "  Dense1: " << dense1Input << " -> " << dense1Output << std::endl;

    uint32_t dense2Input = dense1Output;  // 120
    uint32_t dense2Output = 84;
    std::cout << "  Dense2: " << dense2Input << " -> " << dense2Output << std::endl;

    uint32_t dense3Input = dense2Output;  // 84
    uint32_t dense3Output = 10;
    std::cout << "  Dense3: " << dense3Input << " -> " << dense3Output << std::endl;

    // ========== Build Toeplitz matrices and pack into diagonals ==========
    std::cout << "\nPreparing network weights..." << std::endl;
    TIC(t);

    // Conv1 Toeplitz
    auto toeplitzConv1 = ConstructConv2DToeplitz(conv1Kernel, 28, 28, 1, 0, 1, 1, 1);
    std::vector<std::vector<double>> conv1Diagonals = PackMatDiagWise(toeplitzConv1, batchSize);
    std::size_t conv1Cols = conv1Diagonals.size();
    std::vector<int32_t> conv1Rotations = getOptimalRots(conv1Diagonals, true);
    std::cout << "  Conv1 Toeplitz: " << conv1Cols << " rows, "
              << conv1Rotations.size() << " non-zero diagonals" << std::endl;

    // AvgPool1 Toeplitz
    auto toeplitzPool1 = ConstructConv2DToeplitz(avgpool1Kernel, 24, 24, 2, 0, 1, 1, 1);
    std::vector<std::vector<double>> pool1Diagonals = PackMatDiagWise(toeplitzPool1, batchSize);
    std::size_t pool1Cols = pool1Diagonals.size();
    std::vector<int32_t> pool1Rotations = getOptimalRots(pool1Diagonals, true);
    std::cout << "  AvgPool1 Toeplitz: " << pool1Cols << " rows, "
              << pool1Rotations.size() << " rotation keys needed" << std::endl;

    // Conv2 Toeplitz
    auto toeplitzConv2 = ConstructConv2DToeplitz(conv2Kernel, 12, 12, 1, 0, 1, 1, 1);
    std::vector<std::vector<double>> conv2Diagonals = PackMatDiagWise(toeplitzConv2, batchSize);
    std::size_t conv2Cols = conv2Diagonals.size();
    std::vector<int32_t> conv2Rotations = getOptimalRots(conv2Diagonals, true);
    std::cout << "  Conv2 Toeplitz: " << conv2Cols << " rows, "
              << conv2Rotations.size() << " rotation keys needed" << std::endl;

    // AvgPool2 Toeplitz
    auto toeplitzPool2 = ConstructConv2DToeplitz(avgpool2Kernel, 8, 8, 2, 0, 1, 1, 1);
    std::vector<std::vector<double>> pool2Diagonals = PackMatDiagWise(toeplitzPool2, batchSize);
    std::size_t pool2Cols = pool2Diagonals.size();
    std::vector<int32_t> pool2Rotations = getOptimalRots(pool2Diagonals, true);
    std::cout << "  AvgPool2 Toeplitz: " << pool2Cols << " rows, "
              << pool2Rotations.size() << " rotation keys needed" << std::endl;

    // Dense layers
    std::vector<std::vector<double>> dense1Diagonals = PackMatDiagWise(dense1Weights, batchSize);
    std::size_t dense1Cols = dense1Diagonals.size();
    std::vector<int32_t> dense1Rotations = getOptimalRots(dense1Diagonals, true);
    std::cout << "  Dense1: " << dense1Cols << " rows, "
              << dense1Rotations.size() << " rotation keys needed" << std::endl;

    std::vector<std::vector<double>> dense2Diagonals = PackMatDiagWise(dense2Weights, batchSize);
    std::size_t dense2Cols = dense2Diagonals.size();
    std::vector<int32_t> dense2Rotations = getOptimalRots(dense2Diagonals, true);
    std::cout << "  Dense2: " << dense2Cols << " rows, "
              << dense2Rotations.size() << " rotation keys needed" << std::endl;

    std::vector<std::vector<double>> dense3Diagonals = PackMatDiagWise(dense3Weights, batchSize);
    std::size_t dense3Cols = dense3Diagonals.size();
    std::vector<int32_t> dense3Rotations = getOptimalRots(dense3Diagonals, true);
    std::cout << "  Dense3: " << dense3Cols << " rows, "
              << dense3Rotations.size() << " rotation keys needed" << std::endl;

    // Collect all rotation indices
    std::vector<int32_t> allRotations;
    allRotations.insert(allRotations.end(), conv1Rotations.begin(), conv1Rotations.end());
    allRotations.insert(allRotations.end(), pool1Rotations.begin(), pool1Rotations.end());
    allRotations.insert(allRotations.end(), conv2Rotations.begin(), conv2Rotations.end());
    allRotations.insert(allRotations.end(), pool2Rotations.begin(), pool2Rotations.end());
    allRotations.insert(allRotations.end(), dense1Rotations.begin(), dense1Rotations.end());
    allRotations.insert(allRotations.end(), dense2Rotations.begin(), dense2Rotations.end());
    allRotations.insert(allRotations.end(), dense3Rotations.begin(), dense3Rotations.end());

    // Remove duplicates
    std::sort(allRotations.begin(), allRotations.end());
    allRotations.erase(std::unique(allRotations.begin(), allRotations.end()), allRotations.end());

    std::cout << "  Total unique rotation keys needed: " << allRotations.size() << std::endl;
    std::cout << "  Generating rotation keys..." << std::endl;
    cc->EvalRotateKeyGen(keys.secretKey, allRotations);
    std::cout << "  Rotation key generation complete!" << std::endl;

    std::cout << "Weight preparation time: " << TOC(t) << " ms" << std::endl;

    // ========== Encrypt Input ==========
    std::cout << "\nEncrypting input..." << std::endl;
    TIC(t);
    std::vector<double> flatInput = EncodeMatrix(mnistInput, 784*5);
    auto ptInput = cc->MakeCKKSPackedPlaintext(flatInput);
    auto ctInput = cc->Encrypt(keys.publicKey, ptInput);
    std::cout << "Input encryption time: " << TOC(t) << " ms" << std::endl;
    std::cout << "Initial ciphertext level: " << ctInput->GetLevel() << std::endl;

    // ========== CLEARTEXT FORWARD PASS FOR VALIDATION (REMOVE LATER) ==========
    std::cout << "\nRunning cleartext forward pass for validation..." << std::endl;

    // Prepare 3D input for cleartext (1 channel, 28x28)
    std::vector<std::vector<std::vector<double>>> mnistInput3D(1,
        std::vector<std::vector<double>>(28, std::vector<double>(28)));
    for (int h = 0; h < 28; h++) {
        for (int w = 0; w < 28; w++) {
            mnistInput3D[0][h][w] = mnistInput[h][w];
        }
    }

    // Cleartext Conv1
    auto clearConv1_3D = CleartextConv2D(mnistInput3D, conv1Kernel, 1, 0, &trainedWeights.conv1_bias);
    auto clearConv1 = CleartextFlatten(clearConv1_3D);

    // Cleartext Activation1
    auto clearReLU1 = CleartextActivation(clearConv1, activationType, trainedWeights.scale1);

    // Reshape for pooling
    std::vector<std::vector<std::vector<double>>> clearReLU1_3D(6,
        std::vector<std::vector<double>>(24, std::vector<double>(24)));
    for (int c = 0; c < 6; c++) {
        for (int h = 0; h < 24; h++) {
            for (int w = 0; w < 24; w++) {
                clearReLU1_3D[c][h][w] = clearReLU1[c * 24 * 24 + h * 24 + w];
            }
        }
    }

    // Cleartext Pool1
    auto clearPool1_3D = CleartextAvgPool2D(clearReLU1_3D, 2, 2);
    auto clearPool1 = CleartextFlatten(clearPool1_3D);

    // Cleartext Conv2
    auto clearConv2_3D = CleartextConv2D(clearPool1_3D, conv2Kernel, 1, 0, &trainedWeights.conv2_bias);
    auto clearConv2 = CleartextFlatten(clearConv2_3D);

    // Cleartext Activation2
    auto clearReLU2 = CleartextActivation(clearConv2, activationType, trainedWeights.scale2);

    // Reshape for pooling
    std::vector<std::vector<std::vector<double>>> clearReLU2_3D(16,
        std::vector<std::vector<double>>(8, std::vector<double>(8)));
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 8; h++) {
            for (int w = 0; w < 8; w++) {
                clearReLU2_3D[c][h][w] = clearReLU2[c * 8 * 8 + h * 8 + w];
            }
        }
    }

    // Cleartext Pool2
    auto clearPool2_3D = CleartextAvgPool2D(clearReLU2_3D, 2, 2);
    auto clearPool2 = CleartextFlatten(clearPool2_3D);

    // Cleartext Dense1
    auto clearDense1 = CleartextDense(clearPool2, dense1Weights, &trainedWeights.fc1_bias);

    // Cleartext Activation3
    auto clearReLU3 = CleartextActivation(clearDense1, activationType, trainedWeights.scale3);

    // Cleartext Dense2
    auto clearDense2 = CleartextDense(clearReLU3, dense2Weights, &trainedWeights.fc2_bias);

    // Cleartext Activation4
    auto clearReLU4 = CleartextActivation(clearDense2, activationType, trainedWeights.scale4);

    // Cleartext Dense3
    auto clearDense3 = CleartextDense(clearReLU4, dense3Weights, &trainedWeights.fc3_bias);

    std::cout << "Cleartext forward pass complete!" << std::endl;
    // ========== END CLEARTEXT FORWARD PASS ==========

    // ========== Forward Pass ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Starting LeNet-5 encrypted inference..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Layer 1: Conv1
    std::cout << "\n[Layer 1] Conv1 (28x28x1 -> 24x24x6)..." << std::endl;
    auto ptConv1Diags = MakeCKKSPackedPlaintextVectors(cc, conv1Diagonals);
    auto conv1BiasVec = PrepareBiasVector(trainedWeights.conv1_bias, conv1FlatSize, conv1OutputChannels, conv1OutputHeight * conv1OutputWidth);
    auto ptConv1Bias = cc->MakeCKKSPackedPlaintext(conv1BiasVec);

    TIC(t);
    // ctInput = cc->EvalRotate(ctInput, -conv1Cols);
    auto ctConv1 = EvalMultMatVecDiag(ctInput, ptConv1Diags, 2, conv1Rotations);

    // Add bias
    ctConv1 = cc->EvalAdd(ctConv1, ptConv1Bias);

    double conv1Time = TOC(t);
    std::cout << "  Time: " << conv1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctConv1->GetLevel() << std::endl;

    ptConv1Diags.clear();
    ptConv1Diags.shrink_to_fit();

    // Validate Conv1
    Plaintext ptConv1Result;
    cc->Decrypt(keys.secretKey, ctConv1, &ptConv1Result);
    ptConv1Result->SetLength(conv1FlatSize);
    std::vector<double> encConv1 = ptConv1Result->GetRealPackedValue();

    // Print bounds BEFORE activation (input to ReLU)
    std::cout << "  [PRE-ACTIVATION BOUNDS]" << std::endl;
    PrintBounds(clearConv1, "    Conv1 output (cleartext)");
    PrintBounds(encConv1, "    Conv1 output (encrypted)");

    CompareVectors(clearConv1, encConv1, "Conv1", 1e-1);

    // Layer 2: Activation1
    std::cout << "\n[Layer 2] Activation1 (" << activationName << ")..." << std::endl;
    TIC(t);
    auto ctReLU1 = EvalActivation(cc, ctConv1, activationType, keys.publicKey, conv1FlatSize, slots, scaleSignFHEW, 5, -1272.325288, 861.832868, trainedWeights.scale1);
    double relu1Time = TOC(t);
    std::cout << "  Time: " << relu1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU1->GetLevel() << std::endl;

    // Validate ReLU1
    Plaintext ptReLU1Result;
    cc->Decrypt(keys.secretKey, ctReLU1, &ptReLU1Result);
    ptReLU1Result->SetLength(conv1FlatSize);
    std::vector<double> encReLU1 = ptReLU1Result->GetRealPackedValue();

    // Print bounds for Chebyshev tuning
    PrintBounds(clearReLU1, "Activation1 (cleartext)");
    PrintBounds(encReLU1, "Activation1 (encrypted)");

    CompareVectors(clearReLU1, encReLU1, "ReLU1", 1e-1);

    auto ptPool1Diags = MakeCKKSPackedPlaintextVectors(cc, pool1Diagonals);

    // Layer 3: AvgPool1
    std::cout << "\n[Layer 3] AvgPool1 (24x24x6 -> 12x12x6)..." << std::endl;
    TIC(t);
    // cc->EvalAddInPlace(ctReLU1, cc->EvalRotate(ctReLU1, -pool1Cols));
    auto ctPool1 = EvalMultMatVecDiag(ctReLU1, ptPool1Diags, 2, pool1Rotations);
    double pool1Time = TOC(t);
    std::cout << "  Time: " << pool1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctPool1->GetLevel() << std::endl;

    // Validate Pool1
    Plaintext ptPool1Result;
    cc->Decrypt(keys.secretKey, ctPool1, &ptPool1Result);
    ptPool1Result->SetLength(pool1FlatSize);
    std::vector<double> encPool1 = ptPool1Result->GetRealPackedValue();
    CompareVectors(clearPool1, encPool1, "AvgPool1", 1e-1);

    ptPool1Diags.clear();
    ptPool1Diags.shrink_to_fit();

    // Bootstrap after Pool1 if needed (only when levels are low)
    double bootstrap1Time = 0.0;
    uint32_t levelsRemaining1 = multDepth - ctPool1->GetLevel();
    std::cout << "\n[Bootstrap Check] After AvgPool1: " << levelsRemaining1 << " levels remaining" << std::endl;
    if (levelsRemaining1 < 5) {
        std::cout << "  Bootstrapping needed (< 5 levels remaining)..." << std::endl;
        TIC(t);
        ctPool1 = cc->EvalBootstrap(ctPool1);
        bootstrap1Time = TOC(t);
        std::cout << "  Bootstrapping time: " << bootstrap1Time << " ms" << std::endl;
        std::cout << "  Levels after bootstrap: " << (multDepth - ctPool1->GetLevel()) << std::endl;
    } else {
        std::cout << "  Skipping bootstrap (sufficient levels available)" << std::endl;
    }

    auto ptConv2Diags = MakeCKKSPackedPlaintextVectors(cc, conv2Diagonals);
    auto conv2BiasVec = PrepareBiasVector(trainedWeights.conv2_bias, conv2FlatSize, conv2OutputChannels, conv2OutputHeight * conv2OutputWidth);
    auto ptConv2Bias = cc->MakeCKKSPackedPlaintext(conv2BiasVec);

    // Layer 4: Conv2
    std::cout << "\n[Layer 4] Conv2 (12x12x6 -> 8x8x16)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctPool1, cc->EvalRotate(ctPool1, -conv2Cols));
    cc->EvalAddInPlace(ctPool1, cc->EvalRotate(cc->EvalRotate(ctPool1, -conv2Cols), -conv2Cols));
    auto ctConv2 = EvalMultMatVecDiag(ctPool1, ptConv2Diags, 2, conv2Rotations);

    // Add bias
    ctConv2 = cc->EvalAdd(ctConv2, ptConv2Bias);

    double conv2Time = TOC(t);
    std::cout << "  Time: " << conv2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctConv2->GetLevel() << std::endl;

    // Validate Conv2
    Plaintext ptConv2Result;
    cc->Decrypt(keys.secretKey, ctConv2, &ptConv2Result);
    ptConv2Result->SetLength(conv2FlatSize);
    std::vector<double> encConv2 = ptConv2Result->GetRealPackedValue();

    // Print bounds BEFORE activation
    std::cout << "  [PRE-ACTIVATION BOUNDS]" << std::endl;
    PrintBounds(clearConv2, "    Conv2 output (cleartext)");
    PrintBounds(encConv2, "    Conv2 output (encrypted)");

    CompareVectors(clearConv2, encConv2, "Conv2", 1e-1);

    ptConv2Diags.clear();
    ptConv2Diags.shrink_to_fit();

    // Layer 5: Activation2
    std::cout << "\n[Layer 5] Activation2 (" << activationName << ")..." << std::endl;
    TIC(t);
    auto ctReLU2 = EvalActivation(cc, ctConv2, activationType, keys.publicKey, conv2FlatSize, slots, scaleSignFHEW, 5, -1657.532905, 1155.935255, trainedWeights.scale2);
    double relu2Time = TOC(t);
    std::cout << "  Time: " << relu2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU2->GetLevel() << std::endl;

    // Validate ReLU2
    Plaintext ptReLU2Result;
    cc->Decrypt(keys.secretKey, ctReLU2, &ptReLU2Result);
    ptReLU2Result->SetLength(conv2FlatSize);
    std::vector<double> encReLU2 = ptReLU2Result->GetRealPackedValue();

    // Print bounds for Chebyshev tuning
    PrintBounds(clearReLU2, "Activation2 (cleartext)");
    PrintBounds(encReLU2, "Activation2 (encrypted)");

    CompareVectors(clearReLU2, encReLU2, "ReLU2", 1e-1);

    auto ptPool2Diags = MakeCKKSPackedPlaintextVectors(cc, pool2Diagonals);

    // Layer 6: AvgPool2
    std::cout << "\n[Layer 6] AvgPool2 (8x8x16 -> 4x4x16)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU2, cc->EvalRotate(ctReLU2, -pool2Cols));
    auto ctPool2 = EvalMultMatVecDiag(ctReLU2, ptPool2Diags, 2, pool2Rotations);
    double pool2Time = TOC(t);
    std::cout << "  Time: " << pool2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctPool2->GetLevel() << std::endl;

    // Validate Pool2
    Plaintext ptPool2Result;
    cc->Decrypt(keys.secretKey, ctPool2, &ptPool2Result);
    ptPool2Result->SetLength(pool2FlatSize);
    std::vector<double> encPool2 = ptPool2Result->GetRealPackedValue();
    CompareVectors(clearPool2, encPool2, "AvgPool2", 1e-1);

    ptPool2Diags.clear();
    ptPool2Diags.shrink_to_fit();

    // Bootstrap after Pool2 if needed (only when levels are low)
    double bootstrap2Time = 0.0;
    uint32_t levelsRemaining2 = multDepth - ctPool2->GetLevel();
    std::cout << "\n[Bootstrap Check] After AvgPool2: " << levelsRemaining2 << " levels remaining" << std::endl;
    if (levelsRemaining2 < 5) {
        std::cout << "  Bootstrapping needed (< 5 levels remaining)..." << std::endl;
        TIC(t);
        ctPool2 = cc->EvalBootstrap(ctPool2);
        bootstrap2Time = TOC(t);
        std::cout << "  Bootstrapping time: " << bootstrap2Time << " ms" << std::endl;
        std::cout << "  Levels after bootstrap: " << (multDepth - ctPool2->GetLevel()) << std::endl;
    } else {
        std::cout << "  Skipping bootstrap (sufficient levels available)" << std::endl;
    }

    auto ptDense1Diags = MakeCKKSPackedPlaintextVectors(cc, dense1Diagonals);
    auto dense1BiasVec = PrepareBiasVector(trainedWeights.fc1_bias, dense1Output);
    auto ptDense1Bias = cc->MakeCKKSPackedPlaintext(dense1BiasVec);

    // Layer 7: Dense1
    std::cout << "\n[Layer 7] Dense1 (256 -> 120)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctPool2, cc->EvalRotate(ctPool2, -dense1Cols));
    auto ctDense1 = EvalMultMatVecDiag(ctPool2, ptDense1Diags, 2, dense1Rotations);

    // Add bias
    ctDense1 = cc->EvalAdd(ctDense1, ptDense1Bias);

    double dense1Time = TOC(t);
    std::cout << "  Time: " << dense1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctDense1->GetLevel() << std::endl;

    // VALIDATION: Dense1
    Plaintext ptDense1Result;
    cc->Decrypt(keys.secretKey, ctDense1, &ptDense1Result);
    ptDense1Result->SetLength(dense1Output);
    std::vector<double> encDense1 = ptDense1Result->GetRealPackedValue();

    // Print bounds BEFORE activation
    std::cout << "  [PRE-ACTIVATION BOUNDS]" << std::endl;
    PrintBounds(clearDense1, "    Dense1 output (cleartext)");
    PrintBounds(encDense1, "    Dense1 output (encrypted)");

    CompareVectors(clearDense1, encDense1, "Dense1", 1e-1);

    ptDense1Diags.clear();
    ptDense1Diags.shrink_to_fit();

    // Layer 8: Activation3
    std::cout << "\n[Layer 8] Activation3 (" << activationName << ")..." << std::endl;
    TIC(t);
    auto ctReLU3 = EvalActivation(cc, ctDense1, activationType, keys.publicKey, dense1Output, slots, scaleSignFHEW, 5, -1147.812610, 677.169142, trainedWeights.scale3);
    double relu3Time = TOC(t);
    std::cout << "  Time: " << relu3Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU3->GetLevel() << std::endl;

    // VALIDATION: ReLU3
    Plaintext ptReLU3Result;
    cc->Decrypt(keys.secretKey, ctReLU3, &ptReLU3Result);
    ptReLU3Result->SetLength(dense1Output);
    std::vector<double> encReLU3 = ptReLU3Result->GetRealPackedValue();

    // Print bounds for Chebyshev tuning
    PrintBounds(clearReLU3, "Activation3 (cleartext)");
    PrintBounds(encReLU3, "Activation3 (encrypted)");

    CompareVectors(clearReLU3, encReLU3, "ReLU3", 1e-1);

    // Bootstrap after ReLU3 if needed (only when levels are low)
    double bootstrap3Time = 0.0;
    uint32_t levelsRemaining3 = multDepth - ctReLU3->GetLevel();
    std::cout << "\n[Bootstrap Check] After Activation3: " << levelsRemaining3 << " levels remaining" << std::endl;
    if (levelsRemaining3 < 5) {
        std::cout << "  Bootstrapping needed (< 5 levels remaining)..." << std::endl;
        TIC(t);
        ctReLU3 = cc->EvalBootstrap(ctReLU3);
        bootstrap3Time = TOC(t);
        std::cout << "  Bootstrapping time: " << bootstrap3Time << " ms" << std::endl;
        std::cout << "  Levels after bootstrap: " << (multDepth - ctReLU3->GetLevel()) << std::endl;
    } else {
        std::cout << "  Skipping bootstrap (sufficient levels available)" << std::endl;
    }

    auto ptDense2Diags = MakeCKKSPackedPlaintextVectors(cc, dense2Diagonals);
    auto dense2BiasVec = PrepareBiasVector(trainedWeights.fc2_bias, dense2Output);
    auto ptDense2Bias = cc->MakeCKKSPackedPlaintext(dense2BiasVec);

    // Layer 9: Dense2
    std::cout << "\n[Layer 9] Dense2 (120 -> 84)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU3, cc->EvalRotate(ctReLU3, -dense2Cols));
    auto ctDense2 = EvalMultMatVecDiag(ctReLU3, ptDense2Diags, 2, dense2Rotations);

    // Add bias
    ctDense2 = cc->EvalAdd(ctDense2, ptDense2Bias);

    double dense2Time = TOC(t);
    std::cout << "  Time: " << dense2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctDense2->GetLevel() << std::endl;

    // VALIDATION: Dense2
    Plaintext ptDense2Result;
    cc->Decrypt(keys.secretKey, ctDense2, &ptDense2Result);
    ptDense2Result->SetLength(dense2Output);
    std::vector<double> encDense2 = ptDense2Result->GetRealPackedValue();

    // Print bounds BEFORE activation
    std::cout << "  [PRE-ACTIVATION BOUNDS]" << std::endl;
    PrintBounds(clearDense2, "    Dense2 output (cleartext)");
    PrintBounds(encDense2, "    Dense2 output (encrypted)");

    CompareVectors(clearDense2, encDense2, "Dense2", 1e-1);

    ptDense2Diags.clear();
    ptDense2Diags.shrink_to_fit();

    // DEBUG: Print Dense2 values at indices that will fail in ReLU4
    if (DEBUG_MODE) {
        std::cout << "  [DEBUG] Dense2 values at future-problematic indices [34, 44, 68, 70, 73]:" << std::endl;
        std::vector<size_t> problemIndices = {34, 44, 68, 70, 73};
        for (size_t idx : problemIndices) {
            if (idx < encDense2.size()) {
                std::cout << "    [" << idx << "] cleartext=" << std::fixed << std::setprecision(4)
                          << clearDense2[idx] << ", encrypted=" << encDense2[idx] << std::endl;
            }
        }
    }

    // Layer 10: Activation4
    std::cout << "\n[Layer 10] Activation4 (" << activationName << ")..." << std::endl;
    TIC(t);
    auto ctReLU4 = EvalActivation(cc, ctDense2, activationType, keys.publicKey, dense2Output, slots, scaleSignFHEW, 5, -318.306115, 331.310290, trainedWeights.scale4);
    double relu4Time = TOC(t);
    std::cout << "  Time: " << relu4Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU4->GetLevel() << std::endl;

    // VALIDATION: ReLU4
    Plaintext ptReLU4Result;
    cc->Decrypt(keys.secretKey, ctReLU4, &ptReLU4Result);
    ptReLU4Result->SetLength(dense2Output);
    std::vector<double> encReLU4 = ptReLU4Result->GetRealPackedValue();

    // Print bounds for Chebyshev tuning
    PrintBounds(clearReLU4, "Activation4 (cleartext)");
    PrintBounds(encReLU4, "Activation4 (encrypted)");

    CompareVectors(clearReLU4, encReLU4, "ReLU4", 1e-1);

    // Bootstrap after ReLU4 if needed (only when levels are low)
    double bootstrap4Time = 0.0;
    uint32_t levelsRemaining4 = multDepth - ctReLU4->GetLevel();
    std::cout << "\n[Bootstrap Check] After Activation4: " << levelsRemaining4 << " levels remaining" << std::endl;
    if (levelsRemaining4 < 5) {
        std::cout << "  Bootstrapping needed (< 5 levels remaining)..." << std::endl;
        TIC(t);
        ctReLU4 = cc->EvalBootstrap(ctReLU4);
        bootstrap4Time = TOC(t);
        std::cout << "  Bootstrapping time: " << bootstrap4Time << " ms" << std::endl;
        std::cout << "  Levels after bootstrap: " << (multDepth - ctReLU4->GetLevel()) << std::endl;
    } else {
        std::cout << "  Skipping bootstrap (sufficient levels available)" << std::endl;
    }

    auto ptDense3Diags = MakeCKKSPackedPlaintextVectors(cc, dense3Diagonals);
    auto dense3BiasVec = PrepareBiasVector(trainedWeights.fc3_bias, dense3Output);
    auto ptDense3Bias = cc->MakeCKKSPackedPlaintext(dense3BiasVec);

    // Layer 11: Dense3
    std::cout << "\n[Layer 11] Dense3 (84 -> 10)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU4, cc->EvalRotate(ctReLU4, -dense3Cols));
    auto ctOutput = EvalMultMatVecDiag(ctReLU4, ptDense3Diags, 2, dense3Rotations);

    // Add bias
    ctOutput = cc->EvalAdd(ctOutput, ptDense3Bias);

    double dense3Time = TOC(t);
    std::cout << "  Time: " << dense3Time << " ms" << std::endl;
    std::cout << "  Level: " << ctOutput->GetLevel() << std::endl;

    // VALIDATION: Dense3 (Final Output)
    Plaintext ptDense3Result;
    cc->Decrypt(keys.secretKey, ctOutput, &ptDense3Result);
    ptDense3Result->SetLength(dense3Output);
    std::vector<double> encDense3 = ptDense3Result->GetRealPackedValue();
    CompareVectors(clearDense3, encDense3, "Dense3 (Final)", 1e-1);

    ptDense3Diags.clear();
    ptDense3Diags.shrink_to_fit();

    double totalInferenceTime = conv1Time + relu1Time + pool1Time + bootstrap1Time + conv2Time + relu2Time +
                                pool2Time + bootstrap2Time + dense1Time + relu3Time + bootstrap3Time +
                                dense2Time + relu4Time + bootstrap4Time + dense3Time;
    double totalBootstrapTime = bootstrap1Time + bootstrap2Time + bootstrap3Time + bootstrap4Time;
    std::cout << "\nTotal inference time: " << totalInferenceTime << " ms" << std::endl;
    if (totalBootstrapTime > 0) {
        std::cout << "  (includes " << totalBootstrapTime << " ms for bootstrapping)" << std::endl;
    } else {
        std::cout << "  (no bootstrapping needed - sufficient depth available)" << std::endl;
    }

    // ========== Decrypt and Display Results ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Decrypting results..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    TIC(t);
    Plaintext ptOutput;
    cc->Decrypt(keys.secretKey, ctOutput, &ptOutput);
    ptOutput->SetLength(dense3Output);
    std::vector<double> outputVector = ptOutput->GetRealPackedValue();
    std::cout << "Decryption time: " << TOC(t) << " ms" << std::endl;

    std::cout << "\nOutput logits (10 classes):" << std::endl;
    for (uint32_t i = 0; i < dense3Output; i++) {
        std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(6)
                  << outputVector[i] << std::endl;
    }

    // Find predicted class
    uint32_t predictedClass = 0;
    double maxLogit = outputVector[0];
    for (uint32_t i = 1; i < dense3Output; i++) {
        if (outputVector[i] > maxLogit) {
            maxLogit = outputVector[i];
            predictedClass = i;
        }
    }

    std::cout << "\nPredicted class: " << predictedClass << std::endl;
    std::cout << "Confidence: " << maxLogit << std::endl;

    // ========== Performance Summary ==========
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Performance Summary (LeNet-5)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::left << std::setw(30) << "Layer" << std::setw(15) << "Time (ms)" << "Level" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Conv1 (28x28x1->24x24x6)" << std::setw(15) << conv1Time << ctConv1->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU1" << std::setw(15) << relu1Time << ctReLU1->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "AvgPool1 (24x24x6->12x12x6)" << std::setw(15) << pool1Time << (bootstrap1Time > 0 ? "N/A (bootstrapped)" : std::to_string(ctPool1->GetLevel())) << std::endl;
    if (bootstrap1Time > 0) {
        std::cout << std::left << std::setw(30) << "  + Bootstrap 1" << std::setw(15) << bootstrap1Time << ctPool1->GetLevel() << std::endl;
    }
    std::cout << std::left << std::setw(30) << "Conv2 (12x12x6->8x8x16)" << std::setw(15) << conv2Time << ctConv2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU2" << std::setw(15) << relu2Time << ctReLU2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "AvgPool2 (8x8x16->4x4x16)" << std::setw(15) << pool2Time << (bootstrap2Time > 0 ? "N/A (bootstrapped)" : std::to_string(ctPool2->GetLevel())) << std::endl;
    if (bootstrap2Time > 0) {
        std::cout << std::left << std::setw(30) << "  + Bootstrap 2" << std::setw(15) << bootstrap2Time << ctPool2->GetLevel() << std::endl;
    }
    std::cout << std::left << std::setw(30) << "Dense1 (256->120)" << std::setw(15) << dense1Time << ctDense1->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU3" << std::setw(15) << relu3Time << (bootstrap3Time > 0 ? "N/A (bootstrapped)" : std::to_string(ctReLU3->GetLevel())) << std::endl;
    if (bootstrap3Time > 0) {
        std::cout << std::left << std::setw(30) << "  + Bootstrap 3" << std::setw(15) << bootstrap3Time << ctReLU3->GetLevel() << std::endl;
    }
    std::cout << std::left << std::setw(30) << "Dense2 (120->84)" << std::setw(15) << dense2Time << ctDense2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU4" << std::setw(15) << relu4Time << (bootstrap4Time > 0 ? "N/A (bootstrapped)" : std::to_string(ctReLU4->GetLevel())) << std::endl;
    if (bootstrap4Time > 0) {
        std::cout << std::left << std::setw(30) << "  + Bootstrap 4" << std::setw(15) << bootstrap4Time << ctReLU4->GetLevel() << std::endl;
    }
    std::cout << std::left << std::setw(30) << "Dense3 (84->10)" << std::setw(15) << dense3Time << ctOutput->GetLevel() << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Total Inference" << std::setw(15) << totalInferenceTime << std::endl;
    if (totalBootstrapTime > 0) {
        std::cout << std::left << std::setw(30) << "  (Bootstrapping only)" << std::setw(15) << totalBootstrapTime << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\n✓ MNIST LeNet-5 Inference Complete (Scheme Switching)!" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        int sampleIndex = 8;  // Default to sample 8 (label 5)
        ActivationType activationType = ActivationType::SCHEME_SWITCH;  // Default

        // Parse command line arguments
        if (argc > 1) {
            sampleIndex = std::atoi(argv[1]);
            if (sampleIndex < 0 || sampleIndex > 9999) {
                std::cerr << "Error: Sample index must be between 0 and 9999" << std::endl;
                std::cerr << "Usage: " << argv[0] << " [sample_index] [activation_type]" << std::endl;
                std::cerr << "  activation_type: scheme (default), cheby, square" << std::endl;
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

        MNISTLeNet5Inference(sampleIndex, activationType);
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
