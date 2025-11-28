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
 * @brief Print min/max bounds of decrypted vector
 */
void PrintBounds(const std::vector<double>& vec, const std::string& name) {
    double minVal = *std::min_element(vec.begin(), vec.end());
    double maxVal = *std::max_element(vec.begin(), vec.end());
    std::cout << "  " << name << " bounds: [" << std::fixed << std::setprecision(6)
              << minVal << ", " << maxVal << "]" << std::endl;
}


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

    // ReLU(x) = -(x < 0) + 1
    // Step 1: Compute comparison result (x < 0)
    auto ctComparison = cc->EvalCompareSchemeSwitching(ct, ctZero, NextPow2(numSlots), totalSlots, 0, scaleSign);

    // Step 2: Multiply input by comparison result to get ReLU
    // The comparison returns 1 if x > 0, 0 otherwise
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

// ========== CLEARTEXT VALIDATION FUNCTIONS (END) ==========

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

// ========== CLEARTEXT VALIDATION FUNCTIONS (END) ==========

void MNISTLeNet5Inference(int sampleIndex = 8, ActivationType activationType = ActivationType::SCHEME_SWITCH, uint32_t ChebyDegree = 119, uint32_t ChebyMultDepth = 8, bool useOptimized = false, bool enableValidation = true) {
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

    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;
    uint32_t ringDim = 65536;
    std::vector<uint32_t> levelBudget = {3, 3};
    std::vector<uint32_t> bsgsDim = {0, 0};
    SecurityLevel sl = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE = 25;
    uint32_t slots = 8192;
    uint32_t batchSize = slots;

    // Bootstrapping parameters
    uint32_t levelsAvailableAfterBootstrap = ChebyMultDepth+1;
    uint32_t approxBootstrapDepth = FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    uint32_t multDepth = 1;
    bool enableBootstrapping = false;  // Track if bootstrapping is needed

    if (activationType == ActivationType::CHEBYSHEV) {
        // conv + avg + act + conv + avg + act + fc + act + fc + act + fc
        uint32_t option1 = 1 + 1 + ChebyMultDepth + 1 + 1 + ChebyMultDepth + 1 + ChebyMultDepth + 1 + ChebyMultDepth + 1;
        uint32_t option2 = std::max({1U,1U,ChebyMultDepth,1U,1U,ChebyMultDepth,1U,ChebyMultDepth,1U,ChebyMultDepth,1U}) + approxBootstrapDepth + 1;
        // Prefer option1 unless option2 saves 4+ layers (threshold of 3)
        enableBootstrapping = (option2 + 3 < option1);
        multDepth = enableBootstrapping ? option2 : option1;
    } else if (activationType == ActivationType::SQUARE) {
        // conv + avg + act + conv + avg + act + fc + act + fc + act + fc
        uint32_t option1 = 1 + 1 + 2 + 1 + 1 + 2 + 1 + 2 + 1 + 2 + 1;
        uint32_t option2 = std::max({1U,1U,2U,1U,1U,2U,1U,2U,1U,2U,1U}) + approxBootstrapDepth + 1;
        // Prefer option1 unless option2 saves 4+ layers (threshold of 3)
        enableBootstrapping = (option2 + 3 < option1);
        multDepth = enableBootstrapping ? option2 : option1;
    } else if (activationType == ActivationType::SCHEME_SWITCH) {
        // conv + avg + act + conv + avg + act + fc + act + fc + act + fc
        uint32_t option1 = 1 + 1 + 13 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1;
        uint32_t option2 = std::max({1U,1U,13U,1U,1U,1U,1U,1U,1U,1U,1U}) + approxBootstrapDepth + 1;
        // Prefer option1 unless option2 saves 4+ layers (threshold of 3)
        enableBootstrapping = (option2 + 3 < option1);
        multDepth = enableBootstrapping ? option2 : option1;
    }

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetRingDim(ringDim);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetScalingTechnique(scTech);
    parameters.SetSecurityLevel(sl);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecretKeyDist(secretKeyDist);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    if (enableBootstrapping) cc->Enable(FHE);  // Enable bootstrapping only if needed

    // Only enable scheme switching if needed
    if (activationType == ActivationType::SCHEME_SWITCH) {
        cc->Enable(SCHEMESWITCH);
    }

    std::cout << "CKKS scheme using ring dimension " << cc->GetRingDimension() << std::endl;
    std::cout << "Number of slots: " << slots << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl;
    std::cout << "Activation function: " << activationName << std::endl;

    std::cout << "Ring dimension: " << cc->GetRingDimension() << std::endl;
    std::cout << "Number of moduli: " << cc->GetCryptoParameters()->GetElementParams()->GetParams().size() << std::endl;

    // Calculate actual logQ
    auto moduli = cc->GetCryptoParameters()->GetElementParams()->GetParams();
    uint32_t firstLogQ = moduli[0]->GetModulus().GetMSB();
    std::cout << "Q[i]: " << firstLogQ;
    uint32_t actualLogQ = firstLogQ;
    for (size_t i = 1; i < moduli.size(); i++) {
        uint32_t bits = moduli[i]->GetModulus().GetMSB();
        actualLogQ += bits;
        std::cout << "|" << bits;
    }
    std::cout << std::endl << "Total logQ: " << actualLogQ << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl;

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

    // Setup scheme switching only if needed
    double scaleSignFHEW = 1.0;
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

        std::cout << "FHEW scheme is using lattice parameter " << ccLWE->GetParams()->GetLWEParams()->Getn();
        std::cout << ", logQ " << logQ_ccLWE;
        std::cout << ", modulus q " << ccLWE->GetParams()->GetLWEParams()->Getq() << std::endl << std::endl;
        std::cout << ", and precision " << ccLWE->GetMaxPlaintextSpace().ConvertToInt();  // Small precision << std::endl << std::endl;
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

    // Multiplexing parameters (optimized vs unoptimized)
    // Rule: input_gap = last layer's output_gap, output_gap = input_gap * stride (if optimized)
    uint32_t conv1_input_gap = 1;    // Initial input, no multiplexing
    uint32_t conv1_output_gap = useOptimized ? (conv1_input_gap * 1) : 1;  // stride=1

    uint32_t pool1_input_gap = conv1_output_gap;
    uint32_t pool1_output_gap = useOptimized ? (pool1_input_gap * 2) : 1;  // stride=2

    uint32_t conv2_input_gap = pool1_output_gap;
    uint32_t conv2_output_gap = useOptimized ? (conv2_input_gap * 1) : 1;  // stride=1

    uint32_t pool2_input_gap = conv2_output_gap;
    uint32_t pool2_output_gap = useOptimized ? (pool2_input_gap * 2) : 1;  // stride=2

    std::cout << "\nRunning in " << (useOptimized ? "optimized" : "unoptimized") << " mode" << std::endl;
    std::cout << "Multiplexing gaps:" << std::endl;
    std::cout << "  Conv1: input_gap=" << conv1_input_gap << ", output_gap=" << conv1_output_gap << std::endl;
    std::cout << "  Pool1: input_gap=" << pool1_input_gap << ", output_gap=" << pool1_output_gap << std::endl;
    std::cout << "  Conv2: input_gap=" << conv2_input_gap << ", output_gap=" << conv2_output_gap << std::endl;
    std::cout << "  Pool2: input_gap=" << pool2_input_gap << ", output_gap=" << pool2_output_gap << std::endl;

    // ========== Build Toeplitz matrices and pack into diagonals ==========
    std::cout << "\nPreparing network weights..." << std::endl;
    TIC(t);

    // Conv1 Toeplitz
    auto toeplitzConv1 = ConstructConv2DToeplitz(conv1Kernel, 28, 28, 1, 0, 1, conv1_input_gap, conv1_output_gap);
    std::vector<std::vector<double>> conv1Diagonals = PackMatDiagWise(toeplitzConv1, batchSize);
    std::size_t conv1Cols = conv1Diagonals.size();
    std::vector<bool> conv1NonZeros(conv1Cols);
    std::vector<int32_t> conv1Rotations = getOptimalRots(conv1Diagonals, &conv1NonZeros, useOptimized);
    std::cout << "  Conv1 Toeplitz: " << conv1Cols << " rows, "
              << conv1Rotations.size() << " non-zero diagonals" << std::endl;

    // AvgPool1 Toeplitz
    auto toeplitzPool1 = ConstructConv2DToeplitz(avgpool1Kernel, 24, 24, 2, 0, 1, pool1_input_gap, pool1_output_gap);
    std::vector<std::vector<double>> pool1Diagonals = PackMatDiagWise(toeplitzPool1, batchSize);
    std::size_t pool1Cols = pool1Diagonals.size();
    std::vector<bool> pool1NonZeros(pool1Cols);
    std::vector<int32_t> pool1Rotations = getOptimalRots(pool1Diagonals, &pool1NonZeros, useOptimized);
    std::cout << "  AvgPool1 Toeplitz: " << pool1Cols << " rows, "
              << pool1Rotations.size() << " rotation keys needed" << std::endl;

    // Conv2 Toeplitz
    auto toeplitzConv2 = ConstructConv2DToeplitz(conv2Kernel, 12, 12, 1, 0, 1, conv2_input_gap, conv2_output_gap);
    std::vector<std::vector<double>> conv2Diagonals = PackMatDiagWise(toeplitzConv2, batchSize);
    std::size_t conv2Cols = conv2Diagonals.size();
    std::vector<bool> conv2NonZeros(conv2Cols);
    std::vector<int32_t> conv2Rotations = getOptimalRots(conv2Diagonals, &conv2NonZeros, useOptimized);
    std::cout << "  Conv2 Toeplitz: " << conv2Cols << " rows, "
              << conv2Rotations.size() << " rotation keys needed" << std::endl;

    // AvgPool2 Toeplitz
    auto toeplitzPool2 = ConstructConv2DToeplitz(avgpool2Kernel, 8, 8, 2, 0, 1, pool2_input_gap, pool2_output_gap);
    std::vector<std::vector<double>> pool2Diagonals = PackMatDiagWise(toeplitzPool2, batchSize);
    std::size_t pool2Cols = pool2Diagonals.size();
    std::vector<bool> pool2NonZeros(pool2Cols);
    std::vector<int32_t> pool2Rotations = getOptimalRots(pool2Diagonals, &pool2NonZeros, useOptimized);
    std::cout << "  AvgPool2 Toeplitz: " << pool2Cols << " rows, "
              << pool2Rotations.size() << " rotation keys needed" << std::endl;

    // Dense layer 1 - unmultiplex the pool2 output
    auto dense1 = MultiplexDenseMatrix(dense1Weights, pool2OutputHeight, pool2OutputWidth, pool2_output_gap);
    std::vector<std::vector<double>> dense1Diagonals = PackMatDiagWise(dense1, batchSize);
    std::size_t dense1Cols = dense1Diagonals.size();
    std::vector<bool> dense1NonZeros(dense1Cols);
    std::vector<int32_t> dense1Rotations = getOptimalRots(dense1Diagonals, &dense1NonZeros, useOptimized);
    std::cout << "  Dense1: " << dense1Cols << " rows, "
              << dense1Rotations.size() << " rotation keys needed" << std::endl;

    std::vector<std::vector<double>> dense2Diagonals = PackMatDiagWise(dense2Weights, batchSize);
    std::size_t dense2Cols = dense2Diagonals.size();
    std::vector<bool> dense2NonZeros(dense2Cols);
    std::vector<int32_t> dense2Rotations = getOptimalRots(dense2Diagonals, &dense2NonZeros, useOptimized);
    std::cout << "  Dense2: " << dense2Cols << " rows, "
              << dense2Rotations.size() << " rotation keys needed" << std::endl;

    std::vector<std::vector<double>> dense3Diagonals = PackMatDiagWise(dense3Weights, batchSize);
    std::size_t dense3Cols = dense3Diagonals.size();
    std::vector<bool> dense3NonZeros(dense3Cols);
    std::vector<int32_t> dense3Rotations = getOptimalRots(dense3Diagonals, &dense3NonZeros, useOptimized);
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

    // Calculate multiplexed sizes for layers with output_gap > 1
    // AvgPool1: output_gap = 2
    uint32_t pool1_output_gap_squared = pool1_output_gap * pool1_output_gap;
    uint32_t pool1_super_channels = (pool1OutputChannels + pool1_output_gap_squared - 1) / pool1_output_gap_squared;
    uint32_t pool1_multiplexed_height = pool1OutputHeight * pool1_output_gap;
    uint32_t pool1_multiplexed_width = pool1OutputWidth * pool1_output_gap;
    uint32_t pool1_multiplexed_size = pool1_super_channels * pool1_multiplexed_height * pool1_multiplexed_width;

    // Conv2: output_gap = 2
    uint32_t conv2_output_gap_squared = conv2_output_gap * conv2_output_gap;
    uint32_t conv2_super_channels = (conv2OutputChannels + conv2_output_gap_squared - 1) / conv2_output_gap_squared;
    uint32_t conv2_multiplexed_height = conv2OutputHeight * conv2_output_gap;
    uint32_t conv2_multiplexed_width = conv2OutputWidth * conv2_output_gap;
    uint32_t conv2_multiplexed_size = conv2_super_channels * conv2_multiplexed_height * conv2_multiplexed_width;

    // AvgPool2: output_gap = 4
    uint32_t pool2_output_gap_squared = pool2_output_gap * pool2_output_gap;
    uint32_t pool2_super_channels = (pool2OutputChannels + pool2_output_gap_squared - 1) / pool2_output_gap_squared;
    uint32_t pool2_multiplexed_height = pool2OutputHeight * pool2_output_gap;
    uint32_t pool2_multiplexed_width = pool2OutputWidth * pool2_output_gap;
    uint32_t pool2_multiplexed_size = pool2_super_channels * pool2_multiplexed_height * pool2_multiplexed_width;

    std::cout << "\nMultiplexed output sizes:" << std::endl;
    std::cout << "  AvgPool1: " << pool1_super_channels << "x" << pool1_multiplexed_height << "x" << pool1_multiplexed_width
              << " = " << pool1_multiplexed_size << " (logical: 12x12x6 = " << pool1FlatSize << ")" << std::endl;
    std::cout << "  Conv2: " << conv2_super_channels << "x" << conv2_multiplexed_height << "x" << conv2_multiplexed_width
              << " = " << conv2_multiplexed_size << " (logical: 8x8x16 = " << conv2FlatSize << ")" << std::endl;
    std::cout << "  AvgPool2: " << pool2_super_channels << "x" << pool2_multiplexed_height << "x" << pool2_multiplexed_width
              << " = " << pool2_multiplexed_size << " (logical: 4x4x16 = " << pool2FlatSize << ")" << std::endl;

    // ========== CLEARTEXT FORWARD PASS FOR VALIDATION ==========
    std::vector<double> clearConv1, clearReLU1, clearPool1, clearConv2, clearReLU2, clearPool2;
    std::vector<double> clearDense1, clearReLU3, clearDense2, clearReLU4, clearDense3;

    if (enableValidation) {
        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "Computing cleartext reference values..." << std::endl;
        std::cout << std::string(80, '-') << std::endl;

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
        clearConv1 = CleartextFlatten(clearConv1_3D);
        std::cout << "  Cleartext Conv1 output size: " << clearConv1.size() << std::endl;

        // Cleartext Activation1
        clearReLU1 = CleartextActivation(clearConv1, activationType, trainedWeights.scale1);
        std::cout << "  Cleartext Activation1 output size: " << clearReLU1.size() << std::endl;

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
        clearPool1 = CleartextFlatten(clearPool1_3D);
        std::cout << "  Cleartext Pool1 output size: " << clearPool1.size() << std::endl;

        // Cleartext Conv2
        auto clearConv2_3D = CleartextConv2D(clearPool1_3D, conv2Kernel, 1, 0, &trainedWeights.conv2_bias);
        clearConv2 = CleartextFlatten(clearConv2_3D);
        std::cout << "  Cleartext Conv2 output size: " << clearConv2.size() << std::endl;

        // Cleartext Activation2
        clearReLU2 = CleartextActivation(clearConv2, activationType, trainedWeights.scale2);
        std::cout << "  Cleartext Activation2 output size: " << clearReLU2.size() << std::endl;

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
        clearPool2 = CleartextFlatten(clearPool2_3D);
        std::cout << "  Cleartext Pool2 output size: " << clearPool2.size() << std::endl;

        // Cleartext Dense1
        clearDense1 = CleartextDense(clearPool2, dense1Weights, &trainedWeights.fc1_bias);
        std::cout << "  Cleartext Dense1 output size: " << clearDense1.size() << std::endl;

        // Cleartext Activation3
        clearReLU3 = CleartextActivation(clearDense1, activationType, trainedWeights.scale3);
        std::cout << "  Cleartext Activation3 output size: " << clearReLU3.size() << std::endl;

        // Cleartext Dense2
        clearDense2 = CleartextDense(clearReLU3, dense2Weights, &trainedWeights.fc2_bias);
        std::cout << "  Cleartext Dense2 output size: " << clearDense2.size() << std::endl;

        // Cleartext Activation4
        clearReLU4 = CleartextActivation(clearDense2, activationType, trainedWeights.scale4);
        std::cout << "  Cleartext Activation4 output size: " << clearReLU4.size() << std::endl;

        // Cleartext Dense3
        clearDense3 = CleartextDense(clearReLU4, dense3Weights, &trainedWeights.fc3_bias);
        std::cout << "  Cleartext Dense3 (final) output size: " << clearDense3.size() << std::endl;

        std::cout << "Cleartext reference computation complete!" << std::endl;
    }

    // ========== Forward Pass ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Starting LeNet-5 encrypted inference..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Layer 1: Conv1
    std::cout << "\n[Layer 1] Conv1 (28x28x1 -> 24x24x6)..." << std::endl;
    auto conv1BiasVec = PrepareBiasVector(trainedWeights.conv1_bias, conv1FlatSize, conv1OutputChannels, conv1OutputHeight * conv1OutputWidth);
    auto ptConv1Bias = cc->MakeCKKSPackedPlaintext(conv1BiasVec);

    TIC(t);
    // ctInput = cc->EvalRotate(ctInput, -conv1Cols);
    // TESTING: Use raw diagonals directly instead of encoded plaintexts
    uint32_t hoistingMode = useOptimized ? 2 : 1;
    auto ctConv1 = EvalMultMatVecDiag(ctInput, conv1Diagonals, hoistingMode, conv1Rotations, 0, &conv1NonZeros);

    // Add bias
    ctConv1 = cc->EvalAdd(ctConv1, ptConv1Bias);

    double conv1Time = TOC(t);
    std::cout << "  Time: " << conv1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctConv1->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Conv1 output
    if (enableValidation) {
        Plaintext ptConv1Result;
        cc->Decrypt(keys.secretKey, ctConv1, &ptConv1Result);
        ptConv1Result->SetLength(conv1FlatSize);
        std::vector<double> encConv1 = ptConv1Result->GetRealPackedValue();
        CompareVectors(clearConv1, encConv1, "Conv1", 1e-1);
    }

    // Layer 2: Activation1
    std::cout << "\n[Layer 2] Activation1 (" << activationName << ")..." << std::endl;
    TIC(t);
    auto ctReLU1 = EvalActivation(cc, ctConv1, activationType, keys.publicKey, conv1FlatSize, slots, scaleSignFHEW, ChebyDegree, std::floor(*std::min_element(clearConv1.begin(), clearConv1.end())), std::ceil(*std::max_element(clearConv1.begin(), clearConv1.end())), trainedWeights.scale1);
    double relu1Time = TOC(t);
    std::cout << "  Time: " << relu1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU1->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Activation1 output
    if (enableValidation) {
        Plaintext ptReLU1Result;
        cc->Decrypt(keys.secretKey, ctReLU1, &ptReLU1Result);
        ptReLU1Result->SetLength(conv1FlatSize);
        std::vector<double> encReLU1 = ptReLU1Result->GetRealPackedValue();
        CompareVectors(clearReLU1, encReLU1, "Activation1", 1e-1);
    }

    // Layer 3: AvgPool1
    std::cout << "\n[Layer 3] AvgPool1 (24x24x6 -> 12x12x6";
    if (pool1_output_gap > 1) {
        std::cout << " multiplexed to " << pool1_super_channels << "x" << pool1_multiplexed_height << "x" << pool1_multiplexed_width;
    }
    std::cout << ")..." << std::endl;
    TIC(t);
    // cc->EvalAddInPlace(ctReLU1, cc->EvalRotate(ctReLU1, -pool1Cols));
    // TESTING: Use raw diagonals directly instead of encoded plaintexts
    auto ctPool1 = EvalMultMatVecDiag(ctReLU1, pool1Diagonals, hoistingMode, pool1Rotations, 0, &pool1NonZeros);
    double pool1Time = TOC(t);
    std::cout << "  Time: " << pool1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctPool1->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Pool1 output
    if (enableValidation) {
        Plaintext ptPool1Result;
        cc->Decrypt(keys.secretKey, ctPool1, &ptPool1Result);
        ptPool1Result->SetLength(pool1_multiplexed_size);
        std::vector<double> encPool1 = ptPool1Result->GetRealPackedValue();
        // Only compare the first pool1FlatSize elements (logical output)
        encPool1.resize(pool1FlatSize);
        CompareVectors(clearPool1, encPool1, "AvgPool1", 1e-1);
    }

    auto conv2BiasVec = PrepareBiasVector(trainedWeights.conv2_bias, conv2_multiplexed_size, conv2OutputChannels,
                                          conv2OutputHeight * conv2OutputWidth, conv2_output_gap,
                                          conv2OutputHeight, conv2OutputWidth);
    auto ptConv2Bias = cc->MakeCKKSPackedPlaintext(conv2BiasVec);

    // Layer 4: Conv2
    std::cout << "\n[Layer 4] Conv2 (12x12x6 -> 8x8x16";
    if (conv2_output_gap > 1) {
        std::cout << " multiplexed to " << conv2_super_channels << "x" << conv2_multiplexed_height << "x" << conv2_multiplexed_width;
    }
    std::cout << ")..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctPool1, cc->EvalRotate(ctPool1, -conv2Cols));
    cc->EvalAddInPlace(ctPool1, cc->EvalRotate(cc->EvalRotate(ctPool1, -conv2Cols), -conv2Cols));
    // TESTING: Use raw diagonals directly instead of encoded plaintexts
    auto ctConv2 = EvalMultMatVecDiag(ctPool1, conv2Diagonals, hoistingMode, conv2Rotations, 0, &conv2NonZeros);

    // Add bias
    ctConv2 = cc->EvalAdd(ctConv2, ptConv2Bias);

    double conv2Time = TOC(t);
    std::cout << "  Time: " << conv2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctConv2->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Conv2 output
    if (enableValidation) {
        Plaintext ptConv2Result;
        cc->Decrypt(keys.secretKey, ctConv2, &ptConv2Result);
        ptConv2Result->SetLength(conv2_multiplexed_size);
        std::vector<double> encConv2 = ptConv2Result->GetRealPackedValue();
        // Only compare the first conv2FlatSize elements (logical output)
        encConv2.resize(conv2FlatSize);
        CompareVectors(clearConv2, encConv2, "Conv2", 1e-1);
    }

    // Layer 5: Activation2
    std::cout << "\n[Layer 5] Activation2 (" << activationName << ")..." << std::endl;
    TIC(t);
    auto ctReLU2 = EvalActivation(cc, ctConv2, activationType, keys.publicKey, conv2_multiplexed_size, slots, scaleSignFHEW, ChebyDegree, std::floor(*std::min_element(clearConv2.begin(), clearConv2.end())), std::ceil(*std::max_element(clearConv2.begin(), clearConv2.end())), trainedWeights.scale2);
    double relu2Time = TOC(t);
    std::cout << "  Time: " << relu2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU2->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Activation2 output
    if (enableValidation) {
        Plaintext ptReLU2Result;
        cc->Decrypt(keys.secretKey, ctReLU2, &ptReLU2Result);
        ptReLU2Result->SetLength(conv2_multiplexed_size);
        std::vector<double> encReLU2 = ptReLU2Result->GetRealPackedValue();
        // Only compare the first conv2FlatSize elements (logical output)
        encReLU2.resize(conv2FlatSize);
        CompareVectors(clearReLU2, encReLU2, "Activation2", 1e-1);
    }

    // Layer 6: AvgPool2
    std::cout << "\n[Layer 6] AvgPool2 (8x8x16 -> 4x4x16";
    if (pool2_output_gap > 1) {
        std::cout << " multiplexed to " << pool2_super_channels << "x" << pool2_multiplexed_height << "x" << pool2_multiplexed_width;
    }
    std::cout << ")..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU2, cc->EvalRotate(ctReLU2, -pool2Cols));
    // TESTING: Use raw diagonals directly instead of encoded plaintexts
    auto ctPool2 = EvalMultMatVecDiag(ctReLU2, pool2Diagonals, hoistingMode, pool2Rotations, 0, &pool2NonZeros);
    double pool2Time = TOC(t);
    std::cout << "  Time: " << pool2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctPool2->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Pool2 output
    if (enableValidation) {
        Plaintext ptPool2Result;
        cc->Decrypt(keys.secretKey, ctPool2, &ptPool2Result);
        ptPool2Result->SetLength(pool2_multiplexed_size);
        std::vector<double> encPool2 = ptPool2Result->GetRealPackedValue();
        // Only compare the first pool2FlatSize elements (logical output)
        encPool2.resize(pool2FlatSize);
        CompareVectors(clearPool2, encPool2, "AvgPool2", 1e-1);
    }

    auto dense1BiasVec = PrepareBiasVector(trainedWeights.fc1_bias, dense1Output);
    auto ptDense1Bias = cc->MakeCKKSPackedPlaintext(dense1BiasVec);

    // Layer 7: Dense1
    std::cout << "\n[Layer 7] Dense1 (unmultiplex " << pool2_multiplexed_size << " -> " << pool2FlatSize << " -> " << dense1Output << ")..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctPool2, cc->EvalRotate(ctPool2, -dense1Cols));
    // TESTING: Use raw diagonals directly instead of encoded plaintexts
    auto ctDense1 = EvalMultMatVecDiag(ctPool2, dense1Diagonals, hoistingMode, dense1Rotations, 0, &dense1NonZeros);

    // Add bias
    ctDense1 = cc->EvalAdd(ctDense1, ptDense1Bias);

    double dense1Time = TOC(t);
    std::cout << "  Time: " << dense1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctDense1->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Dense1 output
    if (enableValidation) {
        Plaintext ptDense1Result;
        cc->Decrypt(keys.secretKey, ctDense1, &ptDense1Result);
        ptDense1Result->SetLength(dense1Output);
        std::vector<double> encDense1 = ptDense1Result->GetRealPackedValue();
        CompareVectors(clearDense1, encDense1, "Dense1", 1e-1);
    }

    // Bootstrap if needed (when levels are low)
    double bootstrap1Time = 0.0;
    uint32_t levelsRemaining1 = multDepth - ctDense1->GetLevel();
    std::cout << "\n[Bootstrap Check] " << levelsRemaining1 << " levels remaining after Dense1" << std::endl;
    if (enableBootstrapping && activationType == ActivationType::CHEBYSHEV && levelsRemaining1 <= levelsAvailableAfterBootstrap+1) {
        TIC(t);
        ctDense1 = cc->EvalBootstrap(ctDense1);
        bootstrap1Time = TOC(t);
        std::cout << "  Time: " << bootstrap1Time << " ms" << std::endl;
        std::cout << "  Levels after bootstrap: " << (multDepth - ctDense1->GetLevel()) << std::endl;
    } else {
        std::cout << "  Skipping bootstrap (sufficient levels or disabled)" << std::endl;
    }

    // Layer 8: Activation3
    std::cout << "\n[Layer 8] Activation3 (" << activationName << ")..." << std::endl;
    TIC(t);
    auto ctReLU3 = EvalActivation(cc, ctDense1, activationType, keys.publicKey, dense1Output, slots, scaleSignFHEW, ChebyDegree, std::floor(*std::min_element(clearDense1.begin(), clearDense1.end())), std::ceil(*std::max_element(clearDense1.begin(), clearDense1.end())), trainedWeights.scale3);
    double relu3Time = TOC(t);
    std::cout << "  Time: " << relu3Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU3->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Activation3 output
    if (enableValidation) {
        Plaintext ptReLU3Result;
        cc->Decrypt(keys.secretKey, ctReLU3, &ptReLU3Result);
        ptReLU3Result->SetLength(dense1Output);
        std::vector<double> encReLU3 = ptReLU3Result->GetRealPackedValue();
        CompareVectors(clearReLU3, encReLU3, "Activation3", 1e-1);
    }

    // Bootstrap if needed (when levels are low)
    double bootstrap2Time = 0.0;
    uint32_t levelsRemaining2 = multDepth - ctReLU3->GetLevel();
    std::cout << "\n[Bootstrap Check] " << levelsRemaining2 << " levels remaining after Activation3" << std::endl;
    if (enableBootstrapping && activationType == ActivationType::CHEBYSHEV && levelsRemaining2 <= levelsAvailableAfterBootstrap+1) {
        TIC(t);
        ctReLU3 = cc->EvalBootstrap(ctReLU3);
        bootstrap2Time = TOC(t);
        std::cout << "  Time: " << bootstrap2Time << " ms" << std::endl;
        std::cout << "  Levels after bootstrap: " << (multDepth - ctReLU3->GetLevel()) << std::endl;
    } else {
        std::cout << "  Skipping bootstrap (sufficient levels or disabled)" << std::endl;
    }

    auto dense2BiasVec = PrepareBiasVector(trainedWeights.fc2_bias, dense2Output);
    auto ptDense2Bias = cc->MakeCKKSPackedPlaintext(dense2BiasVec);

    // Layer 9: Dense2
    std::cout << "\n[Layer 9] Dense2 (120 -> 84)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU3, cc->EvalRotate(ctReLU3, -dense2Cols));
    // TESTING: Use raw diagonals directly instead of encoded plaintexts
    auto ctDense2 = EvalMultMatVecDiag(ctReLU3, dense2Diagonals, hoistingMode, dense2Rotations, 0, &dense2NonZeros);

    // Add bias
    ctDense2 = cc->EvalAdd(ctDense2, ptDense2Bias);

    double dense2Time = TOC(t);
    std::cout << "  Time: " << dense2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctDense2->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Dense2 output
    if (enableValidation) {
        Plaintext ptDense2Result;
        cc->Decrypt(keys.secretKey, ctDense2, &ptDense2Result);
        ptDense2Result->SetLength(dense2Output);
        std::vector<double> encDense2 = ptDense2Result->GetRealPackedValue();
        CompareVectors(clearDense2, encDense2, "Dense2", 1e-1);
    }

    // Bootstrap if needed (when levels are low)
    double bootstrap3Time = 0.0;
    uint32_t levelsRemaining4 = multDepth - ctDense2->GetLevel();
    std::cout << "\n[Bootstrap Check] " << levelsRemaining4 << " levels remaining after Dense2" << std::endl;
    if (enableBootstrapping && activationType == ActivationType::CHEBYSHEV && levelsRemaining4 <= levelsAvailableAfterBootstrap+1) {
        TIC(t);
        ctDense2 = cc->EvalBootstrap(ctDense2);
        bootstrap3Time = TOC(t);
        std::cout << "  Time: " << bootstrap3Time << " ms" << std::endl;
        std::cout << "  Levels after bootstrap: " << (multDepth - ctDense2->GetLevel()) << std::endl;
    } else {
        std::cout << "  Skipping bootstrap (sufficient levels or disabled)" << std::endl;
    }

    // Layer 10: Activation4
    std::cout << "\n[Layer 10] Activation4 (" << activationName << ")..." << std::endl;
    TIC(t);
    auto ctReLU4 = EvalActivation(cc, ctDense2, activationType, keys.publicKey, dense2Output, slots, scaleSignFHEW, ChebyDegree, std::floor(*std::min_element(clearDense2.begin(), clearDense2.end())), std::ceil(*std::max_element(clearDense2.begin(), clearDense2.end())), trainedWeights.scale4);
    double relu4Time = TOC(t);
    std::cout << "  Time: " << relu4Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU4->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Activation4 output
    if (enableValidation) {
        Plaintext ptReLU4Result;
        cc->Decrypt(keys.secretKey, ctReLU4, &ptReLU4Result);
        ptReLU4Result->SetLength(dense2Output);
        std::vector<double> encReLU4 = ptReLU4Result->GetRealPackedValue();
        CompareVectors(clearReLU4, encReLU4, "Activation4", 1e-1);
    }

    auto dense3BiasVec = PrepareBiasVector(trainedWeights.fc3_bias, dense3Output);
    auto ptDense3Bias = cc->MakeCKKSPackedPlaintext(dense3BiasVec);

    // Layer 11: Dense3
    std::cout << "\n[Layer 11] Dense3 (84 -> 10)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU4, cc->EvalRotate(ctReLU4, -dense3Cols));
    // TESTING: Use raw diagonals directly instead of encoded plaintexts
    auto ctOutput = EvalMultMatVecDiag(ctReLU4, dense3Diagonals, hoistingMode, dense3Rotations, 0, &dense3NonZeros);

    // Add bias
    ctOutput = cc->EvalAdd(ctOutput, ptDense3Bias);

    double dense3Time = TOC(t);
    std::cout << "  Time: " << dense3Time << " ms" << std::endl;
    std::cout << "  Level: " << ctOutput->GetLevel() << std::endl;

    // Validation: Compare encrypted vs cleartext Dense3 (final) output
    if (enableValidation) {
        Plaintext ptDense3Result;
        cc->Decrypt(keys.secretKey, ctOutput, &ptDense3Result);
        ptDense3Result->SetLength(dense3Output);
        std::vector<double> encDense3 = ptDense3Result->GetRealPackedValue();
        CompareVectors(clearDense3, encDense3, "Dense3 (Final)", 1e-1);
    }

    double totalInferenceTime = conv1Time + relu1Time + pool1Time + conv2Time + relu2Time +
                                pool2Time + dense1Time + bootstrap1Time + relu3Time +
                                bootstrap2Time + dense2Time + bootstrap3Time + relu4Time + dense3Time;
    double totalBootstrapTime = bootstrap1Time + bootstrap2Time + bootstrap3Time;
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
    ptOutput->SetLength(dense3Output);
    std::vector<double> outputVector = ptOutput->GetRealPackedValue();
    std::cout << "Output decryption time: " << TOC(t) << " ms" << std::endl;

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
    std::cout << "Performance Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::left << std::setw(32) << "Layer" << std::setw(15) << "Time (ms)" << "Level" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(32) << "Conv1 (28x28x1 -> 24x24x6)" << std::setw(15) << conv1Time << std::to_string(ctConv1->GetLevel()) << std::endl;
    std::cout << std::left << std::setw(32) << "Activation1" << std::setw(15) << relu1Time << ctReLU1->GetLevel() << std::endl;
    std::cout << std::left << std::setw(32) << "AvgPool1 (24x24x6 -> 12x12x6)" << std::setw(15) << pool1Time << ctPool1->GetLevel() << std::endl;
    std::cout << std::left << std::setw(32) << "Conv2 (12x12x6 -> 8x8x16)" << std::setw(15) << conv2Time << std::to_string(ctConv2->GetLevel()) << std::endl;
    std::cout << std::left << std::setw(32) << "Activation2" << std::setw(15) << relu2Time << ctReLU2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(32) << "AvgPool2 (8x8x16 -> 4x4x16)" << std::setw(15) << pool2Time << ctPool2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(32) << "Dense1 (256 -> 120)" << std::setw(15) << dense1Time << (bootstrap1Time > 0 ? "N/A" : std::to_string(ctDense1->GetLevel())) << std::endl;
    if (bootstrap1Time > 0) {
        std::cout << std::left << std::setw(32) << "  Bootstrap3" << std::setw(15) << bootstrap1Time << ctDense1->GetLevel() << std::endl;
    }
    std::cout << std::left << std::setw(32) << "Activation3" << std::setw(15) << relu3Time << (bootstrap2Time > 0 ? "N/A" : std::to_string(ctReLU3->GetLevel())) << std::endl;
    if (bootstrap2Time > 0) {
        std::cout << std::left << std::setw(32) << "  Bootstrap4" << std::setw(15) << bootstrap2Time << ctReLU3->GetLevel() << std::endl;
    }
    std::cout << std::left << std::setw(32) << "Dense2 (120 -> 84)" << std::setw(15) << dense2Time << (bootstrap3Time > 0 ? "N/A" : std::to_string(ctDense2->GetLevel())) << std::endl;
    if (bootstrap3Time > 0) {
        std::cout << std::left << std::setw(32) << "  Bootstrap5" << std::setw(15) << bootstrap3Time << ctDense2->GetLevel() << std::endl;
    }
    std::cout << std::left << std::setw(32) << "Activation4" << std::setw(15) << relu4Time << ctReLU4->GetLevel() << std::endl;
    std::cout << std::left << std::setw(32) << "Dense3 (84 -> 10)" << std::setw(15) << dense3Time << ctOutput->GetLevel() << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(32) << "Total Inference" << std::setw(15) << totalInferenceTime << std::endl;
    if (totalBootstrapTime > 0) {
        std::cout << std::left << std::setw(32) << "  (Bootstrapping only)" << std::setw(15) << totalBootstrapTime << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\n✓ MNIST LeNet-5 Inference Complete (" << activationName << ")!" << std::endl;
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

        MNISTLeNet5Inference(sampleIndex, activationType, chebyDegree, chebyMultDepth, useOptimized, enableValidation);
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
