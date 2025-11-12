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

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace openfhe_numpy;
using namespace lbcrypto;

// ========== DEBUG MODE ==========
// Set to true to decrypt and print intermediate values after each layer
constexpr bool DEBUG_MODE = true;

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

// ========== CLEARTEXT VALIDATION FUNCTIONS (REMOVE LATER) ==========

/**
 * @brief Cleartext 2D convolution for validation
 */
std::vector<std::vector<std::vector<double>>> CleartextConv2D(
    const std::vector<std::vector<std::vector<double>>>& input,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
    uint32_t stride = 1,
    uint32_t padding = 0
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
    const std::vector<std::vector<double>>& weights
) {
    std::vector<double> output(weights.size(), 0.0);
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < input.size(); j++) {
            output[i] += weights[i][j] * input[j];
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

void MNISTLeNet5Inference() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  MNIST LeNet-5 Network Inference (Scheme Switching)" << std::endl;
    std::cout << "  Architecture: Conv1->ReLU->Pool1->Conv2->ReLU->Pool2->FC1->ReLU->FC2->ReLU->FC3" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // ========== Set Fixed Random Seed for Reproducibility ==========
    srand(42);  // Fixed seed ensures same weights across all implementations
    std::cout << "Random seed: 42 (for reproducible weights)" << std::endl << std::endl;

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

    // ========== Sample MNIST Input ==========
    std::cout << "Creating sample MNIST input..." << std::endl;
    std::vector<std::vector<double>> mnistInput = {
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
    std::cout << "Sample input created (28x28)" << std::endl;

    // ========== Setup Crypto Context for Scheme Switching ==========
    std::cout << "\nSetting up crypto context..." << std::endl;

    ScalingTechnique scTech = FLEXIBLEAUTO;
    uint32_t multDepth = 25;  // Need more depth for LeNet-5
    if (scTech == FLEXIBLEAUTOEXT)
        multDepth += 1;

    uint32_t scaleModSize = 50;
    uint32_t firstModSize = 60;
    uint32_t ringDim = 8192;
    SecurityLevel sl = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE = 25;
    uint32_t slots = 4096;
    uint32_t batchSize = slots;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetScalingTechnique(scTech);
    parameters.SetSecurityLevel(sl);
    parameters.SetRingDim(ringDim);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetNumLargeDigits(3);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(SCHEMESWITCH);

    std::cout << "CKKS scheme using ring dimension " << cc->GetRingDimension() << std::endl;
    std::cout << "Number of slots: " << slots << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl;

    // ========== Key Generation ==========
    std::cout << "\nGenerating keys..." << std::endl;
    TimeVar t;
    TIC(t);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // Setup scheme switching
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
    double scaleSignFHEW = 4.0;
    cc->EvalCompareSwitchPrecompute(pLWE, scaleSignFHEW);

    std::cout << "Key generation time: " << TOC(t) << " ms" << std::endl;

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

    uint32_t dense2Input = dense1Output;  // 120
    uint32_t dense2Output = 84;
    std::vector<std::vector<double>> dense2Weights(dense2Output, std::vector<double>(dense2Input, 0.0));
    for (uint32_t i = 0; i < dense2Output; i++) {
        for (uint32_t j = 0; j < dense2Input; j++) {
            dense2Weights[i][j] = (rand() % 200 - 100) / 200.0;
        }
    }
    std::cout << "  Dense2: " << dense2Input << " -> " << dense2Output << std::endl;

    uint32_t dense3Input = dense2Output;  // 84
    uint32_t dense3Output = 10;
    std::vector<std::vector<double>> dense3Weights(dense3Output, std::vector<double>(dense3Input, 0.0));
    for (uint32_t i = 0; i < dense3Output; i++) {
        for (uint32_t j = 0; j < dense3Input; j++) {
            dense3Weights[i][j] = (rand() % 200 - 100) / 200.0;
        }
    }
    std::cout << "  Dense3: " << dense3Input << " -> " << dense3Output << std::endl;

    // ========== Build Toeplitz matrices and pack into diagonals ==========
    std::cout << "\nPreparing network weights..." << std::endl;
    TIC(t);

    // Conv1 Toeplitz
    auto toeplitzConv1 = ConstructConv2DToeplitz(conv1Kernel, 28, 28, 1, 0, 1, 1, 1, 1);
    std::vector<std::vector<double>> conv1Diagonals = PackMatDiagWise(toeplitzConv1, batchSize);
    std::size_t conv1Cols = conv1Diagonals.size();
    std::vector<int32_t> conv1Rotations = getOptimalRots(conv1Diagonals, true);
    std::cout << "  Conv1 Toeplitz: " << conv1Cols << " rows, "
              << conv1Rotations.size() << " non-zero diagonals" << std::endl;

    // AvgPool1 Toeplitz
    auto toeplitzPool1 = ConstructConv2DToeplitz(avgpool1Kernel, 24, 24, 2, 0, 1, 1, 1, 1);
    std::vector<std::vector<double>> pool1Diagonals = PackMatDiagWise(toeplitzPool1, batchSize);
    std::size_t pool1Cols = pool1Diagonals.size();
    std::vector<int32_t> pool1Rotations = getOptimalRots(pool1Diagonals, true);
    std::cout << "  AvgPool1 Toeplitz: " << pool1Cols << " rows, "
              << pool1Rotations.size() << " rotation keys needed" << std::endl;

    // Conv2 Toeplitz
    auto toeplitzConv2 = ConstructConv2DToeplitz(conv2Kernel, 12, 12, 1, 0, 1, 1, 1, 1);
    std::vector<std::vector<double>> conv2Diagonals = PackMatDiagWise(toeplitzConv2, batchSize);
    std::size_t conv2Cols = conv2Diagonals.size();
    std::vector<int32_t> conv2Rotations = getOptimalRots(conv2Diagonals, true);
    std::cout << "  Conv2 Toeplitz: " << conv2Cols << " rows, "
              << conv2Rotations.size() << " rotation keys needed" << std::endl;

    // AvgPool2 Toeplitz
    auto toeplitzPool2 = ConstructConv2DToeplitz(avgpool2Kernel, 8, 8, 2, 0, 1, 1, 1, 1);
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

    // Encode weight diagonals as plaintexts
    auto ptConv1Diags = MakeCKKSPackedPlaintextVectors(cc, conv1Diagonals);
    auto ptPool1Diags = MakeCKKSPackedPlaintextVectors(cc, pool1Diagonals);
    auto ptConv2Diags = MakeCKKSPackedPlaintextVectors(cc, conv2Diagonals);
    auto ptPool2Diags = MakeCKKSPackedPlaintextVectors(cc, pool2Diagonals);
    auto ptDense1Diags = MakeCKKSPackedPlaintextVectors(cc, dense1Diagonals);
    auto ptDense2Diags = MakeCKKSPackedPlaintextVectors(cc, dense2Diagonals);
    auto ptDense3Diags = MakeCKKSPackedPlaintextVectors(cc, dense3Diagonals);

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
    auto clearConv1_3D = CleartextConv2D(mnistInput3D, conv1Kernel, 1, 0);
    auto clearConv1 = CleartextFlatten(clearConv1_3D);

    // Cleartext ReLU1
    auto clearReLU1 = CleartextReLU(clearConv1);

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
    auto clearConv2_3D = CleartextConv2D(clearPool1_3D, conv2Kernel, 1, 0);
    auto clearConv2 = CleartextFlatten(clearConv2_3D);

    // Cleartext ReLU2
    auto clearReLU2 = CleartextReLU(clearConv2);

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
    auto clearDense1 = CleartextDense(clearPool2, dense1Weights);

    // Cleartext ReLU3
    auto clearReLU3 = CleartextReLU(clearDense1);

    // Cleartext Dense2
    auto clearDense2 = CleartextDense(clearReLU3, dense2Weights);

    // Cleartext ReLU4
    auto clearReLU4 = CleartextReLU(clearDense2);

    // Cleartext Dense3
    auto clearDense3 = CleartextDense(clearReLU4, dense3Weights);

    std::cout << "Cleartext forward pass complete!" << std::endl;
    // ========== END CLEARTEXT FORWARD PASS ==========

    // ========== Forward Pass ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Starting LeNet-5 encrypted inference..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Layer 1: Conv1
    std::cout << "\n[Layer 1] Conv1 (28x28x1 -> 24x24x6)..." << std::endl;
    TIC(t);
    // ctInput = cc->EvalRotate(ctInput, -conv1Cols);
    auto ctConv1 = EvalMultMatVecDiag(ctInput, ptConv1Diags, 2, conv1Rotations);
    double conv1Time = TOC(t);
    std::cout << "  Time: " << conv1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctConv1->GetLevel() << std::endl;

    // Validate Conv1
    Plaintext ptConv1Result;
    cc->Decrypt(keys.secretKey, ctConv1, &ptConv1Result);
    ptConv1Result->SetLength(conv1FlatSize);
    std::vector<double> encConv1 = ptConv1Result->GetRealPackedValue();
    CompareVectors(clearConv1, encConv1, "Conv1", 1e-1);

    // Layer 2: ReLU1
    std::cout << "\n[Layer 2] ReLU1 (scheme switching)..." << std::endl;
    TIC(t);
    auto ctReLU1 = EvalReLUSchemeSwitching(cc, ctConv1, keys.publicKey, conv1FlatSize, slots, scaleSignFHEW);
    double relu1Time = TOC(t);
    std::cout << "  Time: " << relu1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU1->GetLevel() << std::endl;

    // Validate ReLU1
    Plaintext ptReLU1Result;
    cc->Decrypt(keys.secretKey, ctReLU1, &ptReLU1Result);
    ptReLU1Result->SetLength(conv1FlatSize);
    std::vector<double> encReLU1 = ptReLU1Result->GetRealPackedValue();
    CompareVectors(clearReLU1, encReLU1, "ReLU1", 1e-1);

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

    // Layer 4: Conv2
    std::cout << "\n[Layer 4] Conv2 (12x12x6 -> 8x8x16)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctPool1, cc->EvalRotate(ctPool1, -conv2Cols));
    cc->EvalAddInPlace(ctPool1, cc->EvalRotate(ctPool1, -conv2Cols * 2));
    auto ctConv2 = EvalMultMatVecDiag(ctPool1, ptConv2Diags, 2, conv2Rotations);
    double conv2Time = TOC(t);
    std::cout << "  Time: " << conv2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctConv2->GetLevel() << std::endl;

    // Validate Conv2
    Plaintext ptConv2Result;
    cc->Decrypt(keys.secretKey, ctConv2, &ptConv2Result);
    ptConv2Result->SetLength(conv2FlatSize);
    std::vector<double> encConv2 = ptConv2Result->GetRealPackedValue();
    CompareVectors(clearConv2, encConv2, "Conv2", 1e-1);

    // Layer 5: ReLU2
    std::cout << "\n[Layer 5] ReLU2 (scheme switching)..." << std::endl;
    TIC(t);
    auto ctReLU2 = EvalReLUSchemeSwitching(cc, ctConv2, keys.publicKey, conv2FlatSize, slots, scaleSignFHEW);
    double relu2Time = TOC(t);
    std::cout << "  Time: " << relu2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU2->GetLevel() << std::endl;

    // Validate ReLU2
    Plaintext ptReLU2Result;
    cc->Decrypt(keys.secretKey, ctReLU2, &ptReLU2Result);
    ptReLU2Result->SetLength(conv2FlatSize);
    std::vector<double> encReLU2 = ptReLU2Result->GetRealPackedValue();
    CompareVectors(clearReLU2, encReLU2, "ReLU2", 1e-1);

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

    // Layer 7: Dense1
    std::cout << "\n[Layer 7] Dense1 (256 -> 120)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctPool2, cc->EvalRotate(ctPool2, -dense1Cols));
    auto ctDense1 = EvalMultMatVecDiag(ctPool2, ptDense1Diags, 2, dense1Rotations);
    double dense1Time = TOC(t);
    std::cout << "  Time: " << dense1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctDense1->GetLevel() << std::endl;

    // VALIDATION: Dense1
    Plaintext ptDense1Result;
    cc->Decrypt(keys.secretKey, ctDense1, &ptDense1Result);
    ptDense1Result->SetLength(dense1Output);
    std::vector<double> encDense1 = ptDense1Result->GetRealPackedValue();
    CompareVectors(clearDense1, encDense1, "Dense1", 1e-1);

    // Layer 8: ReLU3
    std::cout << "\n[Layer 8] ReLU3 (scheme switching)..." << std::endl;
    TIC(t);
    auto ctReLU3 = EvalReLUSchemeSwitching(cc, ctDense1, keys.publicKey, dense1Output, slots, scaleSignFHEW);
    double relu3Time = TOC(t);
    std::cout << "  Time: " << relu3Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU3->GetLevel() << std::endl;

    // VALIDATION: ReLU3
    Plaintext ptReLU3Result;
    cc->Decrypt(keys.secretKey, ctReLU3, &ptReLU3Result);
    ptReLU3Result->SetLength(dense1Output);
    std::vector<double> encReLU3 = ptReLU3Result->GetRealPackedValue();
    CompareVectors(clearReLU3, encReLU3, "ReLU3", 1e-1);

    // Layer 9: Dense2
    std::cout << "\n[Layer 9] Dense2 (120 -> 84)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU3, cc->EvalRotate(ctReLU3, -dense2Cols));
    auto ctDense2 = EvalMultMatVecDiag(ctReLU3, ptDense2Diags, 2, dense2Rotations);
    double dense2Time = TOC(t);
    std::cout << "  Time: " << dense2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctDense2->GetLevel() << std::endl;

    // VALIDATION: Dense2
    Plaintext ptDense2Result;
    cc->Decrypt(keys.secretKey, ctDense2, &ptDense2Result);
    ptDense2Result->SetLength(dense2Output);
    std::vector<double> encDense2 = ptDense2Result->GetRealPackedValue();
    CompareVectors(clearDense2, encDense2, "Dense2", 1e-1);

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

    // Layer 10: ReLU4
    std::cout << "\n[Layer 10] ReLU4 (scheme switching)..." << std::endl;
    TIC(t);
    auto ctReLU4 = EvalReLUSchemeSwitching(cc, ctDense2, keys.publicKey, dense2Output, slots, scaleSignFHEW);
    double relu4Time = TOC(t);
    std::cout << "  Time: " << relu4Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU4->GetLevel() << std::endl;

    // VALIDATION: ReLU4
    Plaintext ptReLU4Result;
    cc->Decrypt(keys.secretKey, ctReLU4, &ptReLU4Result);
    ptReLU4Result->SetLength(dense2Output);
    std::vector<double> encReLU4 = ptReLU4Result->GetRealPackedValue();
    CompareVectors(clearReLU4, encReLU4, "ReLU4", 1e-1);

    // Layer 11: Dense3
    std::cout << "\n[Layer 11] Dense3 (84 -> 10)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU4, cc->EvalRotate(ctReLU4, -dense3Cols));
    auto ctOutput = EvalMultMatVecDiag(ctReLU4, ptDense3Diags, 2, dense3Rotations);
    double dense3Time = TOC(t);
    std::cout << "  Time: " << dense3Time << " ms" << std::endl;
    std::cout << "  Level: " << ctOutput->GetLevel() << std::endl;

    // VALIDATION: Dense3 (Final Output)
    Plaintext ptDense3Result;
    cc->Decrypt(keys.secretKey, ctOutput, &ptDense3Result);
    ptDense3Result->SetLength(dense3Output);
    std::vector<double> encDense3 = ptDense3Result->GetRealPackedValue();
    CompareVectors(clearDense3, encDense3, "Dense3 (Final)", 1e-1);

    double totalInferenceTime = conv1Time + relu1Time + pool1Time + conv2Time + relu2Time +
                                pool2Time + dense1Time + relu3Time + dense2Time + relu4Time + dense3Time;
    std::cout << "\nTotal inference time: " << totalInferenceTime << " ms" << std::endl;

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
    std::cout << std::left << std::setw(30) << "AvgPool1 (24x24x6->12x12x6)" << std::setw(15) << pool1Time << ctPool1->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "Conv2 (12x12x6->8x8x16)" << std::setw(15) << conv2Time << ctConv2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU2" << std::setw(15) << relu2Time << ctReLU2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "AvgPool2 (8x8x16->4x4x16)" << std::setw(15) << pool2Time << ctPool2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "Dense1 (256->120)" << std::setw(15) << dense1Time << ctDense1->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU3" << std::setw(15) << relu3Time << ctReLU3->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "Dense2 (120->84)" << std::setw(15) << dense2Time << ctDense2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU4" << std::setw(15) << relu4Time << ctReLU4->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "Dense3 (84->10)" << std::setw(15) << dense3Time << ctOutput->GetLevel() << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Total Inference" << std::setw(15) << totalInferenceTime << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\n✓ MNIST LeNet-5 Inference Complete (Scheme Switching)!" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        MNISTLeNet5Inference();
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
