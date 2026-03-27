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
#include <numeric>
#include <chrono>

using namespace openfhe_numpy;
using namespace lbcrypto;

// OpenFHE's TIC/TOC return 0 in Release mode (NDEBUG without PROFILE).
// Use our own wall-clock timing instead.
using SteadyClock = std::chrono::steady_clock;
using TimePoint   = SteadyClock::time_point;
#undef TIC
#undef TOC
#define TIC(t) (t) = SteadyClock::now()
#define TOC(t) (std::chrono::duration<double, std::milli>(SteadyClock::now() - (t)).count())

// ========== TUNABLE FHEW PARAMETERS ==========
constexpr uint32_t FHEW_RING_DIM = 503;    // FHEW LWE dimension
constexpr uint32_t FHEW_LOGQ     = 25;     // FHEW ciphertext modulus bits

// ========== ACTIVATION FUNCTION TYPE ==========
enum class ActivationType {
    SCHEME_SWITCH,
    CHEBYSHEV,
};

/**
 * @brief CIFAR-10 ResNet-20 (No BatchNorm) Encrypted Inference
 *
 * Input: 32x32x3 normalized CIFAR-10 image
 * conv1: 3x3, 3->16, s=1, p=1 -> 32x32x16 -> ReLU
 * Layer1: 3 BasicBlocks (16->16, 32x32)
 * Layer2: 3 BasicBlocks (16->32->32, 32x32->16x16)
 * Layer3: 3 BasicBlocks (32->64->64, 16x16->8x8)
 * AvgPool: 8x8 -> 1x1 (64)
 * FC: 64 -> 10
 *
 * Each BasicBlock:
 *   out = relu(conv1(x))
 *   out = conv2(out)
 *   out += shortcut(x)     // identity or 1x1 conv
 *   out = relu(out)
 */

// ========== HELPER FUNCTIONS ==========

uint32_t GetChebyDepthFromDegree(uint32_t degree) {
    if (degree >= 3 && degree <= 5) return 4;
    if (degree >= 6 && degree <= 13) return 5;
    if (degree >= 14 && degree <= 27) return 6;
    if (degree >= 28 && degree <= 59) return 7;
    if (degree >= 60 && degree <= 119) return 8;
    if (degree >= 120 && degree <= 247) return 9;
    if (degree >= 248 && degree <= 495) return 10;
    if (degree >= 496 && degree <= 1007) return 11;
    throw std::runtime_error("Chebyshev degree out of supported range");
}

void PrintBounds(const std::vector<double>& vec, const std::string& name) {
    double minVal = *std::min_element(vec.begin(), vec.end());
    double maxVal = *std::max_element(vec.begin(), vec.end());
    std::cout << "  " << name << " bounds: [" << std::fixed << std::setprecision(4)
              << minVal << ", " << maxVal << "]" << std::endl;
}

/**
 * @brief Prepare bias vector for conv layers (channel-replicated over spatial)
 */
std::vector<double> PrepareBiasVecConv(
    const std::vector<double>& bias,
    uint32_t channels,
    uint32_t spatialSize,
    uint32_t totalSize
) {
    std::vector<double> biasVec(totalSize, 0.0);
    for (uint32_t c = 0; c < channels; c++) {
        for (uint32_t s = 0; s < spatialSize; s++) {
            biasVec[c * spatialSize + s] = bias[c];
        }
    }
    return biasVec;
}

/**
 * @brief Prepare bias vector for dense/FC layer
 */
std::vector<double> PrepareBiasVecDense(
    const std::vector<double>& bias,
    uint32_t outputSize
) {
    std::vector<double> biasVec(outputSize, 0.0);
    for (size_t i = 0; i < bias.size() && i < outputSize; i++) {
        biasVec[i] = bias[i];
    }
    return biasVec;
}

Ciphertext<DCRTPoly> EvalReLUChebyshev(
    CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct,
    uint32_t polyDegree = 63,
    double lowerBound = -10.0,
    double upperBound = 10.0
) {
    return cc->EvalChebyshevFunction(
        [](double x) -> double { return std::max(0.0, x); },
        ct, lowerBound, upperBound, polyDegree);
}

Ciphertext<DCRTPoly> EvalReLUSchemeSwitching(
    CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct,
    const Ciphertext<DCRTPoly>& ctZero,
    uint32_t numSlots,
    uint32_t totalSlots,
    double scaleSign = 8.0
) {
    auto ctComparison = cc->EvalCompareSchemeSwitching(
        ct, ctZero, NextPow2(numSlots), totalSlots, 0, scaleSign);
    auto ctReLU = cc->EvalMult(ct, cc->EvalSub(1, ctComparison));
    return ctReLU;
}

Ciphertext<DCRTPoly> EvalActivation(
    CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct,
    ActivationType activationType,
    const Ciphertext<DCRTPoly>& ctZero,
    uint32_t numSlots = 0,
    uint32_t totalSlots = 0,
    double scaleSign = 8.0,
    uint32_t chebyDegree = 63,
    double chebyLower = -10.0,
    double chebyUpper = 10.0
) {
    switch (activationType) {
        case ActivationType::SCHEME_SWITCH:
            return EvalReLUSchemeSwitching(cc, ct, ctZero, numSlots, totalSlots, scaleSign);
        case ActivationType::CHEBYSHEV:
            return EvalReLUChebyshev(cc, ct, chebyDegree, chebyLower, chebyUpper);
        default:
            throw std::runtime_error("Unknown activation type");
    }
}

/**
 * @brief Create a duplicated copy of a ciphertext for diagonal mat-vec product.
 * Returns a NEW ciphertext with shifted copies so that rotations wrap correctly.
 * The original ciphertext is NOT modified (safe for residual shortcut reuse).
 */
Ciphertext<DCRTPoly> MakeDuplicated(CryptoContext<DCRTPoly>& cc,
                                     const Ciphertext<DCRTPoly>& ct,
                                     uint32_t inputFlatSize, uint32_t numSlots) {
    if (inputFlatSize >= numSlots) return ct;

    uint32_t copiesNeeded = (numSlots + inputFlatSize - 1) / inputFlatSize;
    if (copiesNeeded <= 1) return ct;

    uint32_t numDups = 0;
    uint32_t covered = 1;
    while (covered < copiesNeeded) { covered *= 2; numDups++; }

    // Build duplicated copy using EvalAdd (not in-place) to preserve original
    auto result = ct;
    uint32_t shift = inputFlatSize;
    for (uint32_t d = 0; d < numDups; d++) {
        result = cc->EvalAdd(result, cc->EvalRotate(result, -(int32_t)shift));
        shift *= 2;
    }
    return result;
}

// ========== CLEARTEXT VALIDATION FUNCTIONS ==========

std::vector<std::vector<std::vector<double>>> CleartextConv2D(
    const std::vector<std::vector<std::vector<double>>>& input,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
    uint32_t stride = 1, uint32_t padding = 0,
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
        out_channels, std::vector<std::vector<double>>(output_height,
                      std::vector<double>(output_width, 0.0)));

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
                output[oc][oh][ow] = sum + (bias ? (*bias)[oc] : 0.0);
            }
        }
    }
    return output;
}

std::vector<double> CleartextFlatten(const std::vector<std::vector<std::vector<double>>>& input) {
    std::vector<double> output;
    for (const auto& ch : input)
        for (const auto& row : ch)
            for (double val : row)
                output.push_back(val);
    return output;
}

std::vector<std::vector<std::vector<double>>> CleartextReshape3D(
    const std::vector<double>& flat, uint32_t C, uint32_t H, uint32_t W
) {
    std::vector<std::vector<std::vector<double>>> out(C,
        std::vector<std::vector<double>>(H, std::vector<double>(W)));
    for (uint32_t c = 0; c < C; c++)
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t w = 0; w < W; w++)
                out[c][h][w] = flat[c * H * W + h * W + w];
    return out;
}

std::vector<double> CleartextReLU(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); i++)
        output[i] = std::max(0.0, input[i]);
    return output;
}

std::vector<double> CleartextAdd(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); i++)
        out[i] = a[i] + b[i];
    return out;
}

std::vector<double> CleartextGlobalAvgPool(const std::vector<double>& flat,
                                            uint32_t C, uint32_t H, uint32_t W) {
    std::vector<double> out(C, 0.0);
    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t w = 0; w < W; w++)
                out[c] += flat[c * H * W + h * W + w];
        out[c] /= (H * W);
    }
    return out;
}

std::vector<double> CleartextDense(
    const std::vector<double>& input,
    const std::vector<std::vector<double>>& weights,
    const std::vector<double>* bias = nullptr
) {
    std::vector<double> output(weights.size(), 0.0);
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < input.size(); j++)
            output[i] += weights[i][j] * input[j];
        if (bias) output[i] += (*bias)[i];
    }
    return output;
}

void CompareVectors(const std::vector<double>& cleartext,
                    const std::vector<double>& encrypted,
                    const std::string& layerName, uint32_t compareLen = 0) {
    uint32_t len = compareLen > 0 ? compareLen : std::min(cleartext.size(), encrypted.size());
    double maxError = 0.0, sumError = 0.0;
    for (uint32_t i = 0; i < len; ++i) {
        double error = std::abs(cleartext[i] - encrypted[i]);
        sumError += error;
        maxError = std::max(maxError, error);
    }
    std::cout << "    " << layerName << " max_err=" << std::scientific
              << std::setprecision(3) << maxError
              << " avg_err=" << sumError / len << std::endl;
}

// ========== BLOCK DATA STRUCTURE ==========

struct BlockToeplitzData {
    // Conv1 diagonals
    std::vector<std::vector<double>> conv1Diags;
    std::vector<bool> conv1NonZeros;
    std::vector<int32_t> conv1Rots;
    uint32_t conv1NumDiags;

    // Conv2 diagonals
    std::vector<std::vector<double>> conv2Diags;
    std::vector<bool> conv2NonZeros;
    std::vector<int32_t> conv2Rots;
    uint32_t conv2NumDiags;

    // Shortcut diagonals (optional)
    bool hasShortcut;
    std::vector<std::vector<double>> scDiags;
    std::vector<bool> scNonZeros;
    std::vector<int32_t> scRots;
    uint32_t scNumDiags;

    // Prepared bias
    std::vector<double> conv1BiasVec;
    std::vector<double> conv2BiasVec;
    std::vector<double> scBiasVec;

    // Dimensions
    uint32_t inputH, inputW, inputCh;
    uint32_t midH, midW, midCh;
    uint32_t outputH, outputW, outputCh;
    uint32_t inputFlatSize, midFlatSize, outputFlatSize;
};

/**
 * @brief Prepare Toeplitz data for one residual block
 */
BlockToeplitzData PrepareBlockData(
    const ResNet20BlockWeights& weights,
    uint32_t inputH, uint32_t inputW, uint32_t inputCh,
    uint32_t midCh, uint32_t stride,
    uint32_t batchSize, bool useBSGS
) {
    BlockToeplitzData data;
    data.inputH = inputH;
    data.inputW = inputW;
    data.inputCh = inputCh;
    data.midH = (inputH + 2 * 1 - 3) / stride + 1;  // padding=1, kernel=3
    data.midW = (inputW + 2 * 1 - 3) / stride + 1;
    data.midCh = midCh;
    data.outputH = data.midH;
    data.outputW = data.midW;
    data.outputCh = midCh;

    data.inputFlatSize = inputH * inputW * inputCh;
    data.midFlatSize = data.midH * data.midW * midCh;
    data.outputFlatSize = data.midFlatSize;

    // Conv1: inputCh -> midCh, 3x3, stride=stride, padding=1
    auto toeplitz1 = ConstructConv2DToeplitz(weights.conv1_weight, inputH, inputW, stride, 1, 1, 1, 1);
    data.conv1Diags = PackMatDiagWise(toeplitz1, batchSize);
    data.conv1NumDiags = data.conv1Diags.size();
    data.conv1NonZeros.resize(data.conv1NumDiags);
    data.conv1Rots = getOptimalRots(data.conv1Diags, &data.conv1NonZeros, useBSGS);
    data.conv1BiasVec = PrepareBiasVecConv(weights.conv1_bias, midCh,
                                            data.midH * data.midW, data.midFlatSize);

    // Conv2: midCh -> midCh, 3x3, stride=1, padding=1
    auto toeplitz2 = ConstructConv2DToeplitz(weights.conv2_weight, data.midH, data.midW, 1, 1, 1, 1, 1);
    data.conv2Diags = PackMatDiagWise(toeplitz2, batchSize);
    data.conv2NumDiags = data.conv2Diags.size();
    data.conv2NonZeros.resize(data.conv2NumDiags);
    data.conv2Rots = getOptimalRots(data.conv2Diags, &data.conv2NonZeros, useBSGS);
    data.conv2BiasVec = PrepareBiasVecConv(weights.conv2_bias, midCh,
                                            data.outputH * data.outputW, data.outputFlatSize);

    // Shortcut
    data.hasShortcut = weights.has_shortcut;
    data.scNumDiags = 0;
    if (weights.has_shortcut) {
        // 1x1 conv, stride=stride, padding=0
        auto toeplitzSC = ConstructConv2DToeplitz(weights.shortcut_weight, inputH, inputW, stride, 0, 1, 1, 1);
        data.scDiags = PackMatDiagWise(toeplitzSC, batchSize);
        data.scNumDiags = data.scDiags.size();
        data.scNonZeros.resize(data.scNumDiags);
        data.scRots = getOptimalRots(data.scDiags, &data.scNonZeros, useBSGS);
        data.scBiasVec = PrepareBiasVecConv(weights.shortcut_bias, midCh,
                                             data.outputH * data.outputW, data.outputFlatSize);
    }

    return data;
}

// ========== MAIN INFERENCE ==========

void CIFAR10ResNet20Inference(
    int sampleIndex = 0,
    ActivationType activationType = ActivationType::SCHEME_SWITCH,
    uint32_t ChebyDegree = 119,
    uint32_t ChebyMultDepth = 8,
    bool useOptimized = false,
    bool enableValidation = true,
    BINFHE_PARAMSET slBinParam = TOY
) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::string activationName = (activationType == ActivationType::SCHEME_SWITCH)
        ? "Scheme Switching" : "Chebyshev Approximation";
    std::cout << "  CIFAR-10 ResNet-20 Inference (" << activationName << ")" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // ========== Load CIFAR-10 Input ==========
    std::string cifarDataDir = "../openfhe_numpy/cpp/data/cifar10";
    std::string actualFile = "";
    int trueLabel = -1;

    for (int label = 0; label < 10; label++) {
        std::stringstream testPath;
        testPath << cifarDataDir << "/cifar10_" << sampleIndex << "_label_" << label << ".bin";
        std::ifstream testFile(testPath.str());
        if (testFile.good()) {
            actualFile = testPath.str();
            trueLabel = label;
            break;
        }
    }
    if (actualFile.empty()) {
        throw std::runtime_error("Could not find CIFAR-10 sample #" + std::to_string(sampleIndex) +
                                 ". Run train_resnet20_cifar10.py first!");
    }

    auto cifarInput3D = LoadCIFAR10Image(actualFile);
    std::cout << "Loaded sample #" << sampleIndex << ", true label: " << trueLabel << std::endl;

    // ========== Network Dimensions ==========
    // Layer sizes: (channels, height, width, flat_size)
    const uint32_t INPUT_CH = 3, INPUT_H = 32, INPUT_W = 32;
    const uint32_t CONV1_CH = 16;
    // Layer1: 16ch, 32x32
    // Layer2: 32ch, 16x16
    // Layer3: 64ch, 8x8
    const uint32_t L1_CH = 16, L1_H = 32, L1_W = 32;
    const uint32_t L2_CH = 32, L2_H = 16, L2_W = 16;
    const uint32_t L3_CH = 64, L3_H = 8, L3_W = 8;

    uint32_t conv1FlatSize = CONV1_CH * INPUT_H * INPUT_W;  // 16384
    uint32_t l3FlatSize = L3_CH * L3_H * L3_W;              // 4096

    // ========== Crypto Context Setup ==========
    std::cout << "\nSetting up crypto context..." << std::endl;

    ScalingTechnique scTech = FLEXIBLEAUTO;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;
    uint32_t ringDim = 32768;
    std::vector<uint32_t> levelBudget = {3, 3};
    std::vector<uint32_t> bsgsDim = {0, 0};
    SecurityLevel sl = HEStd_NotSet;
    BINFHE_PARAMSET slBin = slBinParam;
    uint32_t logQ_ccLWE = FHEW_LOGQ;
    uint32_t slots = 16384;  // Enough for max tensor 32x32x16=16384
    uint32_t batchSize = slots;

    // Depth calculation
    // ResNet-20: conv1 + relu + 9 blocks * (conv1 + relu + conv2 + [shortcut] + relu) + avgpool + fc
    // 9 blocks: 7 identity (4 ops each) + 2 downsample (5 ops each)
    uint32_t approxBootstrapDepth = FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    uint32_t multDepth = 1;
    bool enableBootstrapping = false;

    if (activationType == ActivationType::CHEBYSHEV) {
        // Each block: conv1(1) + relu(cheby) + conv2(1) + relu(cheby) = 2 + 2*chebyDepth
        // 9 blocks + initial conv + relu + avgpool + fc
        // = 1 + chebyDepth + 9*(2 + 2*chebyDepth) + 1 + 1
        uint32_t option1 = 1 + ChebyMultDepth + 9 * (2 + 2 * ChebyMultDepth) + 1 + 1;
        // With bootstrap: max(depth between bootstraps) + bootstrap depth + 1
        uint32_t maxLayerDepth = std::max(ChebyMultDepth + 2, ChebyMultDepth + 1);
        uint32_t option2 = maxLayerDepth + approxBootstrapDepth + 1;
        enableBootstrapping = (option2 + 3 < option1);
        multDepth = enableBootstrapping ? option2 : option1;
    } else if (activationType == ActivationType::SCHEME_SWITCH) {
        // First ReLU (InitReLU) consumes ~13 levels for SS infrastructure (one-time).
        // After that, each operation costs 1 level.
        // levelsAfterBootstrap: usable levels above approxBootstrapDepth.
        // Must be >= 14 so InitReLU can run (1 conv + 13 SS setup).
        // After first bootstrap, only 1-2 levels needed between bootstraps.
        uint32_t levelsAfterBootstrap = 10;
        multDepth = approxBootstrapDepth + levelsAfterBootstrap;
        enableBootstrapping = true;
        std::cout << "levelsAfterBootstrap=" << levelsAfterBootstrap
                  << ", approxBootstrapDepth=" << approxBootstrapDepth
                  << ", multDepth=" << multDepth << std::endl;
    }

    std::cout << "Multiplicative depth: " << multDepth << std::endl;
    std::cout << "Bootstrapping: " << (enableBootstrapping ? "enabled" : "disabled") << std::endl;

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
    if (enableBootstrapping) cc->Enable(FHE);
    if (activationType == ActivationType::SCHEME_SWITCH) cc->Enable(SCHEMESWITCH);

    std::cout << "Ring dimension: " << cc->GetRingDimension() << std::endl;
    std::cout << "Slots: " << slots << std::endl;

    // Print total logQ
    auto moduli = cc->GetCryptoParameters()->GetElementParams()->GetParams();
    uint32_t actualLogQ = 0;
    for (size_t i = 0; i < moduli.size(); i++)
        actualLogQ += moduli[i]->GetModulus().GetMSB();
    std::cout << "Total logQ: " << actualLogQ << " bits (" << moduli.size() << " moduli)" << std::endl;

    // ========== Key Generation ==========
    std::cout << "\nGenerating keys..." << std::endl;
    TimePoint t;
    TIC(t);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    if (enableBootstrapping) {
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, slots);
        cc->EvalBootstrapKeyGen(keys.secretKey, slots);
    }

    double scaleSignFHEW = 1.0;
    Ciphertext<DCRTPoly> ctZero;  // Precomputed encrypted zero
    if (activationType == ActivationType::SCHEME_SWITCH) {
        SchSwchParams params;
        params.SetSecurityLevelCKKS(HEStd_128_classic);
        params.SetSecurityLevelFHEW(slBin);
        params.SetRingDimension(FHEW_RING_DIM);
        params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE);
        params.SetNumSlotsCKKS(slots);
        params.SetNumValues(conv1FlatSize);  // Max tensor = 16384

        auto privateKeyFHEW = cc->EvalSchemeSwitchingSetup(params);
        auto ccLWE = cc->GetBinCCForSchemeSwitch();
        ccLWE->BTKeyGen(privateKeyFHEW);
        cc->EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW);

        auto modulus_LWE = 1 << logQ_ccLWE;
        auto beta = ccLWE->GetBeta().ConvertToInt();
        auto pLWE = modulus_LWE / (2 * beta);
        cc->EvalCompareSwitchPrecompute(pLWE, scaleSignFHEW);

        // Precompute encrypted zero once (reused across all 19 ReLU calls)
        std::vector<double> zeros(slots, 0.0);
        Plaintext ptxtZero = cc->MakeCKKSPackedPlaintext(zeros, 1, 0, nullptr, slots);
        ctZero = cc->Encrypt(keys.publicKey, ptxtZero);
        std::cout << "  Precomputed ctZero, pLWE=" << pLWE << std::endl;
    }
    std::cout << "Key generation time: " << TOC(t) << " ms" << std::endl;

    // ========== Load Weights ==========
    std::cout << "\nLoading ResNet-20 weights..." << std::endl;
    std::string weightsDir = "../openfhe_numpy/cpp/models/resnet20_weight_relu";
    auto W = LoadResNet20Weights(weightsDir);
    std::cout << "  conv1.weight: [" << W.conv1_weight.size() << "," << W.conv1_weight[0].size()
              << "," << W.conv1_weight[0][0].size() << "," << W.conv1_weight[0][0][0].size() << "]" << std::endl;

    // ========== Prepare Toeplitz Matrices ==========
    std::cout << "\nPreparing Toeplitz matrices..." << std::endl;
    TIC(t);

    uint32_t hoistingMode = useOptimized ? 2 : 1;

    // Initial conv1: 3->16, 3x3, s=1, p=1
    auto toeplitzInitConv = ConstructConv2DToeplitz(W.conv1_weight, INPUT_H, INPUT_W, 1, 1, 1, 1, 1);
    auto initConvDiags = PackMatDiagWise(toeplitzInitConv, batchSize);
    uint32_t initConvNumDiags = initConvDiags.size();
    std::vector<bool> initConvNonZeros(initConvNumDiags);
    auto initConvRots = getOptimalRots(initConvDiags, &initConvNonZeros, useOptimized);
    auto initConvBiasVec = PrepareBiasVecConv(W.conv1_bias, CONV1_CH, INPUT_H * INPUT_W, conv1FlatSize);
    std::cout << "  InitConv Toeplitz: " << initConvNumDiags << " diags, "
              << initConvRots.size() << " rots" << std::endl;

    // Prepare all 9 residual blocks
    std::vector<BlockToeplitzData> blocks(9);

    // Layer 1: 3 blocks, 16->16, stride=1, 32x32
    for (int i = 0; i < 3; i++) {
        blocks[i] = PrepareBlockData(W.layer1[i], L1_H, L1_W, L1_CH, L1_CH, 1, batchSize, useOptimized);
        std::cout << "  Layer1.Block" << i << ": conv1=" << blocks[i].conv1NumDiags
                  << " conv2=" << blocks[i].conv2NumDiags << " diags" << std::endl;
    }

    // Layer 2: block 0 (16->32, stride=2), blocks 1-2 (32->32, stride=1)
    blocks[3] = PrepareBlockData(W.layer2[0], L1_H, L1_W, L1_CH, L2_CH, 2, batchSize, useOptimized);
    std::cout << "  Layer2.Block0: conv1=" << blocks[3].conv1NumDiags
              << " conv2=" << blocks[3].conv2NumDiags
              << " sc=" << blocks[3].scNumDiags << " diags" << std::endl;
    for (int i = 1; i < 3; i++) {
        blocks[3+i] = PrepareBlockData(W.layer2[i], L2_H, L2_W, L2_CH, L2_CH, 1, batchSize, useOptimized);
        std::cout << "  Layer2.Block" << i << ": conv1=" << blocks[3+i].conv1NumDiags
                  << " conv2=" << blocks[3+i].conv2NumDiags << " diags" << std::endl;
    }

    // Layer 3: block 0 (32->64, stride=2), blocks 1-2 (64->64, stride=1)
    blocks[6] = PrepareBlockData(W.layer3[0], L2_H, L2_W, L2_CH, L3_CH, 2, batchSize, useOptimized);
    std::cout << "  Layer3.Block0: conv1=" << blocks[6].conv1NumDiags
              << " conv2=" << blocks[6].conv2NumDiags
              << " sc=" << blocks[6].scNumDiags << " diags" << std::endl;
    for (int i = 1; i < 3; i++) {
        blocks[6+i] = PrepareBlockData(W.layer3[i], L3_H, L3_W, L3_CH, L3_CH, 1, batchSize, useOptimized);
        std::cout << "  Layer3.Block" << i << ": conv1=" << blocks[6+i].conv1NumDiags
                  << " conv2=" << blocks[6+i].conv2NumDiags << " diags" << std::endl;
    }

    // Global AvgPool: 64ch, 8x8 -> 1x1 (as conv 64->64, 8x8, stride=8)
    std::vector<std::vector<std::vector<std::vector<double>>>> avgpoolKernel(L3_CH);
    for (uint32_t oc = 0; oc < L3_CH; oc++) {
        avgpoolKernel[oc].resize(L3_CH);
        for (uint32_t ic = 0; ic < L3_CH; ic++) {
            avgpoolKernel[oc][ic].resize(L3_H, std::vector<double>(L3_W, 0.0));
            if (oc == ic) {
                double val = 1.0 / (L3_H * L3_W);  // 1/64
                for (uint32_t kh = 0; kh < L3_H; kh++)
                    for (uint32_t kw = 0; kw < L3_W; kw++)
                        avgpoolKernel[oc][ic][kh][kw] = val;
            }
        }
    }
    auto toeplitzAvgPool = ConstructConv2DToeplitz(avgpoolKernel, L3_H, L3_W, L3_H, 0, 1, 1, 1);
    auto avgpoolDiags = PackMatDiagWise(toeplitzAvgPool, batchSize);
    uint32_t avgpoolNumDiags = avgpoolDiags.size();
    std::vector<bool> avgpoolNonZeros(avgpoolNumDiags);
    auto avgpoolRots = getOptimalRots(avgpoolDiags, &avgpoolNonZeros, useOptimized);
    std::cout << "  AvgPool Toeplitz: " << avgpoolNumDiags << " diags, "
              << avgpoolRots.size() << " rots" << std::endl;

    // FC: 64 -> 10
    auto fcDiags = PackMatDiagWise(W.fc_weight, batchSize);
    uint32_t fcNumDiags = fcDiags.size();
    std::vector<bool> fcNonZeros(fcNumDiags);
    auto fcRots = getOptimalRots(fcDiags, &fcNonZeros, useOptimized);
    auto fcBiasVec = PrepareBiasVecDense(W.fc_bias, 10);
    std::cout << "  FC Toeplitz: " << fcNumDiags << " diags, "
              << fcRots.size() << " rots" << std::endl;

    // Collect ALL rotation indices
    std::vector<int32_t> allRotations;
    auto addRots = [&](const std::vector<int32_t>& rots) {
        allRotations.insert(allRotations.end(), rots.begin(), rots.end());
    };
    addRots(initConvRots);
    for (auto& blk : blocks) {
        addRots(blk.conv1Rots);
        addRots(blk.conv2Rots);
        if (blk.hasShortcut) addRots(blk.scRots);
    }
    addRots(avgpoolRots);
    addRots(fcRots);

    // Add duplication rotation keys
    // We need negative shifts for input duplication
    auto addDupRots = [&](uint32_t flatSize) {
        uint32_t shift = flatSize;
        uint32_t copiesNeeded = (slots + flatSize - 1) / flatSize;
        uint32_t covered = 1;
        while (covered < copiesNeeded) {
            allRotations.push_back(-(int32_t)shift);
            shift *= 2;
            covered *= 2;
        }
    };
    // Initial input duplication is handled by EncodeMatrix, not rotation
    // But subsequent layers need duplication rotations
    addDupRots(conv1FlatSize);   // for layer1 block inputs (16384)
    addDupRots(L2_CH * L2_H * L2_W);  // for layer2 block inputs (8192)
    addDupRots(L3_CH * L3_H * L3_W);  // for layer3 block inputs (4096)
    addDupRots(64);              // for FC input

    // Deduplicate
    std::sort(allRotations.begin(), allRotations.end());
    allRotations.erase(std::unique(allRotations.begin(), allRotations.end()), allRotations.end());

    std::cout << "  Total unique rotation keys: " << allRotations.size() << std::endl;
    std::cout << "  Generating rotation keys..." << std::endl;
    cc->EvalRotateKeyGen(keys.secretKey, allRotations);

    std::cout << "Weight preparation time: " << TOC(t) << " ms" << std::endl;

    // ========== Encrypt Input ==========
    std::cout << "\nEncrypting input..." << std::endl;
    TIC(t);

    // Flatten CIFAR-10 image to CHW and replicate for initial conv
    // Reshape 3D to 2D for EncodeMatrix: [3*32, 32]
    std::vector<std::vector<double>> cifarInput2D(INPUT_CH * INPUT_H, std::vector<double>(INPUT_W));
    for (uint32_t c = 0; c < INPUT_CH; c++)
        for (uint32_t h = 0; h < INPUT_H; h++)
            for (uint32_t w = 0; w < INPUT_W; w++)
                cifarInput2D[c * INPUT_H + h][w] = cifarInput3D[c][h][w];

    // Replicate input to fill all slots (capped at slot count)
    auto flatInput = EncodeMatrix(cifarInput2D, slots);
    auto ptInput = cc->MakeCKKSPackedPlaintext(flatInput);
    auto ctInput = cc->Encrypt(keys.publicKey, ptInput);
    std::cout << "Input encryption time: " << TOC(t) << " ms" << std::endl;
    std::cout << "Initial level: " << ctInput->GetLevel() << std::endl;

    // ========== Cleartext Forward Pass ==========
    std::vector<double> clearCurrent;
    std::vector<double> clearFinal;

    // Track per-layer cleartext outputs for Chebyshev bounds
    struct ClearBlockResult {
        std::vector<double> afterConv1;
        std::vector<double> afterReLU1;
        std::vector<double> afterConv2;
        std::vector<double> afterShortcut;
        std::vector<double> afterAdd;
        std::vector<double> afterReLU2;
    };
    std::vector<ClearBlockResult> clearBlocks(9);

    std::vector<double> clearInitConv, clearInitReLU;
    std::vector<double> clearAvgPool, clearFC;

    if (enableValidation) {
        std::cout << "\nComputing cleartext reference..." << std::endl;

        // Initial conv + relu
        auto clearConv3D = CleartextConv2D(cifarInput3D, W.conv1_weight, 1, 1, &W.conv1_bias);
        clearInitConv = CleartextFlatten(clearConv3D);
        clearInitReLU = CleartextReLU(clearInitConv);
        clearCurrent = clearInitReLU;

        // Process 9 blocks
        const ResNet20BlockWeights* blockWeightsArr[9] = {
            &W.layer1[0], &W.layer1[1], &W.layer1[2],
            &W.layer2[0], &W.layer2[1], &W.layer2[2],
            &W.layer3[0], &W.layer3[1], &W.layer3[2],
        };
        uint32_t blockInputH[9] = {32,32,32, 32,16,16, 16,8,8};
        uint32_t blockInputW[9] = {32,32,32, 32,16,16, 16,8,8};
        uint32_t blockInputCh[9] = {16,16,16, 16,32,32, 32,64,64};
        uint32_t blockMidCh[9] = {16,16,16, 32,32,32, 64,64,64};
        uint32_t blockStride[9] = {1,1,1, 2,1,1, 2,1,1};

        for (int b = 0; b < 9; b++) {
            auto& bw = *blockWeightsArr[b];
            uint32_t iH = blockInputH[b], iW = blockInputW[b], iC = blockInputCh[b];
            uint32_t mC = blockMidCh[b];
            uint32_t st = blockStride[b];
            uint32_t oH = (iH + 2 - 3) / st + 1;
            uint32_t oW = (iW + 2 - 3) / st + 1;

            auto input3D = CleartextReshape3D(clearCurrent, iC, iH, iW);

            // Conv1
            auto conv1_3D = CleartextConv2D(input3D, bw.conv1_weight, st, 1, &bw.conv1_bias);
            clearBlocks[b].afterConv1 = CleartextFlatten(conv1_3D);

            // ReLU1
            clearBlocks[b].afterReLU1 = CleartextReLU(clearBlocks[b].afterConv1);

            // Conv2
            auto relu1_3D = CleartextReshape3D(clearBlocks[b].afterReLU1, mC, oH, oW);
            auto conv2_3D = CleartextConv2D(relu1_3D, bw.conv2_weight, 1, 1, &bw.conv2_bias);
            clearBlocks[b].afterConv2 = CleartextFlatten(conv2_3D);

            // Shortcut
            if (bw.has_shortcut) {
                auto sc_3D = CleartextConv2D(input3D, bw.shortcut_weight, st, 0, &bw.shortcut_bias);
                clearBlocks[b].afterShortcut = CleartextFlatten(sc_3D);
            } else {
                clearBlocks[b].afterShortcut = clearCurrent;
            }

            // Add
            clearBlocks[b].afterAdd = CleartextAdd(clearBlocks[b].afterConv2, clearBlocks[b].afterShortcut);

            // ReLU2
            clearBlocks[b].afterReLU2 = CleartextReLU(clearBlocks[b].afterAdd);
            clearCurrent = clearBlocks[b].afterReLU2;
        }

        // Global AvgPool
        clearAvgPool = CleartextGlobalAvgPool(clearCurrent, L3_CH, L3_H, L3_W);

        // FC
        clearFC = CleartextDense(clearAvgPool, W.fc_weight, &W.fc_bias);

        // Cleartext prediction
        auto maxIt = std::max_element(clearFC.begin(), clearFC.end());
        uint32_t clearPred = std::distance(clearFC.begin(), maxIt);
        std::cout << "  Cleartext prediction: " << clearPred << " (true: " << trueLabel << ")" << std::endl;
    }

    // ========== Encrypted Forward Pass ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Starting ResNet-20 encrypted inference..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    double totalTime = 0.0;
    auto ctCurrent = ctInput;

    // ----- Initial Conv -----
    std::cout << "\n[InitConv] 3x3, 3->16 (32x32->32x32)..." << std::endl;
    TIC(t);
    auto ptInitBias = cc->MakeCKKSPackedPlaintext(initConvBiasVec);
    ctCurrent = EvalMultMatVecDiag(ctCurrent, initConvDiags, hoistingMode,
                                    initConvRots, 0, &initConvNonZeros);
    ctCurrent = cc->EvalAdd(ctCurrent, ptInitBias);
    double initConvTime = TOC(t);
    totalTime += initConvTime;
    std::cout << "  Time: " << initConvTime << " ms, Level: " << ctCurrent->GetLevel() << std::endl;

    if (enableValidation) {
        Plaintext pt; cc->Decrypt(keys.secretKey, ctCurrent, &pt);
        pt->SetLength(conv1FlatSize);
        CompareVectors(clearInitConv, pt->GetRealPackedValue(), "InitConv", conv1FlatSize);
    }

    // ----- Initial ReLU -----
    std::cout << "\n[InitReLU] (" << activationName << ")..." << std::endl;
    TIC(t);
    double initLower = -10.0, initUpper = 10.0;
    if (enableValidation && !clearInitConv.empty()) {
        initLower = std::floor(*std::min_element(clearInitConv.begin(), clearInitConv.end()));
        initUpper = std::ceil(*std::max_element(clearInitConv.begin(), clearInitConv.end()));
    }
    ctCurrent = EvalActivation(cc, ctCurrent, activationType, ctZero,
                                conv1FlatSize, slots, scaleSignFHEW,
                                ChebyDegree, initLower, initUpper);
    double initReluTime = TOC(t);
    totalTime += initReluTime;
    std::cout << "  Time: " << initReluTime << " ms, Level: " << ctCurrent->GetLevel() << std::endl;

    if (enableValidation) {
        Plaintext pt; cc->Decrypt(keys.secretKey, ctCurrent, &pt);
        pt->SetLength(conv1FlatSize);
        CompareVectors(clearInitReLU, pt->GetRealPackedValue(), "InitReLU", conv1FlatSize);
    }

    // ----- 9 Residual Blocks -----
    const char* blockNames[9] = {
        "L1.B0", "L1.B1", "L1.B2",
        "L2.B0", "L2.B1", "L2.B2",
        "L3.B0", "L3.B1", "L3.B2"
    };

    for (int b = 0; b < 9; b++) {
        auto& blk = blocks[b];
        std::cout << "\n[" << blockNames[b] << "] "
                  << blk.inputCh << "ch " << blk.inputH << "x" << blk.inputW
                  << " -> " << blk.outputCh << "ch " << blk.outputH << "x" << blk.outputW
                  << (blk.hasShortcut ? " (downsample)" : "") << std::endl;

        // Save original for shortcut (before duplication modifies anything)
        auto ctShortcutInput = ctCurrent;

        // --- Conv1 ---
        TIC(t);
        // Duplicate for mat-vec wrapping (returns new ct, original preserved)
        auto ctDup1 = MakeDuplicated(cc, ctCurrent, blk.inputFlatSize, slots);
        auto ptConv1Bias = cc->MakeCKKSPackedPlaintext(blk.conv1BiasVec);
        ctCurrent = EvalMultMatVecDiag(ctDup1, blk.conv1Diags, hoistingMode,
                                        blk.conv1Rots, 0, &blk.conv1NonZeros);
        ctCurrent = cc->EvalAdd(ctCurrent, ptConv1Bias);
        double conv1Time = TOC(t);
        totalTime += conv1Time;
        std::cout << "  Conv1: " << conv1Time << " ms, L=" << ctCurrent->GetLevel();

        if (enableValidation) {
            Plaintext pt; cc->Decrypt(keys.secretKey, ctCurrent, &pt);
            pt->SetLength(blk.midFlatSize);
            CompareVectors(clearBlocks[b].afterConv1, pt->GetRealPackedValue(),
                          "Conv1", blk.midFlatSize);
        }

        // --- ReLU1 ---
        TIC(t);
        double lower1 = -10.0, upper1 = 10.0;
        if (enableValidation && !clearBlocks[b].afterConv1.empty()) {
            lower1 = std::floor(*std::min_element(clearBlocks[b].afterConv1.begin(),
                                                   clearBlocks[b].afterConv1.end()));
            upper1 = std::ceil(*std::max_element(clearBlocks[b].afterConv1.begin(),
                                                  clearBlocks[b].afterConv1.end()));
        }
        ctCurrent = EvalActivation(cc, ctCurrent, activationType, ctZero,
                                    blk.midFlatSize, slots, scaleSignFHEW,
                                    ChebyDegree, lower1, upper1);
        double relu1Time = TOC(t);
        totalTime += relu1Time;
        std::cout << " | ReLU1: " << relu1Time << " ms, L=" << ctCurrent->GetLevel();

        // Bootstrap if needed
        double bsTime = 0.0;
        if (enableBootstrapping) {
            uint32_t levelsLeft = multDepth - ctCurrent->GetLevel();
            uint32_t needed = (activationType == ActivationType::CHEBYSHEV)
                ? ChebyMultDepth + 2 : 3;
            if (levelsLeft <= needed + 1) {
                TIC(t);
                ctCurrent = cc->EvalBootstrap(ctCurrent);
                bsTime = TOC(t);
                totalTime += bsTime;
                std::cout << " | BS: " << bsTime << " ms";
            }
        }

        // --- Conv2 ---
        TIC(t);
        auto ctDup2 = MakeDuplicated(cc, ctCurrent, blk.midFlatSize, slots);
        auto ptConv2Bias = cc->MakeCKKSPackedPlaintext(blk.conv2BiasVec);
        ctCurrent = EvalMultMatVecDiag(ctDup2, blk.conv2Diags, hoistingMode,
                                        blk.conv2Rots, 0, &blk.conv2NonZeros);
        ctCurrent = cc->EvalAdd(ctCurrent, ptConv2Bias);
        double conv2Time = TOC(t);
        totalTime += conv2Time;
        std::cout << " | Conv2: " << conv2Time << " ms, L=" << ctCurrent->GetLevel();

        // --- Shortcut ---
        TIC(t);
        Ciphertext<DCRTPoly> ctShortcut;
        double scTime = 0.0;
        if (blk.hasShortcut) {
            auto ctScDup = MakeDuplicated(cc, ctShortcutInput, blk.inputFlatSize, slots);
            auto ptScBias = cc->MakeCKKSPackedPlaintext(blk.scBiasVec);
            ctShortcut = EvalMultMatVecDiag(ctScDup, blk.scDiags, hoistingMode,
                                             blk.scRots, 0, &blk.scNonZeros);
            ctShortcut = cc->EvalAdd(ctShortcut, ptScBias);
        } else {
            ctShortcut = ctShortcutInput;
        }
        scTime = TOC(t);
        totalTime += scTime;
        if (blk.hasShortcut)
            std::cout << " | SC: " << scTime << " ms, L=" << ctShortcut->GetLevel();

        // --- Residual Add ---
        TIC(t);
        ctCurrent = cc->EvalAdd(ctCurrent, ctShortcut);
        double addTime = TOC(t);
        totalTime += addTime;
        std::cout << " | Add: L=" << ctCurrent->GetLevel();

        if (enableValidation) {
            Plaintext pt; cc->Decrypt(keys.secretKey, ctCurrent, &pt);
            pt->SetLength(blk.outputFlatSize);
            CompareVectors(clearBlocks[b].afterAdd, pt->GetRealPackedValue(),
                          "Add", blk.outputFlatSize);
        }

        // --- ReLU2 (post-add) ---
        TIC(t);
        double lower2 = -10.0, upper2 = 10.0;
        if (enableValidation && !clearBlocks[b].afterAdd.empty()) {
            lower2 = std::floor(*std::min_element(clearBlocks[b].afterAdd.begin(),
                                                   clearBlocks[b].afterAdd.end()));
            upper2 = std::ceil(*std::max_element(clearBlocks[b].afterAdd.begin(),
                                                  clearBlocks[b].afterAdd.end()));
        }
        ctCurrent = EvalActivation(cc, ctCurrent, activationType, ctZero,
                                    blk.outputFlatSize, slots, scaleSignFHEW,
                                    ChebyDegree, lower2, upper2);
        double relu2Time = TOC(t);
        totalTime += relu2Time;
        std::cout << " | ReLU2: " << relu2Time << " ms, L=" << ctCurrent->GetLevel() << std::endl;

        // Bootstrap if needed
        if (enableBootstrapping) {
            uint32_t levelsLeft = multDepth - ctCurrent->GetLevel();
            uint32_t needed = (activationType == ActivationType::CHEBYSHEV)
                ? ChebyMultDepth + 2 : 3;
            if (levelsLeft <= needed + 1) {
                TIC(t);
                ctCurrent = cc->EvalBootstrap(ctCurrent);
                double bsTime2 = TOC(t);
                totalTime += bsTime2;
                std::cout << "  [Bootstrap] " << bsTime2 << " ms, L=" << ctCurrent->GetLevel() << std::endl;
            }
        }

        if (enableValidation) {
            Plaintext pt; cc->Decrypt(keys.secretKey, ctCurrent, &pt);
            pt->SetLength(blk.outputFlatSize);
            CompareVectors(clearBlocks[b].afterReLU2, pt->GetRealPackedValue(),
                          std::string(blockNames[b]) + " output", blk.outputFlatSize);
        }
    }

    // ----- Global Average Pooling -----
    std::cout << "\n[AvgPool] 8x8x64 -> 1x1x64..." << std::endl;
    TIC(t);
    auto ctDupPool = MakeDuplicated(cc, ctCurrent, l3FlatSize, slots);
    auto ctPooled = EvalMultMatVecDiag(ctDupPool, avgpoolDiags, hoistingMode,
                                        avgpoolRots, 0, &avgpoolNonZeros);
    double avgpoolTime = TOC(t);
    totalTime += avgpoolTime;
    std::cout << "  Time: " << avgpoolTime << " ms, Level: " << ctPooled->GetLevel() << std::endl;

    if (enableValidation) {
        Plaintext pt; cc->Decrypt(keys.secretKey, ctPooled, &pt);
        pt->SetLength(L3_CH);
        CompareVectors(clearAvgPool, pt->GetRealPackedValue(), "AvgPool", L3_CH);
    }

    // ----- FC Layer -----
    std::cout << "\n[FC] 64 -> 10..." << std::endl;
    TIC(t);
    auto ctDupFC = MakeDuplicated(cc, ctPooled, L3_CH, slots);
    auto ptFcBias = cc->MakeCKKSPackedPlaintext(fcBiasVec);
    auto ctOutput = EvalMultMatVecDiag(ctDupFC, fcDiags, hoistingMode,
                                        fcRots, 0, &fcNonZeros);
    ctOutput = cc->EvalAdd(ctOutput, ptFcBias);
    double fcTime = TOC(t);
    totalTime += fcTime;
    std::cout << "  Time: " << fcTime << " ms, Level: " << ctOutput->GetLevel() << std::endl;

    // ========== Decrypt and Display Results ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    TIC(t);
    Plaintext ptOutput;
    cc->Decrypt(keys.secretKey, ctOutput, &ptOutput);
    ptOutput->SetLength(10);
    auto outputVector = ptOutput->GetRealPackedValue();
    std::cout << "Decryption time: " << TOC(t) << " ms" << std::endl;

    std::cout << "\nOutput logits:" << std::endl;
    const char* classNames[] = {"plane", "car", "bird", "cat", "deer",
                                 "dog", "frog", "horse", "ship", "truck"};
    for (uint32_t i = 0; i < 10; i++) {
        std::cout << "  " << std::setw(6) << classNames[i] << ": "
                  << std::fixed << std::setprecision(4) << outputVector[i] << std::endl;
    }

    uint32_t predictedClass = 0;
    double maxLogit = outputVector[0];
    for (uint32_t i = 1; i < 10; i++) {
        if (outputVector[i] > maxLogit) {
            maxLogit = outputVector[i];
            predictedClass = i;
        }
    }

    std::cout << "\nPredicted: " << classNames[predictedClass]
              << " (class " << predictedClass << ")";
    if (trueLabel >= 0) {
        std::cout << " | True: " << classNames[trueLabel]
                  << " (class " << trueLabel << ")";
        std::cout << (predictedClass == (uint32_t)trueLabel ? " CORRECT" : " INCORRECT");
    }
    std::cout << std::endl;

    if (enableValidation && !clearFC.empty()) {
        CompareVectors(clearFC, outputVector, "Final output", 10);
    }

    std::cout << "\nTotal inference time: " << totalTime << " ms" << std::endl;
    std::cout << "\nResNet-20 CIFAR-10 Inference Complete!" << std::endl;
}

// ========== MAIN ==========

int main(int argc, char* argv[]) {
    try {
        int sampleIndex = 0;
        ActivationType activationType = ActivationType::SCHEME_SWITCH;
        uint32_t chebyDegree = 119;
        uint32_t chebyMultDepth = 8;
        bool useOptimized = true;
        bool enableValidation = true;
        BINFHE_PARAMSET slBin = TOY;

        if (argc > 1) {
            sampleIndex = std::atoi(argv[1]);
        }

        if (argc > 2) {
            std::string actStr = argv[2];
            if (actStr == "scheme") {
                activationType = ActivationType::SCHEME_SWITCH;
            } else if (actStr == "cheby") {
                activationType = ActivationType::CHEBYSHEV;
            } else {
                std::cerr << "Unknown activation: " << actStr << " (use: scheme, cheby)" << std::endl;
                return 1;
            }
        }

        if (argc > 3) {
            if (activationType == ActivationType::CHEBYSHEV) {
                chebyDegree = std::atoi(argv[3]);
                chebyMultDepth = GetChebyDepthFromDegree(chebyDegree);
                if (argc > 4) useOptimized = (std::atoi(argv[4]) != 0);
            } else {
                std::string secStr = argv[3];
                std::transform(secStr.begin(), secStr.end(), secStr.begin(), ::tolower);
                if (secStr == "toy") slBin = TOY;
                else if (secStr == "std128") slBin = STD128;
                else { std::cerr << "Unknown security: " << secStr << std::endl; return 1; }
                if (argc > 4) useOptimized = (std::atoi(argv[4]) != 0);
            }
        }

        CIFAR10ResNet20Inference(sampleIndex, activationType, chebyDegree,
                                  chebyMultDepth, useOptimized, enableValidation, slBin);
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
