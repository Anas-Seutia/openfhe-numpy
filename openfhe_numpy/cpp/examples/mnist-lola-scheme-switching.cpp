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
constexpr bool DEBUG_MODE = false;

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

    // ReLU(x) = x * (x > 0)
    // Step 1: Compute comparison result (x > 0)
    auto ctComparison = cc->EvalCompareSchemeSwitching(ct, ctZero, NextPow2(numSlots), totalSlots, 0, scaleSign);

    // Step 2: Multiply input by comparison result to get ReLU
    // The comparison returns 1 if x > 0, 0 otherwise
    // We need to invert: (1 - comparison) to get mask
    auto ctReLU = cc->EvalMult(ct, cc->EvalAdd(cc->EvalMult(ctComparison, -1), 1));

    return ctReLU;
}

void MNISTLoLaInference() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  MNIST LoLa Network Inference (Scheme Switching)" << std::endl;
    std::cout << "  Architecture: Conv -> ReLU -> Dense -> ReLU -> Dense" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // ========== Set Fixed Random Seed for Reproducibility ==========
    srand(42);  // Fixed seed ensures same weights across all implementations
    std::cout << "Random seed: 42 (for reproducible weights)" << std::endl << std::endl;

    // ========== Network Parameters ==========
    std::cout << "Network Architecture:" << std::endl;
    std::cout << "  Input: 28x28 MNIST image (1 channel)" << std::endl;
    std::cout << "  Conv: 5x5 kernel, 5 output channels, stride=2 -> 12x12x5" << std::endl;
    std::cout << "  ReLU: Scheme switching (CKKS-FHEW-CKKS)" << std::endl;
    std::cout << "  Dense1: 720 -> 64 neurons" << std::endl;
    std::cout << "  ReLU: Scheme switching" << std::endl;
    std::cout << "  Dense2: 64 -> 10 neurons (output)" << std::endl << std::endl;

    // ========== Sample MNIST Input (simplified) ==========
    // Create a simple test pattern (normally would load from MNIST dataset)
    std::cout << "Creating sample MNIST input..." << std::endl;
    std::vector<std::vector<double>> mnistInput(28, std::vector<double>(28, 0.0));

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

    std::cout << "Sample input created (28x28)" << std::endl;

    // ========== Setup Crypto Context for Scheme Switching ==========
    std::cout << "\nSetting up crypto context..." << std::endl;

    ScalingTechnique scTech = FLEXIBLEAUTO;
    uint32_t multDepth = 20;  // Need more depth for full network
    if (scTech == FLEXIBLEAUTOEXT)
        multDepth += 1;

    uint32_t scaleModSize = 50;
    uint32_t firstModSize = 60;
    uint32_t ringDim = 8192;  // Larger ring for MNIST
    SecurityLevel sl = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE = 25;
    uint32_t slots = 2048;  // Enough for 720 elements from conv output
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
    params.SetNumValues(720);

    auto privateKeyFHEW = cc->EvalSchemeSwitchingSetup(params);
    auto ccLWE = cc->GetBinCCForSchemeSwitch();
    ccLWE->BTKeyGen(privateKeyFHEW);
    cc->EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW);

    auto modulus_LWE = 1 << logQ_ccLWE;
    auto beta = ccLWE->GetBeta().ConvertToInt();
    auto pLWE = modulus_LWE / (2 * beta);
    double scaleSignFHEW = 8.0;
    cc->EvalCompareSwitchPrecompute(pLWE, scaleSignFHEW);

    std::cout << "Key generation time: " << TOC(t) << " ms" << std::endl;

    // ========== Define Network Weights (SPARSE kernels for memory efficiency) ==========
    std::cout << "\nInitializing network weights..." << std::endl;

    // Conv layer: 5 output channels, 1 input channel, 5x5 SPARSE kernel
    std::vector<std::vector<std::vector<std::vector<double>>>> convKernel(5);
    for (int oc = 0; oc < 5; oc++) {
        convKernel[oc].resize(1);  // 1 input channel
        convKernel[oc][0].resize(5, std::vector<double>(5, 0.0));  // Initialize to zeros
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    convKernel[oc][0][i][j] = (rand() % 200 - 100) / 100.0;
                }
            }
        // convKernel[0][0][0][0] = 1.0;   // Top-left
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
    PrintKernelDebug(convKernel, "Conv kernel");

    // Dense layer 1: 720 -> 64 (SPARSE for memory efficiency)
    uint32_t dense1Input = flattenedSize;
    uint32_t dense1Output = 64;
    std::vector<std::vector<double>> dense1Weights(dense1Output, std::vector<double>(dense1Input, 0.0));
    // Only connect each output to 10 random inputs (instead of all 720)
    for (uint32_t i = 0; i < dense1Output; i++) {
        for (uint32_t j = 0; j < dense1Input; j++) {
            dense1Weights[i][j] = (rand() % 200 - 100) / 200.0;
        }
        // dense1Weights[i][i] = 1.0;
    }
    std::cout << "  Dense1 shape: " << dense1Input << " -> " << dense1Output << " (SPARSE: 10 connections per neuron)" << std::endl;
    PrintWeightsDebug(dense1Weights, "Dense1 weights");

    // Dense layer 2: 64 -> 10
    uint32_t dense2Input = dense1Output;
    uint32_t dense2Output = 10;
    std::vector<std::vector<double>> dense2Weights(dense2Output, std::vector<double>(dense2Input, 0.0));
    for (uint32_t i = 0; i < dense2Output; i++) {
        for (uint32_t j = 0; j < dense2Input; j++) {
            dense2Weights[i][j] = (rand() % 200 - 100) / 200.0;
        }
        // dense2Weights[i][i] = 1.0;
    }
    std::cout << "  Dense2 shape: " << dense2Input << " -> " << dense2Output << std::endl;
    PrintWeightsDebug(dense2Weights, "Dense2 weights");

    // ========== Build Toeplitz matrices and pack into diagonals ==========
    std::cout << "\nPreparing encrypted network weights..." << std::endl;

    // Convolution layer
    TIC(t);
    auto toeplitzConv = ConstructConv2DToeplitz(convKernel, 28, 28, convStride, convPadding, 1, 1, 1, 1);
    std::vector<std::vector<double>> convDiagonals = PackMatDiagWise(toeplitzConv, batchSize);
    std::size_t convCols = convDiagonals.size();
    std::vector<int32_t> convRotations = getOptimalRots(convDiagonals, true);
    std::cout << "  Conv Toeplitz: " << convCols << " rows, "
              << convRotations.size() << " non-zero diagonals" << std::endl;

    // Dense layer 1
    std::vector<std::vector<double>> dense1Diagonals = PackMatDiagWise(dense1Weights, batchSize);
    std::vector<int32_t> dense1Rotations = getOptimalRots(dense1Diagonals, true);
    std::size_t dense1Cols = dense1Diagonals.size();
    std::cout << "  Dense1: " << dense1Rotations.size() << " non-zero diagonals" << std::endl;

    // Dense layer 2
    std::vector<std::vector<double>> dense2Diagonals = PackMatDiagWise(dense2Weights, batchSize);
    std::vector<int32_t> dense2Rotations = getOptimalRots(dense2Diagonals, true);
    std::size_t dense2Cols = dense2Diagonals.size();
    std::cout << "  Dense2: " << dense2Rotations.size() << " non-zero diagonals" << std::endl;

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
    std::cout << "  Rotation key generation complete!" << std::endl;

    // Encode weight diagonals as PLAINTEXTS (not encrypted - saves massive memory!)
    // For neural network inference, encrypted input + plaintext weights is standard
    auto ptConvDiags = MakeCKKSPackedPlaintextVectors(cc, convDiagonals);
    auto ptDense1Diags = MakeCKKSPackedPlaintextVectors(cc, dense1Diagonals);
    auto ptDense2Diags = MakeCKKSPackedPlaintextVectors(cc, dense2Diagonals);

    std::cout << "Weight preparation time: " << TOC(t) << " ms" << std::endl;
    std::cout << "  (Using plaintext weights)" << std::endl;

    // ========== Encrypt Input ==========
    std::cout << "\nEncrypting input..." << std::endl;
    TIC(t);
    std::vector<double> flatInput = EncodeMatrix(mnistInput, 784*2);
    auto ptInput = cc->MakeCKKSPackedPlaintext(flatInput);
    auto ctInput = cc->Encrypt(keys.publicKey, ptInput);
    std::cout << "Input encryption time: " << TOC(t) << " ms" << std::endl;
    std::cout << "Initial ciphertext level: " << ctInput->GetLevel() << std::endl;

    // ========== Forward Pass ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Starting encrypted inference..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Layer 1: Convolution
    std::cout << "\n[Layer 1] Convolution (28x28x1 -> 12x12x5)..." << std::endl;
    TIC(t);
    // cc->EvalAddInPlace(ctInput, cc->EvalRotate(ctInput, -convCols));
    auto ctConvOut = EvalMultMatVecDiag(ctInput, ptConvDiags, 2, convRotations);
    double convTime = TOC(t);
    std::cout << "  Time: " << convTime << " ms" << std::endl;
    std::cout << "  Level: " << ctConvOut->GetLevel() << std::endl;
    PrintDebugValues(cc, ctConvOut, keys.secretKey, "Conv output", 10, flattenedSize);
    
    // Layer 2: ReLU
    std::cout << "\n[Layer 2] ReLU (scheme switching)..." << std::endl;
    TIC(t);
    auto ctReLU1 = EvalReLUSchemeSwitching(cc, ctConvOut, keys.publicKey, 720, slots, scaleSignFHEW);
    double relu1Time = TOC(t);
    std::cout << "  Time: " << relu1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU1->GetLevel() << std::endl;
    PrintDebugValues(cc, ctReLU1, keys.secretKey, "ReLU1 output", 10, flattenedSize);
    
    // Layer 3: Dense 1 (720 -> 64)
    std::cout << "\n[Layer 3] Dense1 (720 -> 64)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU1, cc->EvalRotate(ctReLU1, -dense1Cols));
    auto ctDense1Out = EvalMultMatVecDiag(ctReLU1, ptDense1Diags, 2, dense1Rotations);
    double dense1Time = TOC(t);
    std::cout << "  Time: " << dense1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctDense1Out->GetLevel() << std::endl;
    PrintDebugValues(cc, ctDense1Out, keys.secretKey, "Dense1 output", 10, dense1Output);
    
    // Layer 4: ReLU
    std::cout << "\n[Layer 4] ReLU (scheme switching)..." << std::endl;
    TIC(t);
    auto ctReLU2 = EvalReLUSchemeSwitching(cc, ctDense1Out, keys.publicKey, 64, slots, scaleSignFHEW);
    double relu2Time = TOC(t);
    std::cout << "  Time: " << relu2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctReLU2->GetLevel() << std::endl;
    PrintDebugValues(cc, ctReLU2, keys.secretKey, "ReLU2 output", 10, dense1Output);
    
    // Layer 5: Dense 2 (64 -> 10)
    std::cout << "\n[Layer 5] Dense2 (64 -> 10)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU2, cc->EvalRotate(ctReLU2, -dense2Cols));
    auto ctOutput = EvalMultMatVecDiag(ctReLU2, ptDense2Diags, 2, dense2Rotations);
    double dense2Time = TOC(t);
    std::cout << "  Time: " << dense2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctOutput->GetLevel() << std::endl;
    PrintDebugValues(cc, ctOutput, keys.secretKey, "Final output", 10, dense2Output);

    double totalInferenceTime = convTime + relu1Time + dense1Time + relu2Time + dense2Time;
    std::cout << "\nTotal inference time: " << totalInferenceTime << " ms" << std::endl;

    // ========== Decrypt and Display Results ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Decrypting results..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    TIC(t);
    Plaintext ptOutput;
    cc->Decrypt(keys.secretKey, ctOutput, &ptOutput);
    ptOutput->SetLength(dense2Output);
    std::vector<double> outputVector = ptOutput->GetRealPackedValue();
    std::cout << "Decryption time: " << TOC(t) << " ms" << std::endl;

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

    std::cout << "\nPredicted class: " << predictedClass << std::endl;
    std::cout << "Confidence: " << maxLogit << std::endl;

    // ========== Performance Summary ==========
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Performance Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::left << std::setw(30) << "Layer" << std::setw(15) << "Time (ms)" << "Level" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Convolution" << std::setw(15) << convTime << ctConvOut->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU 1" << std::setw(15) << relu1Time << ctReLU1->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "Dense 1" << std::setw(15) << dense1Time << ctDense1Out->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU 2" << std::setw(15) << relu2Time << ctReLU2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "Dense 2" << std::setw(15) << dense2Time << ctOutput->GetLevel() << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Total Inference" << std::setw(15) << totalInferenceTime << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\n✓ MNIST LoLa Inference Complete (Scheme Switching)!" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        MNISTLoLaInference();
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
