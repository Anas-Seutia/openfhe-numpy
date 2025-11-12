#include "numpy_enc_matrix.h"
#include "openfhe.h"
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
 * @brief MNIST LoLa Network Architecture (Chebyshev Approximation for ReLU)
 *
 * Network: Conv -> ReLU -> Dense -> ReLU -> Dense
 * - Input: 28x28 MNIST image (1 channel)
 * - Conv: 5x5 kernel, 5 output channels, stride=2, no padding -> 12x12x5
 * - ReLU: Chebyshev approximation
 * - Dense1: 12x12x5 = 720 -> 64 neurons
 * - ReLU: Chebyshev approximation
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
 * @brief Perform dense (fully connected) layer using diagonal method
 * Using PLAINTEXT weights (not encrypted) to save memory
 */
Ciphertext<DCRTPoly> EvalDenseLayer(
    CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ctInput,
    const std::vector<Plaintext>& ptWeightDiags,
    std::vector<int32_t>& rotationIndices
) {
    return EvalMultMatVecDiag(ctInput, ptWeightDiags, 1, rotationIndices);
} 

void MNISTLoLaInference() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  MNIST LoLa Network Inference (Chebyshev Approximation)" << std::endl;
    std::cout << "  Architecture: Conv -> ReLU -> Dense -> ReLU -> Dense" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // ========== Set Fixed Random Seed for Reproducibility ==========
    srand(42);  // Fixed seed ensures same weights across all implementations
    std::cout << "Random seed: 42 (for reproducible weights)" << std::endl << std::endl;

    // ========== Network Parameters ==========
    std::cout << "Network Architecture:" << std::endl;
    std::cout << "  Input: 28x28 MNIST image (1 channel)" << std::endl;
    std::cout << "  Conv: 5x5 kernel, 5 output channels, stride=2 -> 12x12x5" << std::endl;
    std::cout << "  ReLU: Chebyshev approximation (degree 63)" << std::endl;
    std::cout << "  Dense1: 720 -> 64 neurons" << std::endl;
    std::cout << "  ReLU: Chebyshev approximation (degree 63)" << std::endl;
    std::cout << "  Dense2: 64 -> 10 neurons (output)" << std::endl << std::endl;

    // ========== Sample MNIST Input (simplified) ==========
    // Create a simple test pattern (normally would load from MNIST dataset)
    std::cout << "Creating sample MNIST input..." << std::endl;
    std::vector<std::vector<double>> mnistInput(28, std::vector<double>(28, 0.0));

    // Create a simple vertical edge pattern in the center
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

    // ========== Setup Crypto Context ==========
    std::cout << "\nSetting up crypto context..." << std::endl;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(8192);  // Larger ring for MNIST
    
    #if NATIVEINT == 128
    uint32_t scalingModSize = 78;
    uint32_t firstModSize = 89;
    #else
    uint32_t scalingModSize = 50;
    uint32_t firstModSize = 60;
    #endif
    
    parameters.SetScalingModSize(scalingModSize);
    parameters.SetFirstModSize(firstModSize);
    
    // For Chebyshev approximation of degree 63, we need sufficient depth
    // Each ReLU consumes ~6-7 levels, plus conv and dense layers
    uint32_t polyDegree = 58;
    uint32_t multDepth = 20;  // Increased depth for full network with approximations
    
    parameters.SetMultiplicativeDepth(multDepth);
    
    uint32_t batchSize = 2048;  // Enough for 720 elements from conv output
    parameters.SetBatchSize(batchSize);
    
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    std::cout << "CKKS scheme using ring dimension " << cc->GetRingDimension() << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl;
    std::cout << "Chebyshev polynomial degree: " << polyDegree << std::endl;

    // ========== Key Generation ==========
    std::cout << "\nGenerating keys..." << std::endl;
    TimeVar t;
    TIC(t);
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    std::cout << "Key generation time: " << TOC(t) << " ms" << std::endl;

    // ========== Define Network Weights (random for demo) ==========
    std::cout << "\nInitializing network weights..." << std::endl;

    // Conv layer: 5 output channels, 1 input channel, 5x5 kernel
    std::vector<std::vector<std::vector<std::vector<double>>>> convKernel(5);
    for (int oc = 0; oc < 5; oc++) {
        convKernel[oc].resize(1);  // 1 input channel
        convKernel[oc][0].resize(5, std::vector<double>(5, 0.0));
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

    // Dense layer 1: 720 -> 64
    uint32_t dense1Input = flattenedSize;
    uint32_t dense1Output = 64;
    std::vector<std::vector<double>> dense1Weights(dense1Output, std::vector<double>(dense1Input, 0.0));
    for (uint32_t i = 0; i < dense1Output; i++) {
        for (uint32_t j = 0; j < dense1Input; j++) {
            dense1Weights[i][j] = (rand() % 200 - 100) / 200.0;;  // Scaled weights
        }
    }
    std::cout << "  Dense1 shape: " << dense1Input << " -> " << dense1Output << std::endl;

    // Dense layer 2: 64 -> 10
    uint32_t dense2Input = dense1Output;
    uint32_t dense2Output = 10;
    std::vector<std::vector<double>> dense2Weights(dense2Output, std::vector<double>(dense2Input, 0.0));
    for (uint32_t i = 0; i < dense2Output; i++) {
        for (uint32_t j = 0; j < dense2Input; j++) {
            dense2Weights[i][j] = (rand() % 200 - 100) / 200.0;
        }
    }
    std::cout << "  Dense2 shape: " << dense2Input << " -> " << dense2Output << std::endl;

    // ========== Build Toeplitz matrices and pack into diagonals ==========
    std::cout << "\nPreparing encrypted network weights..." << std::endl;

    // Convolution layer
    TIC(t);
    auto toeplitzConv = ConstructConv2DToeplitz(convKernel, 28, 28, convStride, convPadding, 1, 1, 1, 1);
    std::size_t convRows = toeplitzConv.size();
    std::vector<std::vector<double>> convDiagonals = PackMatDiagWise(toeplitzConv, batchSize);
    std::vector<int32_t> convRotations = getOptimalRots(convDiagonals);
    std::cout << "  Conv Toeplitz: " << convRows << " rows, "
                    << convRotations.size() << " non-zero diagonals" << std::endl;
    
    // Dense layer 1
    std::vector<std::vector<double>> dense1Diagonals = PackMatDiagWise(dense1Weights, batchSize);
    std::vector<int32_t> dense1Rotations = getOptimalRots(dense1Diagonals);
    std::size_t dense1Rows = dense1Weights.size();
    std::cout << "  Dense1: " << dense1Rotations.size() << " non-zero diagonals" << std::endl;
    
    // Dense layer 2
    std::vector<std::vector<double>> dense2Diagonals = PackMatDiagWise(dense2Weights, batchSize);
    std::vector<int32_t> dense2Rotations = getOptimalRots(dense2Diagonals);
    std::size_t dense2Rows = dense2Weights.size();
    std::cout << "  Dense2: " << dense2Rotations.size() << " non-zero diagonals" << std::endl;

    // Collect all rotation indices (one key per index for faster inference)
    std::vector<int32_t> allRotations;
    allRotations.insert(allRotations.end(), convRotations.begin(), convRotations.end());
    allRotations.insert(allRotations.end(), dense1Rotations.begin(), dense1Rotations.end());
    allRotations.insert(allRotations.end(), dense2Rotations.begin(), dense2Rotations.end());
    allRotations.push_back(-convRows);
    allRotations.push_back(-dense1Rows);
    allRotations.push_back(-dense2Rows);

    // Remove duplicates
    std::sort(allRotations.begin(), allRotations.end());
    allRotations.erase(std::unique(allRotations.begin(), allRotations.end()), allRotations.end());

    std::cout << "  Total unique rotation keys needed: " << allRotations.size() << std::endl;
    std::cout << "  Generating rotation keys..." << std::endl;

    cc->EvalRotateKeyGen(keyPair.secretKey, allRotations);
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
    auto ctInput = cc->Encrypt(keyPair.publicKey, ptInput);
    std::cout << "Input encryption time: " << TOC(t) << " ms" << std::endl;
    std::cout << "Initial ciphertext level: " << ctInput->GetLevel() << std::endl;

    // ========== Forward Pass ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Starting encrypted inference..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Layer 1: Convolution
    std::cout << "\n[Layer 1] Convolution (28x28x1 -> 12x12x5)..." << std::endl;
    TIC(t);
    auto ctConvOut = EvalMultMatVecDiag(ctInput, ptConvDiags, 1, convRotations);
    double convTime = TOC(t);
    std::cout << "  Time: " << convTime << " ms" << std::endl;
    std::cout << "  Level: " << ctConvOut->GetLevel() << std::endl;
    PrintDebugValues(cc, ctConvOut, keyPair.secretKey, "Conv output", 10, flattenedSize);

    // Layer 2: ReLU (Chebyshev approximation)
    std::cout << "\n[Layer 2] ReLU (Chebyshev approximation, degree " << polyDegree << ")..." << std::endl;
    TIC(t);
    // NOTE: Run mnist-lola-cleartext first to get accurate bounds from Conv output
    // Update these bounds based on cleartext analysis for best approximation accuracy
    double relu1Lower = -1028.5;
    double relu1Upper = 937.9;
    auto ctReLU1 = EvalReLUChebyshev(cc, ctConvOut, polyDegree, relu1Lower, relu1Upper);
    double relu1Time = TOC(t);
    std::cout << "  Time: " << relu1Time << " ms" << std::endl;
    std::cout << "  Bounds: [" << relu1Lower << ", " << relu1Upper << "]" << std::endl;
    std::cout << "  Level: " << ctReLU1->GetLevel() << std::endl;
    PrintDebugValues(cc, ctReLU1, keyPair.secretKey, "ReLU1 output", 10, flattenedSize);

    // Layer 3: Dense 1 (720 -> 64)
    std::cout << "\n[Layer 3] Dense1 (720 -> 64)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU1, cc->EvalRotate(ctReLU1, -convRows));
    auto ctDense1Out = EvalDenseLayer(cc, ctReLU1, ptDense1Diags, dense1Rotations);
    double dense1Time = TOC(t);
    std::cout << "  Time: " << dense1Time << " ms" << std::endl;
    std::cout << "  Level: " << ctDense1Out->GetLevel() << std::endl;
    PrintDebugValues(cc, ctDense1Out, keyPair.secretKey, "Dense1 output", 10, dense1Output);

    // Layer 4: ReLU (Chebyshev approximation)
    std::cout << "\n[Layer 4] ReLU (Chebyshev approximation, degree " << polyDegree << ")..." << std::endl;
    TIC(t);
    // NOTE: Run mnist-lola-cleartext first to get accurate bounds from Dense1 output
    // Update these bounds based on cleartext analysis for best approximation accuracy
    double relu2Lower = -2376.5;
    double relu2Upper = 2330.2;
    auto ctReLU2 = EvalReLUChebyshev(cc, ctDense1Out, polyDegree, relu2Lower, relu2Upper);
    double relu2Time = TOC(t);
    std::cout << "  Time: " << relu2Time << " ms" << std::endl;
    std::cout << "  Bounds: [" << relu2Lower << ", " << relu2Upper << "]" << std::endl;
    std::cout << "  Level: " << ctReLU2->GetLevel() << std::endl;
    PrintDebugValues(cc, ctReLU2, keyPair.secretKey, "ReLU2 output", 10, dense1Output);

    // Layer 5: Dense 2 (64 -> 10)
    std::cout << "\n[Layer 5] Dense2 (64 -> 10)..." << std::endl;
    TIC(t);
    cc->EvalAddInPlace(ctReLU2, cc->EvalRotate(ctReLU2, -dense1Rows));
    auto ctOutput = EvalDenseLayer(cc, ctReLU2, ptDense2Diags, dense2Rotations);
    double dense2Time = TOC(t);
    std::cout << "  Time: " << dense2Time << " ms" << std::endl;
    std::cout << "  Level: " << ctOutput->GetLevel() << std::endl;
    PrintDebugValues(cc, ctOutput, keyPair.secretKey, "Final output", 10, dense2Output);

    double totalInferenceTime = convTime + relu1Time + dense1Time + relu2Time + dense2Time;
    std::cout << "\nTotal inference time: " << totalInferenceTime << " ms" << std::endl;

    // ========== Decrypt and Display Results ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Decrypting results..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    TIC(t);
    Plaintext ptOutput;
    cc->Decrypt(keyPair.secretKey, ctOutput, &ptOutput);
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
    std::cout << std::left << std::setw(30) << "ReLU 1 (Chebyshev)" << std::setw(15) << relu1Time << ctReLU1->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "Dense 1" << std::setw(15) << dense1Time << ctDense1Out->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "ReLU 2 (Chebyshev)" << std::setw(15) << relu2Time << ctReLU2->GetLevel() << std::endl;
    std::cout << std::left << std::setw(30) << "Dense 2" << std::setw(15) << dense2Time << ctOutput->GetLevel() << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Total Inference" << std::setw(15) << totalInferenceTime << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\n✓ MNIST LoLa Inference Complete (Chebyshev Approximation)!" << std::endl;
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
