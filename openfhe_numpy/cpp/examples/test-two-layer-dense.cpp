#include "numpy_enc_matrix.h"
#include "openfhe.h"
#include "numpy_utils.h"
#include "numpy_helper_functions.h"
#include "conv_helper_function.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace openfhe_numpy;
using namespace lbcrypto;

constexpr bool DEBUG_MODE = true;

/**
 * @brief Cleartext Dense layer computation
 */
std::vector<double> CleartextDense(
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

/**
 * @brief Cleartext ReLU
 */
std::vector<double> CleartextReLU(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::max(0.0, input[i]);
    }
    return output;
}

/**
 * @brief Print first N values
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
 * @brief Decrypt and print first N values for debugging
 */
void PrintDebugValuesCT(
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

    std::cout << "  [DEBUG] " << name << " (first " << std::min(numValues, values.size()) << " values):";
    std::cout << std::endl << "    ";
    for (size_t i = 0; i < std::min(numValues, values.size()); i++) {
        std::cout << std::fixed << std::setprecision(4) << values[i];
        if (i < std::min(numValues, values.size()) - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

/**
 * @brief Compare two vectors and report differences
 */
void CompareVectors(const std::vector<double>& expected, const std::vector<double>& actual,
                    const std::string& name, double tolerance = 1e-3) {
    size_t len = std::min(expected.size(), actual.size());
    double maxError = 0.0;
    size_t errorCount = 0;

    for (size_t i = 0; i < len; i++) {
        double error = std::abs(expected[i] - actual[i]);
        if (error > tolerance) {
            errorCount++;
            if (error > maxError) {
                maxError = error;
            }
        }
    }

    std::cout << "  " << name << " comparison:" << std::endl;
    std::cout << "    Max error: " << std::fixed << std::setprecision(6) << maxError << std::endl;
    std::cout << "    Elements with error > " << tolerance << ": " << errorCount << " / " << len << std::endl;

    if (errorCount == 0) {
        std::cout << "    ✓ PASS" << std::endl;
    } else {
        std::cout << "    ✗ FAIL" << std::endl;
        // Print first few mismatches
        std::cout << "    First mismatches:" << std::endl;
        int printed = 0;
        for (size_t i = 0; i < len && printed < 5; i++) {
            double error = std::abs(expected[i] - actual[i]);
            if (error > tolerance) {
                std::cout << "      [" << i << "] expected: " << expected[i]
                         << ", actual: " << actual[i] << ", error: " << error << std::endl;
                printed++;
            }
        }
    }
}

void TestTwoLayerDense() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  TEST: Two-Layer Dense Network" << std::endl;
    std::cout << "  Architecture: Dense1 -> ReLU -> Dense2" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // ========== Set Fixed Random Seed ==========
    srand(42);
    std::cout << "Random seed: 42 (for reproducible weights)" << std::endl << std::endl;

    // ========== Network Parameters ==========
    uint32_t inputSize = 720;   // MNIST: 28x28 flattened and pooled
    uint32_t hidden1Size = 64;  // Hidden layer
    uint32_t outputSize = 10;   // 10 digit classes

    std::cout << "Network Architecture:" << std::endl;
    std::cout << "  Input: " << inputSize << " values" << std::endl;
    std::cout << "  Dense1: " << inputSize << " -> " << hidden1Size << std::endl;
    std::cout << "  ReLU" << std::endl;
    std::cout << "  Dense2: " << hidden1Size << " -> " << outputSize << std::endl << std::endl;

    // ========== Create Input ==========
    std::vector<double> input(inputSize);
    for (uint32_t i = 0; i < inputSize; i++) {
        input[i] = (rand() % 2000 - 1000) / 1000.0;  // Range [-1, 1], normalized
    }
    std::cout << "Created input vector (size " << inputSize << ")" << std::endl;
    PrintDebugValues(input, "Input", 10);  // Show first 10 values

    // ========== Create Weights ==========
    std::cout << "\nInitializing weights..." << std::endl;

    // Dense1 weights: hidden1Size x inputSize
    std::vector<std::vector<double>> dense1Weights(hidden1Size, std::vector<double>(inputSize, 0.0));
    for (uint32_t i = 0; i < hidden1Size; i++) {
        for (uint32_t j = 0; j < inputSize; j++) {
            dense1Weights[i][j] = (rand() % 200 - 100) / 200.0;  // Range [-0.5, 0.5]
        }
    }
    std::cout << "  Dense1 weights: " << hidden1Size << " x " << inputSize << std::endl;

    // Dense2 weights: outputSize x hidden1Size
    std::vector<std::vector<double>> dense2Weights(outputSize, std::vector<double>(hidden1Size, 0.0));
    for (uint32_t i = 0; i < outputSize; i++) {
        for (uint32_t j = 0; j < hidden1Size; j++) {
            dense2Weights[i][j] = (rand() % 200 - 100) / 200.0;  // Range [-0.5, 0.5]
        }
    }
    std::cout << "  Dense2 weights: " << outputSize << " x " << hidden1Size << std::endl;

    // ========== Cleartext Computation ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Cleartext Computation" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    auto dense1Out_clear = CleartextDense(input, dense1Weights);
    std::cout << "\n[Layer 1] Dense1 (" << inputSize << " -> " << hidden1Size << ")" << std::endl;
    PrintDebugValues(dense1Out_clear, "Dense1 output", 10);

    auto relu1Out_clear = CleartextReLU(dense1Out_clear);
    std::cout << "\n[Layer 2] ReLU" << std::endl;
    PrintDebugValues(relu1Out_clear, "ReLU output", 10);

    auto dense2Out_clear = CleartextDense(relu1Out_clear, dense2Weights);
    std::cout << "\n[Layer 3] Dense2 (" << hidden1Size << " -> " << outputSize << ")" << std::endl;
    PrintDebugValues(dense2Out_clear, "Dense2 output", outputSize);

    std::cout << "\nCleartext final output (all " << outputSize << " values):" << std::endl;
    for (uint32_t i = 0; i < outputSize; i++) {
        std::cout << "  [" << i << "] " << std::fixed << std::setprecision(6) << dense2Out_clear[i] << std::endl;
    }

    // ========== FHE Setup ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "FHE Setup" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    uint32_t multDepth = 10;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 2048;  // Large enough for MNIST (need at least 720)

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(8192);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    uint32_t actualBatchSize = cc->GetRingDimension() / 2;  // Get actual batch size
    std::cout << "CKKS scheme using ring dimension " << cc->GetRingDimension() << std::endl;
    std::cout << "Requested batch size: " << batchSize << std::endl;
    std::cout << "Actual batch size: " << actualBatchSize << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl;

    // if (actualBatchSize < inputSize) {
    //     std::cout << "\nWARNING: Batch size (" << actualBatchSize
    //               << ") is smaller than input size (" << inputSize << ")!" << std::endl;
    //     std::cout << "This will cause issues. Need larger ring dimension." << std::endl;
    // }

    // batchSize = actualBatchSize;  // Use actual batch size

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    // ========== Prepare FHE Weights ==========
    std::cout << "\nPreparing FHE weights..." << std::endl;

    // Pack Dense1 weights
    std::vector<std::vector<double>> dense1Diagonals = PackMatDiagWise(dense1Weights, batchSize);
    std::vector<int32_t> dense1Rotations = getOptimalRots(dense1Diagonals);
    std::size_t dense1Rows = dense1Weights.size();
    std::cout << "  Dense1: " << dense1Rows << " rows, " << dense1Rotations.size() << " non-zero diagonals" << std::endl;

    // Pack Dense2 weights
    std::vector<std::vector<double>> dense2Diagonals = PackMatDiagWise(dense2Weights, batchSize);
    std::vector<int32_t> dense2Rotations = getOptimalRots(dense2Diagonals);
    std::size_t dense2Rows = dense2Weights.size();
    std::cout << "  Dense2: " << dense2Rows << " rows, " << dense2Rotations.size() << " non-zero diagonals" << std::endl;

    // Collect all rotation indices
    std::vector<int32_t> allRotations;
    allRotations.insert(allRotations.end(), dense1Rotations.begin(), dense1Rotations.end());
    allRotations.insert(allRotations.end(), dense2Rotations.begin(), dense2Rotations.end());
    allRotations.push_back(-inputSize);
    allRotations.push_back(-dense1Rows);
    allRotations.push_back(-dense2Rows);

    // Remove duplicates
    std::sort(allRotations.begin(), allRotations.end());
    allRotations.erase(std::unique(allRotations.begin(), allRotations.end()), allRotations.end());

    std::cout << "  Total unique rotation keys needed: " << allRotations.size() << std::endl;
    cc->EvalRotateKeyGen(keyPair.secretKey, allRotations);

    // Encode weights as plaintexts
    auto ptDense1Diags = MakeCKKSPackedPlaintextVectors(cc, dense1Diagonals);
    auto ptDense2Diags = MakeCKKSPackedPlaintextVectors(cc, dense2Diagonals);

    // ========== Encrypt Input ==========
    std::cout << "\nEncrypting input..." << std::endl;
    std::vector<double> paddedInput = input;
    paddedInput.resize(batchSize, 0.0);
    auto ptInput = cc->MakeCKKSPackedPlaintext(paddedInput);
    auto ctInput = cc->Encrypt(keyPair.publicKey, ptInput);
    std::cout << "Initial ciphertext level: " << ctInput->GetLevel() << std::endl;

    // ========== FHE Computation ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "FHE Computation" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Layer 1: Dense1
    std::cout << "\n[Layer 1] Dense1 (" << inputSize << " -> " << hidden1Size << ")" << std::endl;
    cc->EvalAddInPlace(ctInput, cc->EvalRotate(ctInput, -inputSize));
    auto ctDense1Out = EvalMultMatVecDiag(ctInput, ptDense1Diags, dense1Rotations);
    std::cout << "  Level: " << ctDense1Out->GetLevel() << std::endl;
    PrintDebugValuesCT(cc, ctDense1Out, keyPair.secretKey, "Dense1 output", 10, hidden1Size);

    // Decrypt and compare
    Plaintext ptDense1Result;
    cc->Decrypt(keyPair.secretKey, ctDense1Out, &ptDense1Result);
    ptDense1Result->SetLength(hidden1Size);
    std::vector<double> dense1Out_fhe = ptDense1Result->GetRealPackedValue();
    CompareVectors(dense1Out_clear, dense1Out_fhe, "Dense1");

    // Layer 2: ReLU (using simple approximation: keep positive, zero negative)
    std::cout << "\n[Layer 2] ReLU (cleartext on decrypted values for testing)" << std::endl;
    auto relu1Out_fhe = CleartextReLU(dense1Out_fhe);
    PrintDebugValues(relu1Out_fhe, "ReLU output", 10);
    CompareVectors(relu1Out_clear, relu1Out_fhe, "ReLU");

    // Re-encrypt after ReLU for next layer
    std::vector<double> paddedRelu1 = relu1Out_fhe;
    paddedRelu1.resize(batchSize, 0.0);
    auto ptRelu1 = cc->MakeCKKSPackedPlaintext(paddedRelu1);
    auto ctRelu1 = cc->Encrypt(keyPair.publicKey, ptRelu1);

    // Layer 3: Dense2
    std::cout << "\n[Layer 3] Dense2 (" << hidden1Size << " -> " << outputSize << ")" << std::endl;

    // IMPORTANT: Add the folding operation before Dense2
    std::cout << "  Applying folding: EvalAdd(ct, EvalRotate(ct, -" << dense1Rows << "))" << std::endl;
    cc->EvalAddInPlace(ctRelu1, cc->EvalRotate(ctRelu1, -dense1Rows));

    auto ctDense2Out = EvalMultMatVecDiag(ctRelu1, ptDense2Diags, dense2Rotations);
    std::cout << "  Level: " << ctDense2Out->GetLevel() << std::endl;
    PrintDebugValuesCT(cc, ctDense2Out, keyPair.secretKey, "Dense2 output", outputSize, outputSize);  // Show all values

    // Decrypt and compare
    Plaintext ptDense2Result;
    cc->Decrypt(keyPair.secretKey, ctDense2Out, &ptDense2Result);
    ptDense2Result->SetLength(outputSize);
    std::vector<double> dense2Out_fhe = ptDense2Result->GetRealPackedValue();

    std::cout << "\nFHE final output (all " << outputSize << " values):" << std::endl;
    for (uint32_t i = 0; i < outputSize; i++) {
        std::cout << "  [" << i << "] " << std::fixed << std::setprecision(6) << dense2Out_fhe[i] << std::endl;
    }

    CompareVectors(dense2Out_clear, dense2Out_fhe, "Dense2");

    // ========== Summary ==========
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Tested two-layer dense network with fixed random seed" << std::endl;
    std::cout << "Architecture: " << inputSize << " -> " << hidden1Size << " -> " << outputSize << std::endl;
    std::cout << "✓ Test Complete!" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        TestTwoLayerDense();
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
