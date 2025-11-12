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
    double sumError = 0.0;

    for (size_t i = 0; i < len; i++) {
        double error = std::abs(expected[i] - actual[i]);
        sumError += error;
        if (error > tolerance) {
            errorCount++;
            if (error > maxError) {
                maxError = error;
            }
        }
    }

    double avgError = sumError / len;
    std::cout << "  [VALIDATION] " << name << ":" << std::endl;
    std::cout << "    Max error: " << std::fixed << std::setprecision(6) << maxError << std::endl;
    std::cout << "    Avg error: " << avgError << std::endl;
    std::cout << "    Elements with error > " << tolerance << ": " << errorCount << " / " << len;

    if (errorCount == 0) {
        std::cout << " ✓ PASS" << std::endl;
    } else {
        std::cout << " ✗ FAIL" << std::endl;
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

void TestTwoLayerAvgPool() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  TEST: Two-Layer Average Pooling" << std::endl;
    std::cout << "  Architecture: AvgPool1 -> AvgPool2" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    // ========== Set Fixed Random Seed ==========
    srand(42);
    std::cout << "Random seed: 42 (for reproducible input)" << std::endl << std::endl;

    // ========== Network Parameters ==========
    // Start with 12x12 input with 4 channels
    // Pool1: 2x2, stride=2 -> 6x6x4
    // Pool2: 2x2, stride=2 -> 3x3x4
    uint32_t input_channels = 4;
    uint32_t input_height = 12;
    uint32_t input_width = 12;
    uint32_t pool_kernel = 2;
    uint32_t pool_stride = 2;

    uint32_t pool1_out_height = (input_height - pool_kernel) / pool_stride + 1;  // 6
    uint32_t pool1_out_width = (input_width - pool_kernel) / pool_stride + 1;    // 6
    uint32_t pool1_flat_size = input_channels * pool1_out_height * pool1_out_width;  // 144

    uint32_t pool2_out_height = (pool1_out_height - pool_kernel) / pool_stride + 1;  // 3
    uint32_t pool2_out_width = (pool1_out_width - pool_kernel) / pool_stride + 1;    // 3
    uint32_t pool2_flat_size = input_channels * pool2_out_height * pool2_out_width;  // 36

    std::cout << "Network Architecture:" << std::endl;
    std::cout << "  Input: " << input_channels << " channels, " << input_height << "x" << input_width << std::endl;
    std::cout << "  AvgPool1: 2x2 kernel, stride=2 -> " << pool1_out_height << "x" << pool1_out_width << "x" << input_channels << " = " << pool1_flat_size << std::endl;
    std::cout << "  AvgPool2: 2x2 kernel, stride=2 -> " << pool2_out_height << "x" << pool2_out_width << "x" << input_channels << " = " << pool2_flat_size << std::endl << std::endl;

    // ========== Create Input ==========
    std::vector<std::vector<std::vector<double>>> input3D(
        input_channels,
        std::vector<std::vector<double>>(input_height, std::vector<double>(input_width))
    );

    for (uint32_t c = 0; c < input_channels; c++) {
        for (uint32_t h = 0; h < input_height; h++) {
            for (uint32_t w = 0; w < input_width; w++) {
                input3D[c][h][w] = (rand() % 2000 - 1000) / 100.0;  // Range [-10, 10]
            }
        }
    }
    std::cout << "Created input (" << input_channels << "x" << input_height << "x" << input_width << ")" << std::endl;
    std::vector<double> inputFlat = CleartextFlatten(input3D);
    PrintDebugValues(inputFlat, "Input", 10);

    // ========== Create Average Pooling Kernels ==========
    std::cout << "\nInitializing average pooling kernels..." << std::endl;

    // AvgPool1: 4->4 channels, 2x2 kernel, stride=2
    std::vector<std::vector<std::vector<std::vector<double>>>> avgpool1Kernel(input_channels);
    for (uint32_t oc = 0; oc < input_channels; oc++) {
        avgpool1Kernel[oc].resize(input_channels);
        for (uint32_t ic = 0; ic < input_channels; ic++) {
            avgpool1Kernel[oc][ic].resize(pool_kernel, std::vector<double>(pool_kernel, 0.0));
            if (oc == ic) {  // Identity mapping for each channel
                for (uint32_t i = 0; i < pool_kernel; i++) {
                    for (uint32_t j = 0; j < pool_kernel; j++) {
                        avgpool1Kernel[oc][ic][i][j] = 0.25;  // Average pooling
                    }
                }
            }
        }
    }
    std::cout << "  AvgPool1 kernel: " << input_channels << " channels, " << pool_kernel << "x" << pool_kernel << std::endl;

    // AvgPool2: 4->4 channels, 2x2 kernel, stride=2
    std::vector<std::vector<std::vector<std::vector<double>>>> avgpool2Kernel(input_channels);
    for (uint32_t oc = 0; oc < input_channels; oc++) {
        avgpool2Kernel[oc].resize(input_channels);
        for (uint32_t ic = 0; ic < input_channels; ic++) {
            avgpool2Kernel[oc][ic].resize(pool_kernel, std::vector<double>(pool_kernel, 0.0));
            if (oc == ic) {  // Identity mapping for each channel
                for (uint32_t i = 0; i < pool_kernel; i++) {
                    for (uint32_t j = 0; j < pool_kernel; j++) {
                        avgpool2Kernel[oc][ic][i][j] = 0.25;  // Average pooling
                    }
                }
            }
        }
    }
    std::cout << "  AvgPool2 kernel: " << input_channels << " channels, " << pool_kernel << "x" << pool_kernel << std::endl;

    // ========== Cleartext Computation ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Cleartext Computation" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    auto pool1_3D_clear = CleartextAvgPool2D(input3D, pool_kernel, pool_stride);
    auto pool1_clear = CleartextFlatten(pool1_3D_clear);
    std::cout << "\n[Layer 1] AvgPool1 (" << input_height << "x" << input_width << "x" << input_channels
              << " -> " << pool1_out_height << "x" << pool1_out_width << "x" << input_channels << ")" << std::endl;
    PrintDebugValues(pool1_clear, "AvgPool1 output", 10);

    auto pool2_3D_clear = CleartextAvgPool2D(pool1_3D_clear, pool_kernel, pool_stride);
    auto pool2_clear = CleartextFlatten(pool2_3D_clear);
    std::cout << "\n[Layer 2] AvgPool2 (" << pool1_out_height << "x" << pool1_out_width << "x" << input_channels
              << " -> " << pool2_out_height << "x" << pool2_out_width << "x" << input_channels << ")" << std::endl;
    PrintDebugValues(pool2_clear, "AvgPool2 output", pool2_flat_size);

    std::cout << "\nCleartext final output (all " << pool2_flat_size << " values):" << std::endl;
    for (uint32_t i = 0; i < pool2_flat_size; i++) {
        std::cout << "  [" << i << "] " << std::fixed << std::setprecision(6) << pool2_clear[i] << std::endl;
    }

    // ========== FHE Setup ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "FHE Setup" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    uint32_t multDepth = 10;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 2048;

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

    std::cout << "CKKS scheme using ring dimension " << cc->GetRingDimension() << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl;

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    // ========== Prepare FHE Weights ==========
    std::cout << "\nPreparing FHE weights..." << std::endl;

    // Build Toeplitz matrices
    auto toeplitzPool1 = ConstructConv2DToeplitz(avgpool1Kernel, input_height, input_width, pool_stride, 0, 1, 1, 1, 1);
    std::vector<std::vector<double>> pool1Diagonals = PackMatDiagWise(toeplitzPool1, batchSize);
    std::size_t pool1Cols = pool1Diagonals.size();
    std::vector<int32_t> pool1Rotations = getOptimalRots(pool1Diagonals, true);
    std::cout << "  AvgPool1 Toeplitz: " << pool1Cols << " rows, " << pool1Rotations.size() << " rotation keys needed" << std::endl;

    auto toeplitzPool2 = ConstructConv2DToeplitz(avgpool2Kernel, pool1_out_height, pool1_out_width, pool_stride, 0, 1, 1, 1, 1);
    std::vector<std::vector<double>> pool2Diagonals = PackMatDiagWise(toeplitzPool2, batchSize);
    std::size_t pool2Cols = pool2Diagonals.size();
    std::vector<int32_t> pool2Rotations = getOptimalRots(pool2Diagonals, true);
    std::cout << "  AvgPool2 Toeplitz: " << pool2Cols << " rows, " << pool2Rotations.size() << " rotation keys needed" << std::endl;

    // Collect all rotation indices
    std::vector<int32_t> allRotations;
    allRotations.insert(allRotations.end(), pool1Rotations.begin(), pool1Rotations.end());
    allRotations.insert(allRotations.end(), pool2Rotations.begin(), pool2Rotations.end());
    allRotations.push_back(-static_cast<int32_t>(pool1Cols));
    allRotations.push_back(-static_cast<int32_t>(pool2Cols));

    // Remove duplicates
    std::sort(allRotations.begin(), allRotations.end());
    allRotations.erase(std::unique(allRotations.begin(), allRotations.end()), allRotations.end());

    std::cout << "  Total unique rotation keys needed: " << allRotations.size() << std::endl;
    cc->EvalRotateKeyGen(keyPair.secretKey, allRotations);

    // Encode weights as plaintexts
    auto ptPool1Diags = MakeCKKSPackedPlaintextVectors(cc, pool1Diagonals);
    auto ptPool2Diags = MakeCKKSPackedPlaintextVectors(cc, pool2Diagonals);

    // ========== Encrypt Input ==========
    std::cout << "\nEncrypting input..." << std::endl;
    std::vector<double> paddedInput = inputFlat;
    paddedInput.resize(batchSize, 0.0);
    auto ptInput = cc->MakeCKKSPackedPlaintext(paddedInput);
    auto ctInput = cc->Encrypt(keyPair.publicKey, ptInput);
    std::cout << "Initial ciphertext level: " << ctInput->GetLevel() << std::endl;

    // ========== FHE Computation ==========
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "FHE Computation" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Layer 1: AvgPool1
    std::cout << "\n[Layer 1] AvgPool1 (" << input_height << "x" << input_width << "x" << input_channels
              << " -> " << pool1_out_height << "x" << pool1_out_width << "x" << input_channels << ")" << std::endl;
    std::cout << "  Applying folding: EvalAdd(ct, EvalRotate(ct, -" << pool1Cols << "))" << std::endl;
    cc->EvalAddInPlace(ctInput, cc->EvalRotate(ctInput, -static_cast<int32_t>(pool1Cols)));
    auto ctPool1 = EvalMultMatVecDiag(ctInput, ptPool1Diags, 2, pool1Rotations);
    std::cout << "  Level: " << ctPool1->GetLevel() << std::endl;
    PrintDebugValuesCT(cc, ctPool1, keyPair.secretKey, "AvgPool1 output", 10, pool1_flat_size);

    // Decrypt and compare
    Plaintext ptPool1Result;
    cc->Decrypt(keyPair.secretKey, ctPool1, &ptPool1Result);
    ptPool1Result->SetLength(pool1_flat_size);
    std::vector<double> pool1_fhe = ptPool1Result->GetRealPackedValue();
    CompareVectors(pool1_clear, pool1_fhe, "AvgPool1", 1e-2);

    // Layer 2: AvgPool2
    std::cout << "\n[Layer 2] AvgPool2 (" << pool1_out_height << "x" << pool1_out_width << "x" << input_channels
              << " -> " << pool2_out_height << "x" << pool2_out_width << "x" << input_channels << ")" << std::endl;
    std::cout << "  Applying folding: EvalAdd(ct, EvalRotate(ct, -" << pool2Cols << "))" << std::endl;
    cc->EvalAddInPlace(ctPool1, cc->EvalRotate(ctPool1, -static_cast<int32_t>(pool2Cols)));
    auto ctPool2 = EvalMultMatVecDiag(ctPool1, ptPool2Diags, 2, pool2Rotations);
    std::cout << "  Level: " << ctPool2->GetLevel() << std::endl;
    PrintDebugValuesCT(cc, ctPool2, keyPair.secretKey, "AvgPool2 output", pool2_flat_size, pool2_flat_size);

    // Decrypt and compare
    Plaintext ptPool2Result;
    cc->Decrypt(keyPair.secretKey, ctPool2, &ptPool2Result);
    ptPool2Result->SetLength(pool2_flat_size);
    std::vector<double> pool2_fhe = ptPool2Result->GetRealPackedValue();

    std::cout << "\nFHE final output (all " << pool2_flat_size << " values):" << std::endl;
    for (uint32_t i = 0; i < pool2_flat_size; i++) {
        std::cout << "  [" << i << "] " << std::fixed << std::setprecision(6) << pool2_fhe[i] << std::endl;
    }

    CompareVectors(pool2_clear, pool2_fhe, "AvgPool2", 1e-2);

    // ========== Summary ==========
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Tested two consecutive average pooling layers with fixed random seed" << std::endl;
    std::cout << "Architecture: " << input_height << "x" << input_width << "x" << input_channels
              << " -> " << pool1_out_height << "x" << pool1_out_width << "x" << input_channels
              << " -> " << pool2_out_height << "x" << pool2_out_width << "x" << input_channels << std::endl;
    std::cout << "✓ Test Complete!" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        TestTwoLayerAvgPool();
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
