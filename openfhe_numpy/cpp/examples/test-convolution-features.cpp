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

using namespace openfhe_numpy;
using namespace lbcrypto;

CryptoContext<DCRTPoly> GenerateCryptoContext(uint32_t multDepth, uint32_t batchSize = 0) {
    uint32_t scaleModSize = 59;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    return cc;
}

/**
 * @brief Compare FHE result with expected cleartext result
 *
 * @param fheResult Flat vector from FHE computation
 * @param expected 3D expected result (out_channels, height, width)
 * @param tolerance Maximum allowed difference
 * @return true if results match within tolerance
 */
bool CompareResults(const std::vector<double>& fheResult,
                    const std::vector<std::vector<std::vector<double>>>& expected,
                    double tolerance = 1e-6) {
    // Flatten expected result
    std::vector<double> expectedFlat;
    for (const auto& channel : expected) {
        for (const auto& row : channel) {
            for (double val : row) {
                expectedFlat.push_back(val);
            }
        }
    }

    if (fheResult.size() != expectedFlat.size()) {
        std::cout << "❌ Size mismatch: FHE=" << fheResult.size()
                  << ", Expected=" << expectedFlat.size() << std::endl;
        return false;
    }

    double maxError = 0.0;
    size_t errorCount = 0;

    for (size_t i = 0; i < fheResult.size(); ++i) {
        double error = std::abs(fheResult[i] - expectedFlat[i]);
        if (error > tolerance) {
            if (errorCount < 5) {  // Print first 5 errors
                std::cout << "❌ Mismatch at index " << i
                          << ": FHE=" << fheResult[i]
                          << ", Expected=" << expectedFlat[i]
                          << ", Error=" << error << std::endl;
            }
            errorCount++;
        }
        maxError = std::max(maxError, error);
    }

    if (errorCount > 0) {
        std::cout << "❌ Total errors: " << errorCount << "/" << fheResult.size()
                  << ", Max error: " << maxError << std::endl;
        return false;
    }

    std::cout << "✅ Results match! Max error: " << maxError << std::endl;
    return true;
}

/**
 * @brief General cleartext 2D convolution
 *
 * @param input 3D input tensor (in_channels, height, width)
 * @param kernel 4D kernel tensor (out_channels, in_channels, kernel_height, kernel_width)
 * @param stride Convolution stride
 * @param padding Zero padding size
 * @param dilation Kernel dilation
 * @return 3D output tensor (out_channels, output_height, output_width)
 */
std::vector<std::vector<std::vector<double>>> NaiveConv2D(
    const std::vector<std::vector<std::vector<double>>>& input,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
    uint32_t stride = 1,
    uint32_t padding = 0,
    uint32_t dilation = 1
) {
    uint32_t in_channels = input.size();
    uint32_t input_height = input[0].size();
    uint32_t input_width = input[0][0].size();
    uint32_t out_channels = kernel.size();
    uint32_t kernel_height = kernel[0][0].size();
    uint32_t kernel_width = kernel[0][0][0].size();
    
    uint32_t output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    uint32_t output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
    
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
                            int32_t ih = oh * stride - padding + kh * dilation;
                            int32_t iw = ow * stride - padding + kw * dilation;
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

void Test5x5Kernel() {
    std::cout << "\n=== TEST 2: 5x5 Kernel (1 output channel) ===" << std::endl;

    std::vector<std::vector<std::vector<std::vector<double>>>> kernel = {
        {{ {1, 0, -1, 0, 1}, {2, 0, -2, 0, 2}, {3, 0, -3, 0, 3}, {2, 0, -2, 0, 2}, {1, 0, -1, 0, 1} }}
    };

    std::vector<std::vector<std::vector<double>>> input = {
        {  // Channel 0
            {1, 2, 3, 4, 5, 6, 7, 8},
            {9, 10, 11, 12, 13, 14, 15, 16},
            {17, 18, 19, 20, 21, 22, 23, 24},
            {25, 26, 27, 28, 29, 30, 31, 32},
            {33, 34, 35, 36, 37, 38, 39, 40},
            {41, 42, 43, 44, 45, 46, 47, 48},
            {49, 50, 51, 52, 53, 54, 55, 56},
            {57, 58, 59, 60, 61, 62, 63, 64}
        }
    };

    uint32_t input_height = input[0].size();
    uint32_t input_width = input[0][0].size();
    uint32_t stride = 1, padding = 0, dilation = 1;
    uint32_t multDepth = 10;

    std::cout << "\nInput (1 channel, " << input_height << "x" << input_width << "):" << std::endl;
    PrintMatrix(input[0]);

    auto expected = NaiveConv2D(input, kernel, stride, padding, dilation);
    std::cout << "\n--- Expected Cleartext Result ---" << std::endl;
    PrintMatrix(expected[0]);

    // FHE Computation
    std::cout << "\n--- FHE Computation ---\n";
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(multDepth);
    std::size_t batchSize = cc->GetRingDimension() / 2;

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto toeplitzMatrix = ConstructConv2DToeplitz(kernel, input_height, input_width, stride, padding, dilation, 1, 1, 1);
    std::size_t nRows = toeplitzMatrix.size();
    // std::size_t nCols = !toeplitzMatrix.empty() ? toeplitzMatrix[0].size() : 0;

    std::vector<std::vector<double>> diagonals = PackMatDiagWise(toeplitzMatrix, batchSize);
    std::vector<double> flatVec = EncodeMatrix(input[0], batchSize);
    std::vector<int32_t> rotationIndices = getOptimalRots(diagonals);
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);

    auto ptVec = cc->MakeCKKSPackedPlaintext(flatVec);
    auto ptDiags = MakeCKKSPackedPlaintextVectors(cc, diagonals);
    auto ctVec = cc->Encrypt(keyPair.publicKey, ptVec);
    auto ctDiags = EncryptVectors(cc, keyPair.publicKey, ptDiags);

    auto ctResult = EvalMultMatVecDiag(ctVec, ctDiags, 1, rotationIndices);

    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, ctResult, &ptResult);
    ptResult->SetLength(nRows);
    std::vector<double> result = ptResult->GetRealPackedValue();

    std::cout << "\n--- FHE Encrypted Result ---" << std::endl;
    PrintVector(result);

    std::cout << "\n--- Verification ---" << std::endl;
    bool passed = CompareResults(result, expected);
    std::cout << (passed ? "\n✓ Test Passed!" : "\n✗ Test Failed!") << std::endl;
}

void TestMultipleOutputChannels() {
    std::cout << "\n=== TEST 1: Multiple Output Channels (2 channels, 3x3 kernel) ===" << std::endl;

    std::vector<std::vector<std::vector<std::vector<double>>>> kernel = {
        {
            {{1, 0, -1},
             {2, 0, -2},
             {1, 0, -1}}
        },
        {
            {{1, 2, 1},
             {0, 0, 0},
             {-1,-2,-1}}
        },
        {
            {{0, 1, 0},
             {1, 4, 1},
             {0, 1, 0}}
        }
    };

    std::vector<std::vector<std::vector<double>>> input = {
        {  // Channel 0
            {1, 2, 3, 4, 5, 6},
            {7, 8, 9, 10, 11, 12},
            {13, 14, 15, 16, 17, 18},
            {19, 20, 21, 22, 23, 24},
            {25, 26, 27, 28, 29, 30},
            {31, 32, 33, 34, 35, 36}
        }
    };

    uint32_t input_height = input[0].size();
    uint32_t input_width = input[0][0].size();
    uint32_t stride = 1, padding = 0, dilation = 1;
    uint32_t multDepth = 10;

    std::cout << "\nInput (1 channel, " << input_height << "x" << input_width << "):" << std::endl;
    PrintMatrix(input[0]);

    auto expected = NaiveConv2D(input, kernel, stride, padding, dilation);
    std::cout << "\n--- Expected Cleartext Result ---" << std::endl;
    for (size_t oc = 0; oc < expected.size(); ++oc) {
        std::cout << "Output Channel " << oc << ":" << std::endl;
        PrintMatrix(expected[oc]);
    }

    // FHE Computation - Following demo-2d-convolution.cpp pattern
    std::cout << "\n--- FHE Computation ---\n";
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(multDepth);
    std::size_t batchSize = cc->GetRingDimension() / 2;

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey); 

    auto toeplitzMatrix = ConstructConv2DToeplitz(kernel, input_height, input_width, stride, padding, dilation, 1, 1, 1);
    std::size_t nRows = toeplitzMatrix.size();
    // std::size_t nCols = !toeplitzMatrix.empty() ? toeplitzMatrix[0].size() : 0;

    std::vector<std::vector<double>> diagonals = PackMatDiagWise(toeplitzMatrix, batchSize);
    std::vector<double> flatVec = EncodeMatrix(input[0], batchSize);
    std::vector<int32_t> rotationIndices = getOptimalRots(diagonals);
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);

    auto ptVec = cc->MakeCKKSPackedPlaintext(flatVec);
    auto ptDiags = MakeCKKSPackedPlaintextVectors(cc, diagonals);
    auto ctVec = cc->Encrypt(keyPair.publicKey, ptVec);
    auto ctDiags = EncryptVectors(cc, keyPair.publicKey, ptDiags);

    auto ctResult = EvalMultMatVecDiag(ctVec, ctDiags, 1, rotationIndices);

    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, ctResult, &ptResult);
    ptResult->SetLength(nRows);
    std::vector<double> result = ptResult->GetRealPackedValue();

    std::cout << "\n--- FHE Encrypted Result ---" << std::endl;
    PrintVector(result);

    std::cout << "\n--- Verification ---" << std::endl;
    bool passed = CompareResults(result, expected);
    std::cout << (passed ? "\n✓ Test Passed!" : "\n✗ Test Failed!") << std::endl;
}


void TestMultipleInputChannels() {
    std::cout << "\n=== TEST 3: Multiple Input Channels (3 input channels, 1 output channel, 3x3 kernel) ===" << std::endl;
    
    // Kernel: (out_channels=1, in_channels=3, height=3, width=3)
    std::vector<std::vector<std::vector<std::vector<double>>>> kernel = {
        {  // Output channel 0
            {  // Input channel 0
                {1, 0, -1},
                {2, 0, -2},
                {1, 0, -1}
            },
            {  // Input channel 1
                {1, 2, 1},
                {0, 0, 0},
                {-1, -2, -1}
            },
            {  // Input channel 2
                {0, 1, 0},
                {1, -4, 1},
                {0, 1, 0}
            }
        }
    };

    // Input: 3 channels of 6x6 images
    std::vector<std::vector<std::vector<double>>> input = {
        {  // Channel 0
            {1, 2, 3, 4, 5, 6},
            {7, 8, 9, 10, 11, 12},
            {13, 14, 15, 16, 17, 18},
            {19, 20, 21, 22, 23, 24},
            {25, 26, 27, 28, 29, 30},
            {31, 32, 33, 34, 35, 36}
        },
        {  // Channel 1
            {2, 3, 4, 5, 6, 7},
            {8, 9, 10, 11, 12, 13},
            {14, 15, 16, 17, 18, 19},
            {20, 21, 22, 23, 24, 25},
            {26, 27, 28, 29, 30, 31},
            {32, 33, 34, 35, 36, 37}
        },
        {  // Channel 2
            {3, 4, 5, 6, 7, 8},
            {9, 10, 11, 12, 13, 14},
            {15, 16, 17, 18, 19, 20},
            {21, 22, 23, 24, 25, 26},
            {27, 28, 29, 30, 31, 32},
            {33, 34, 35, 36, 37, 38}
        }
    };

    uint32_t input_height = input[0].size();
    uint32_t input_width = input[0][0].size();
    uint32_t stride = 1, padding = 0, dilation = 1;
    uint32_t multDepth = 10;

    std::cout << "\nInput (3 channels, " << input_height << "x" << input_width << "):" << std::endl;
    for (size_t c = 0; c < input.size(); ++c) {
        std::cout << "Channel " << c << ":" << std::endl;
        PrintMatrix(input[c]);
    }

    auto expected = NaiveConv2D(input, kernel, stride, padding, dilation);
    std::cout << "\n--- Expected Cleartext Result ---" << std::endl;
    for (size_t oc = 0; oc < expected.size(); ++oc) {
        std::cout << "Output Channel " << oc << ":" << std::endl;
        PrintMatrix(expected[oc]);
    }

    // Flatten input for FHE processing (channels stacked vertically)
    std::vector<std::vector<double>> input_flattened;
    for (const auto& channel : input) {
        for (const auto& row : channel) {
            input_flattened.push_back(row);
        }
    }

    // FHE Computation
    std::cout << "\n--- FHE Computation ---\n";
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(multDepth);
    std::size_t batchSize = cc->GetRingDimension() / 2;

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto toeplitzMatrix = ConstructConv2DToeplitz(kernel, input_height, input_width, stride, padding, dilation, 1, 1, 1);
    std::size_t nRows = toeplitzMatrix.size();
    std::size_t nCols = !toeplitzMatrix.empty() ? toeplitzMatrix[0].size() : 0;

    std::cout << "Toeplitz matrix size: " << nRows << " × " << nCols << std::endl;

    std::vector<std::vector<double>> diagonals = PackMatDiagWise(toeplitzMatrix, batchSize);
    std::vector<double> flatVec = EncodeMatrix(input_flattened, batchSize);
    std::vector<int32_t> rotationIndices = getOptimalRots(diagonals);
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);

    auto ptVec = cc->MakeCKKSPackedPlaintext(flatVec);
    auto ptDiags = MakeCKKSPackedPlaintextVectors(cc, diagonals);
    auto ctVec = cc->Encrypt(keyPair.publicKey, ptVec);
    auto ctDiags = EncryptVectors(cc, keyPair.publicKey, ptDiags);

    auto ctResult = EvalMultMatVecDiag(ctVec, ctDiags, 1, rotationIndices);

    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, ctResult, &ptResult);
    ptResult->SetLength(nRows);
    std::vector<double> result = ptResult->GetRealPackedValue();

    std::cout << "\n--- FHE Encrypted Result (should be 4x4 output) ---" << std::endl;
    PrintVector(result);

    std::cout << "\n--- Verification ---" << std::endl;
    bool passed = CompareResults(result, expected);
    std::cout << (passed ? "\n✓ Test Passed!" : "\n✗ Test Failed!") << std::endl;
}

void TestBothFeatures() {
    std::cout << "\n=== TEST 4: Multiple Input & Output Channels (2 in, 2 out, 3x3 kernel) ===" << std::endl;

    // Kernel: (out_channels=2, in_channels=2, height=3, width=3)
    std::vector<std::vector<std::vector<std::vector<double>>>> kernel = {
        {  // Output channel 0
            {  // Input channel 0
                {1, 0, -1},
                {2, 0, -2},
                {1, 0, -1}
            },
            {  // Input channel 1
                {1, 2, 1},
                {0, 0, 0},
                {-1, -2, -1}
            }
        },
        {  // Output channel 1
            {  // Input channel 0
                {0, 1, 0},
                {1, -4, 1},
                {0, 1, 0}
            },
            {  // Input channel 1
                {-1, 0, 1},
                {-2, 0, 2},
                {-1, 0, 1}
            }
        }
    };

    // Input: 2 channels of 5x5 images
    std::vector<std::vector<std::vector<double>>> input = {
        {  // Channel 0
            {1, 2, 3, 4, 5},
            {6, 7, 8, 9, 10},
            {11, 12, 13, 14, 15},
            {16, 17, 18, 19, 20},
            {21, 22, 23, 24, 25}
        },
        {  // Channel 1
            {2, 3, 4, 5, 6},
            {7, 8, 9, 10, 11},
            {12, 13, 14, 15, 16},
            {17, 18, 19, 20, 21},
            {22, 23, 24, 25, 26}
        }
    };

    uint32_t input_height = input[0].size();
    uint32_t input_width = input[0][0].size();
    uint32_t stride = 1, padding = 0, dilation = 1;
    uint32_t multDepth = 10;

    std::cout << "\nInput (2 channels, " << input_height << "x" << input_width << "):" << std::endl;
    for (size_t c = 0; c < input.size(); ++c) {
        std::cout << "Channel " << c << ":" << std::endl;
        PrintMatrix(input[c]);
    }

    auto expected = NaiveConv2D(input, kernel, stride, padding, dilation);
    std::cout << "\n--- Expected Cleartext Result ---" << std::endl;
    for (size_t oc = 0; oc < expected.size(); ++oc) {
        std::cout << "Output Channel " << oc << ":" << std::endl;
        PrintMatrix(expected[oc]);
    }

    // Flatten input for FHE processing (channels stacked vertically)
    std::vector<std::vector<double>> input_flattened;
    for (const auto& channel : input) {
        for (const auto& row : channel) {
            input_flattened.push_back(row);
        }
    }

    // FHE Computation
    std::cout << "\n--- FHE Computation ---\n";
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(multDepth);
    std::size_t batchSize = cc->GetRingDimension() / 2;

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto toeplitzMatrix = ConstructConv2DToeplitz(kernel, input_height, input_width, stride, padding, dilation, 1, 1, 1);
    std::size_t nRows = toeplitzMatrix.size();
    std::size_t nCols = !toeplitzMatrix.empty() ? toeplitzMatrix[0].size() : 0;

    std::cout << "Toeplitz matrix size: " << nRows << " × " << nCols << std::endl;

    std::vector<std::vector<double>> diagonals = PackMatDiagWise(toeplitzMatrix, batchSize);
    std::vector<double> flatVec = EncodeMatrix(input_flattened, batchSize);
    std::vector<int32_t> rotationIndices = getOptimalRots(diagonals);
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);

    auto ptVec = cc->MakeCKKSPackedPlaintext(flatVec);
    auto ptDiags = MakeCKKSPackedPlaintextVectors(cc, diagonals);
    auto ctVec = cc->Encrypt(keyPair.publicKey, ptVec);
    auto ctDiags = EncryptVectors(cc, keyPair.publicKey, ptDiags);

    auto ctResult = EvalMultMatVecDiag(ctVec, ctDiags, 1, rotationIndices);

    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, ctResult, &ptResult);
    ptResult->SetLength(nRows);
    std::vector<double> result = ptResult->GetRealPackedValue();

    std::cout << "\n--- FHE Encrypted Result ---" << std::endl;
    PrintVector(result);

    std::cout << "\n--- Verification ---" << std::endl;
    bool passed = CompareResults(result, expected);
    std::cout << (passed ? "\n✓ Test Passed!" : "\n✗ Test Failed!") << std::endl;
}

int main(int argc, char* argv[]) {
    int choice = 0;

    if (argc > 1) {
        choice = atoi(argv[1]);
    } else {
        std::cout << "OpenFHE Convolution Feature Tests\n"
                  << "---------------------------------\n"
                  << "1. 5x5 Kernel (1 output channel, 1 input channel)\n"
                  << "2. Multiple Output Channels (3 output channels, 1 input channel, 3x3 kernel)\n"
                  << "3. Multiple Input Channels (1 output channel, 3 input channels, 3x3 kernel)\n"
                  << "4. Multiple Input & Output Channels (2 output channels, 2 input channels, 3x3 kernel)\n"
                  << "5. Run All Tests\n"
                  << "Enter choice (default=5): ";
        std::cin >> choice;
    }

    switch (choice) {
        case 1:
            Test5x5Kernel();
            break;
        case 2:
            TestMultipleOutputChannels();
            break;
        case 3:
            TestMultipleInputChannels();
            break;
        case 4:
            TestBothFeatures();
            break;
        case 5:
        default:
            Test5x5Kernel();
            TestMultipleOutputChannels();
            TestMultipleInputChannels();
            TestBothFeatures();
            break;
    }

    std::cout << "\n=== All Tests Complete ===" << std::endl;
    return 0;
}
