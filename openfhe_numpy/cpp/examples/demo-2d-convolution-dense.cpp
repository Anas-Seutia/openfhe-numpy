#include "numpy_enc_matrix.h"
#include "openfhe.h"
#include "numpy_utils.h"
#include "numpy_helper_functions.h"
#include "conv_helper_function.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace openfhe_numpy;
using namespace lbcrypto;

/**
 * @brief Generate a CKKS crypto context with specified parameters
 *
 * @param multDepth Multiplicative depth
 * @param batchSize Optional batch size (default: 0)
 * @return CryptoContext<DCRTPoly> Configured crypto context
 */
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
 * @brief Naive cleartext 2D convolution
 *
 * @param input Input image as 2D matrix (height x width)
 * @param kernel 4D kernel (out_channels, in_channels, kernel_height, kernel_width)
 * @param stride Stride for convolution
 * @param padding Padding for convolution
 * @param dilation Dilation for convolution
 * @return 2D output matrix
 */
std::vector<std::vector<double>> NaiveConv2D(
    const std::vector<std::vector<double>>& input,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
    uint32_t stride = 1,
    uint32_t padding = 0,
    uint32_t dilation = 1
) {
    uint32_t input_height = input.size();
    uint32_t input_width = input[0].size();
    uint32_t out_channels = kernel.size();
    uint32_t in_channels = kernel[0].size();
    uint32_t kernel_height = kernel[0][0].size();
    uint32_t kernel_width = kernel[0][0][0].size();

    // Compute output dimensions
    uint32_t output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    uint32_t output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    // Initialize output
    std::vector<std::vector<double>> output(output_height, std::vector<double>(output_width, 0.0));

    // Perform convolution
    for (uint32_t oh = 0; oh < output_height; ++oh) {
        for (uint32_t ow = 0; ow < output_width; ++ow) {
            double sum = 0.0;

            // Sum over all output and input channels
            for (uint32_t oc = 0; oc < out_channels; ++oc) {
                for (uint32_t ic = 0; ic < in_channels; ++ic) {
                    for (uint32_t kh = 0; kh < kernel_height; ++kh) {
                        for (uint32_t kw = 0; kw < kernel_width; ++kw) {
                            // Calculate input position
                            int32_t ih = oh * stride - padding + kh * dilation;
                            int32_t iw = ow * stride - padding + kw * dilation;

                            // Check bounds (handle padding)
                            if (ih >= 0 && ih < (int32_t)input_height &&
                                iw >= 0 && iw < (int32_t)input_width) {
                                sum += input[ih][iw] * kernel[oc][ic][kh][kw];
                            }
                        }
                    }
                }
            }

            output[oh][ow] = sum;
        }
    }

    return output;
}

/**
 * @brief Construct a dense (non-multiplexed) convolution matrix
 *
 * @param kernel 4D kernel (out_channels, in_channels, kernel_height, kernel_width)
 * @param input_height Input height
 * @param input_width Input width
 * @param stride Stride
 * @param padding Padding
 * @param dilation Dilation
 * @return Dense convolution matrix
 */
std::vector<std::vector<double>> ConstructDenseConvMatrix(
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
    const uint32_t &input_height,
    const uint32_t &input_width,
    const uint32_t &stride,
    const uint32_t &padding,
    const uint32_t &dilation
) {
    // Use ConstructConv2DToeplitz with gap=1 (no multiplexing) to get base dense matrix
    return ConstructConv2DToeplitz(kernel, input_height, input_width, stride, padding, dilation, 1, 1);
}

void MatrixVectorProduct_Diag(std::vector<std::vector<double>> inputMatrix, std::vector<double> inputVector) {
    std::cout << "=== DEMO: Conv. (Matrix-Vector Product) with Diagonal Encoding ===\n" << std::endl;

    uint multDepth = 10;

    printf("Matrix dimensions: %zu x %zu\n", inputMatrix.size(), inputMatrix[0].size());
    printf("Vector size: %zu\n\n", inputVector.size());

    std::cout << "Initializing CryptoContext...\n";
    TimeVar t_setup;
    TIC(t_setup);
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(multDepth);
    double time_setup = TOC(t_setup);
    std::cout << "Setup time: " << time_setup << " ms" << std::endl;

    std::size_t nRows = inputMatrix.size();
    std::size_t nCols = !inputMatrix.empty() ? inputMatrix[0].size() : 0;
    std::size_t batchSize = cc->GetRingDimension() / 2;

    // Generate keys
    std::cout << "Generating keys...\n";
    TimeVar t_keygen;
    TIC(t_keygen);
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    // Pack matrix into diagonals
    std::cout << "Packing matrix into diagonals...\n";
    std::vector<std::vector<double>> diagonals = PackMatDiagWise(inputMatrix, batchSize);
    std::vector<double> flatVec = PackVecColWise(inputVector, nCols, batchSize);

    // Generate rotation keys
    std::vector<int32_t> rotationIndices = getOptimalRots(diagonals);
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);

    double time_keygen = TOC(t_keygen);
    std::cout << "Key generation time: " << time_keygen << " ms" << std::endl;

    std::cout << "Encrypting input vector and diagonals...\n";
    TimeVar t_encrypt;
    TIC(t_encrypt);
    auto ptVec = cc->MakeCKKSPackedPlaintext(flatVec);
    auto ptDiags = MakeCKKSPackedPlaintextVectors(cc, diagonals);
    auto ctVec = cc->Encrypt(keyPair.publicKey, ptVec);
    auto ctDiags = EncryptVectors(cc, keyPair.publicKey, ptDiags);
    double time_encrypt = TOC(t_encrypt);
    std::cout << "Encryption time: " << time_encrypt << " ms" << std::endl;

    std::cout << "\n--- Plaintext Matrix-Vector Product ---\n";
    PrintVector(MulMatVec(inputMatrix, inputVector));

    // Perform encrypted mat-vector multiplication
    std::cout << "\nPerforming homomorphic matrix-vector multiplication...\n";
    TimeVar t_mult;
    TIC(t_mult);
    Ciphertext<DCRTPoly> ctResult = EvalMultMatVecDiag(ctVec, ctDiags, 1, rotationIndices);
    double time_mult = TOC(t_mult);
    std::cout << "Homomorphic multiplication time: " << time_mult << " ms" << std::endl;

    // Decrypt result
    std::cout << "Decrypting result...\n";
    TimeVar t_decrypt;
    TIC(t_decrypt);
    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, ctResult, &ptResult);
    ptResult->SetLength(nRows);
    std::vector<double> resultVector = ptResult->GetRealPackedValue();
    double time_decrypt = TOC(t_decrypt);

    std::cout << "--- Homomorphic Computation Result ---\n";
    PrintVector(resultVector);
    std::cout << "Decryption time: " << time_decrypt << " ms" << std::endl;
    std::cout << "Matrix-Vector Demo Complete.\n";
}

/**
 * @brief Main function
 */
int main(int argc, char* argv[]) {
    // ========================================================================
    // CONFIGURATION PARAMETERS - Change these to test different shapes
    // ========================================================================

    // Kernel shape: (out_channels, in_channels, kernel_height, kernel_width)
    uint32_t out_channels = 1;
    uint32_t in_channels = 5;
    uint32_t kernel_height = 5;
    uint32_t kernel_width = 5;

    std::cout << "Generating kernel with shape: ("
              << out_channels << ", " << in_channels << ", "
              << kernel_height << ", " << kernel_width << ")" << std::endl;

    // Auto-generate 4D kernel filled with 1s
    std::vector<std::vector<std::vector<std::vector<double>>>> inputKernel(
        out_channels,
        std::vector<std::vector<std::vector<double>>>(
            in_channels,
            std::vector<std::vector<double>>(
                kernel_height,
                std::vector<double>(kernel_width, 1.0)
            )
        )
    );

    // Input test matrix (manually set)
    std::vector<std::vector<double>> input2DMatrix = {
        {0,  7,  8,  10, 1,  2,  7,  6},
        {0,  1,  1,  9,  7,  5,  1,  7},
        {8,  8,  4,  5,  8,  2,  6,  1},
        {1,  0,  0,  1,  10, 3,  1,  7},
        {7,  8,  2,  5,  3,  2,  10, 9},
        {0,  3,  4,  10, 10, 5,  2,  5},
        {2,  5,  0,  2,  8,  8,  5,  9},
        {5,  1,  10, 6,  2,  8,  6,  3}
    };

    // Convolution parameters
    uint32_t input_height = 28;   // Expected input height
    uint32_t input_width = 28;    // Expected input width
    uint32_t stride = 2;
    uint32_t padding = 0;
    uint32_t dilation = 1;

    // Multiplexing parameters (used in case 1 and elsewhere)
    uint32_t output_gap = 1;

    // ========================================================================
    // Demo selection
    // ========================================================================
    int choice = 0;

    if (argc > 1) {
        choice = atoi(argv[1]);
    }
    else {
        std::cout << "\nDense Convolution Matrix Demo\n"
                  << "-----------------------------\n"
                  << "1. Multiplexed Matrix → Unmultiplex (MultiplexDenseMatrix)\n"
                  << "2. Dense Matrix → Diagonalize\n"
                  << "3. Full Convolution Test\n"
                  << "Enter choice (default=1): ";
        std::cin >> choice;
    }

    switch (choice) {
        case 1: {
            std::cout << "\n=== Case 1: Multiplexed Matrix + Unmultiplex (MultiplexDenseMatrix) ===\n" << std::endl;

            // Build multiplexed convolution matrix with input_gap > 1
            uint32_t test_input_gap = 2;  // Use gap=2 for multiplexing
            std::cout << "Building multiplexed convolution matrix with input_gap=" << test_input_gap << "...\n";
            auto multiplexedMatrix = ConstructConv2DToeplitz(inputKernel, input_height, input_width,
                                                            stride, padding, dilation,
                                                            test_input_gap, output_gap);
            std::cout << "Multiplexed matrix dimensions: " << multiplexedMatrix.size() << " x "
                      << multiplexedMatrix[0].size() << "\n" << std::endl;

            std::cout << "Multiplexed Matrix (first 10x10):\n";
            for (size_t i = 0; i < std::min(size_t(10), multiplexedMatrix.size()); ++i) {
                for (size_t j = 0; j < std::min(size_t(10), multiplexedMatrix[i].size()); ++j) {
                    std::cout << std::fixed << std::setprecision(1) << multiplexedMatrix[i][j] << " ";
                }
                std::cout << std::endl;
            }

            // Apply MultiplexDenseMatrix to unmultiplex (reorder columns)
            std::cout << "\nApplying MultiplexDenseMatrix to unmultiplex input (input_gap="
                      << test_input_gap << ")...\n";
            std::cout << "This reorders columns from multiplexed to standard layout.\n";
            auto unmultiplexedMatrix = MultiplexDenseMatrix(multiplexedMatrix,
                                                           input_height, input_width, test_input_gap);
            std::cout << "Unmultiplexed matrix dimensions: " << unmultiplexedMatrix.size() << " x "
                      << unmultiplexedMatrix[0].size() << "\n" << std::endl;

            std::cout << "Unmultiplexed Matrix (first 10x10):\n";
            for (size_t i = 0; i < std::min(size_t(10), unmultiplexedMatrix.size()); ++i) {
                for (size_t j = 0; j < std::min(size_t(10), unmultiplexedMatrix[i].size()); ++j) {
                    std::cout << std::fixed << std::setprecision(1) << unmultiplexedMatrix[i][j] << " ";
                }
                std::cout << std::endl;
            }

            // Compare with non-multiplexed matrix
            std::cout << "\nFor comparison, non-multiplexed matrix (gap=1):\n";
            auto standardMatrix = ConstructDenseConvMatrix(inputKernel, input_height, input_width,
                                                          stride, padding, dilation);
            std::cout << "Standard matrix dimensions: " << standardMatrix.size() << " x "
                      << standardMatrix[0].size() << "\n" << std::endl;
            break;
        }
        case 2: {
            std::cout << "\n=== Case 2: Dense Matrix + Diagonalize ===\n" << std::endl;

            // Build dense matrix
            auto denseMatrix = ConstructDenseConvMatrix(inputKernel, input_height, input_width, stride, padding, dilation);
            std::cout << "Dense matrix dimensions: " << denseMatrix.size() << " x " << denseMatrix[0].size() << "\n" << std::endl;

            // Diagonalize
            std::size_t num_slots = 64 * 64;
            std::cout << "Diagonalizing with " << num_slots << " slots...\n";
            auto diagonals = PackMatDiagWise(denseMatrix, num_slots);
            auto rotations = getOptimalRots(diagonals);

            std::cout << "Number of diagonals: " << diagonals.size() << std::endl;
            std::cout << "Number of non-zero diagonals: " << rotations.size() - 1 << "\n" << std::endl;

            int count = 0;
            for (const int32_t diag_idx : rotations) {
                if (diag_idx < 0) continue;
                std::cout << "  Diagonal " << count << " (rotation " << diag_idx << ", first 10 values): [";
                for (size_t i = 0; i < std::min(size_t(10), diagonals[diag_idx].size()); ++i) {
                    std::cout << std::fixed << std::setprecision(1) << diagonals[diag_idx][i];
                    if (i < std::min(size_t(10), diagonals[diag_idx].size()) - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                count++;
            }
            break;
        }
        case 3: {
            std::cout << "\n=== Case 3: Full Convolution Test ===\n" << std::endl;

            // Build dense matrix
            auto denseMatrix = ConstructDenseConvMatrix(inputKernel, input_height, input_width, stride, padding, dilation);

            // Encode input matrix as vector
            auto inputVector = EncodeMatrix(input2DMatrix, 64);

            // Run homomorphic computation
            MatrixVectorProduct_Diag(denseMatrix, inputVector);

            // Show cleartext result
            std::cout << std::endl << "--- Cleartext Convolution Result ---" << std::endl;
            PrintMatrix(NaiveConv2D(input2DMatrix, inputKernel, stride, padding, dilation));
            break;
        }
        default:
            std::cout << "Invalid choice. Please select 1, 2, or 3." << std::endl;
            break;
    }

    return 0;
}
