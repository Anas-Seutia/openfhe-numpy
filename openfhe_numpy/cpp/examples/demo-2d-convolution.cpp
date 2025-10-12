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
    // Step 1: Setup CryptoContext

    // A. Specify main parameters

    /* A2) Bit-length of scaling factor.
    * CKKS works for real numbers, but these numbers are encoded as integers.
    * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
    * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
    * integer). Say the result of a computation based on m' is 130, then at
    * decryption, the scaling factor is removed so the user is presented with
    * the real number result of 0.13.
    */
    uint32_t scaleModSize = 50;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    return cc;
}

/**
 * @brief Demonstrate matrix-vector multiplication using CRC encoding
 * 
 * @param inputMatrix 2D cleartext matrix
 * @param inputVector input cleartext vector
 */
void MatrixVectorProduct_CRC(std::vector<std::vector<double>> inputMatrix, std::vector<double> inputVector) {
    std::cout << "=== DEMO: Conv. pt2 (Matrix-Vector Product) with CRC Encoding ===" << std::endl;

    uint multDepth = 10 ;

    printf("\nMatrix: \n");
    PrintMatrix(inputMatrix);

    printf("\nVector: \n");
    PrintVector(inputVector);

    std::cout << "Initializing CryptoContext...\n";
    TimeVar t_setup;
    TIC(t_setup);
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(multDepth);
    double time_setup          = TOC(t_setup);
    std::cout << "Setup time: " << time_setup << " ms" << std::endl;

    // Generate keys
    std::cout << "Generating keys...\n";
    TimeVar t_keygen;
    TIC(t_keygen);
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    auto sumColsKey    = cc->EvalSumColsKeyGen(keyPair.secretKey);
    double time_keygen = TOC(t_keygen);
    std::cout << "Key generation time: " << time_keygen << " ms" << std::endl;

    // Encode and encrypt mat and vector

    std::size_t nRows          = inputMatrix.size();
    std::size_t nCols          = !inputMatrix.empty() ? inputMatrix[0].size() : 0;
    std::size_t paddedRowCount = NextPow2(nCols);

    std::size_t batchSize         = cc->GetRingDimension() / 2;

    std::vector<double> flatMat = EncodeMatrix(inputMatrix, batchSize);
    std::vector<double> flatVec = PackVecColWise(inputVector, nCols, batchSize);


    print_range(flatMat, 0 , 32);
    print_range(flatVec, 0 , 32);

    std::cout << "Encrypting inputs...\n";
    TimeVar t_encrypt;
    TIC(t_encrypt);
    auto ptMat          = cc->MakeCKKSPackedPlaintext(flatMat);
    auto ptVec          = cc->MakeCKKSPackedPlaintext(flatVec);
    auto ctMat          = cc->Encrypt(keyPair.publicKey, ptMat);
    auto ctVec          = cc->Encrypt(keyPair.publicKey, ptVec);
    double time_encrypt = TOC(t_encrypt);
    std::cout << "Encryption time: " << time_encrypt << " ms" << std::endl;

    std::cout << "\n--- Plaintext Matrix-Vector Product ---\n";
    PrintVector(MulMatVec(inputMatrix, inputVector));

    // Perform encrypted mat-vector multiplication
    std::cout << "\nPerforming homomorphic matrix-vector multiplication...";
    TimeVar t_mult;
    TIC(t_mult);
    auto ctProd      = EvalMultMatVec(sumColsKey, MatVecEncoding::MM_CRC, paddedRowCount, ctVec, ctMat);
    double time_mult = TOC(t_mult);
    std::cout << "\nHomomorphic multiplication time: " << time_mult << " ms" << std::endl;

    // Decrypt result
    std::cout << "Decrypting result...\n";
    TimeVar t_decrypt;
    TIC(t_decrypt);
    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, ctProd, &ptResult);
    ptResult->SetLength(nCols * nRows);
    std::vector<double> resultVector = ptResult->GetRealPackedValue();
    double time_decrypt              = TOC(t_decrypt);

    std::cout << "--- Homomorphic Computation Result ---\n";
    PrintVector(resultVector);
    std::cout << "Decryption time: " << time_decrypt << " ms" << std::endl;
    std::cout << "Matrix-Vector Demo Complete.\n";
}

std::vector<std::vector<double>> ToeplitzConv_Packing(
    const std::vector<std::vector<std::vector<std::vector<double>>>>& inputKernel,
    const uint32_t &input_height,
    const uint32_t &input_width,
    const uint32_t &stride,
    const uint32_t &padding,
    const uint32_t &dilation,
    const uint32_t &batch_size,
    const uint32_t &input_gap,
    const uint32_t &output_gap
) {
    std::cout << "=== DEMO: Conv. pt1 with Toeplitz Convolution Packing ===" << std::endl;

    const std::vector<std::vector<double>> toeplitz = ConstructConv2DToeplitz(inputKernel, input_height, input_width, stride, padding, dilation, batch_size, input_gap, output_gap);
    PrintMatrix(toeplitz);
    return toeplitz;
}

std::map<uint32_t, std::vector<double>> DiagonalConv_Packing(
    const std::vector<std::vector<double>> matrix,
    const std::size_t &num_slots
) {
    std::cout << "=== DEMO: Conv. pt2 with Diagonal Packing ===" << std::endl;

    const std::map<uint32_t, std::vector<double>> diagonalized = PackMatDiagWise(matrix, num_slots);
    for (const auto& diag_entry : diagonalized) {
        uint32_t diag_idx = diag_entry.first;
        const auto& values = diag_entry.second;
        
        std::cout << "  Diagonal " << diag_idx << " (first 10 values): [";
        for (size_t i = 0; i < std::min(size_t(64), values.size()); ++i) {
            std::cout << std::fixed << std::setprecision(2) << values[i];
            if (i < std::min(size_t(64), values.size()) - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    return diagonalized;
}

/**
 * @brief Main function
 */
int main(int argc, char* argv[]) {
    int choice = 0;

    // Correct 4D kernel: (out_channels=1, in_channels=1, height=3, width=3)
    std::vector<std::vector<std::vector<std::vector<double>>>> inputkernel = {
        {  // Output channel 0
            {  // Input channel 0
                {7, 0, 1},
                {3, 5, 0},
                {1, 8, 0}
            }
        }
    };

    std::vector<double> inputMatrix =  {0,  7,  8,  10, 1,  2,  7,  6,
                                        0,  1,  1,  9,  7,  5,  1,  7,
                                        8,  8,  4,  5,  8,  2,  6,  1,
                                        1,  0,  0,  1,  10, 3,  1,  7,
                                        7,  8,  2,  5,  3,  2,  10, 9,
                                        0,  3,  4,  10, 10, 5,  2,  5,
                                        2,  5,  0,  2,  8,  8,  5,  9,
                                        5,  1,  10, 6,  2,  8,  6,  3};

    uint32_t input_height = 8;
    uint32_t input_width = 8;
    uint32_t stride = 1;
    uint32_t padding = 0;
    uint32_t dilation = 1;
    uint32_t batch_size = 1;
    uint32_t input_gap = 1;
    uint32_t output_gap = 1;

    /** result: [[85. 107. 149. 203.  84.  90.]
     *          [ 66.  60.  59. 204. 118.  89.]
     *          [134.  85.  83. 119. 126. 111.]
     *          [ 92.  70. 125. 130. 140. 105.]
     *          [108.  95.  95. 183. 158.  96.]
     *          [ 48. 127. 106. 143. 202. 145.]]
     */

    if (argc > 1) {
        choice = atoi(argv[1]);
    }
    else {
        std::cout << "OpenFHE Matrix Operations Demo\n"
                  << "-------------------------------\n"
                  << "1. Toeplitz Packing Only\n"
                  << "2. Toeplitz + Diagonalize Packing\n"
                  << "3. Full Convolution\n"
                  << "Enter choice (default=1): ";
        std::cin >> choice;
    }

    switch (choice) {
        case 2:
            DiagonalConv_Packing(ToeplitzConv_Packing(inputkernel, input_height, input_width, stride, padding, dilation, batch_size, input_gap, output_gap), 4096);
            break;
        case 3:
            MatrixVectorProduct_CRC(ToeplitzConv_Packing(inputkernel, input_height, input_width, stride, padding, dilation, batch_size, input_gap, output_gap), inputMatrix);
            break;
        case 1:
        default:
            ToeplitzConv_Packing(inputkernel, input_height, input_width, stride, padding, dilation, batch_size, input_gap, output_gap);
            break;
    }

    return 0;
}