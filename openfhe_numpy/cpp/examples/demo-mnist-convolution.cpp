#include "numpy_enc_matrix.h"
#include "openfhe.h"
#include "numpy_utils.h"
#include "numpy_helper_functions.h"
#include "conv_helper_function.h"

#include <iostream>
#include <vector>

using namespace openfhe_numpy;
using namespace lbcrypto;

CryptoContext<DCRTPoly> GenerateCryptoContext(uint32_t multDepth, uint32_t batchSize = 0) {
    uint32_t scaleModSize = 50;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    std::cout << "Ring dimension: " << cc->GetRingDimension() << std::endl;
    return cc;
}

void RunMNISTConvolution(const std::vector<std::vector<double>>& inputImage,
                         const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
                         uint32_t stride = 1, uint32_t padding = 0) {

    std::cout << "\n=== MNIST Convolution Demo ===\n" << std::endl;

    uint32_t input_height = inputImage.size();
    uint32_t input_width = inputImage[0].size();
    uint32_t batchSize = NextPow2(input_height*input_width);
    uint32_t multDepth = 10;

    // Setup
    TimeVar t;
    TIC(t);
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(multDepth);
    std::cout << "Setup: " << TOC(t) << " ms\n";

    // Keygen
    TIC(t);
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto toeplitzMatrix = ConstructConv2DToeplitz(kernel, input_height, input_width,
                                                   stride, padding, 1, 1, 1, 1);
    auto diagonals = PackMatDiagWise(toeplitzMatrix, batchSize);
    auto rotationIndices = getOptimalRots(diagonals);
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);
    std::cout << "Keygen: " << TOC(t) << " ms\n";

    // Encrypt
    TIC(t);
    auto flatVec = EncodeMatrix(inputImage, batchSize);
    auto ptVec = cc->MakeCKKSPackedPlaintext(flatVec);
    auto ctVec = cc->Encrypt(keyPair.publicKey, ptVec);
    std::cout << "Input Encryption: " << TOC(t) << " ms\n";
    auto ptDiags = MakeCKKSPackedPlaintextVectors(cc, diagonals);
    auto ctDiags = EncryptVectors(cc, keyPair.publicKey, ptDiags);
    std::cout << "Weight Encryption: " << TOC(t) << " ms\n";

    // Compute
    TIC(t);
    // auto ctResult = EvalMultMatVecDiag(ctVec, ptDiags, rotationIndices);
    auto ctResult = EvalMultMatVecDiag(ctVec, ctDiags, 1, rotationIndices);
    std::cout << "Computation: " << TOC(t) << " ms\n";

    // Decrypt
    TIC(t);
    Plaintext ptResult;
    cc->Decrypt(keyPair.secretKey, ctResult, &ptResult);
    ptResult->SetLength(batchSize);
    auto result = ptResult->GetRealPackedValue();
    std::cout << "Decryption: " << TOC(t) << " ms\n";

    std::cout << "\nEncrypted Result:\n";
    PrintVector(std::vector<double>(result.begin(), result.begin() + batchSize));

    std::cout << "\n=== Demo Complete ===\n";
}

int main() {
    // MNIST 28x28 sample image (normalized pixel values 0-1, scaled to 0-255 range)
    std::vector<std::vector<double>> mnistSample = {
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

    // Simple edge detection kernel (Sobel-like)
    std::vector<std::vector<std::vector<std::vector<double>>>> edgeKernel = {
        {{
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        }}
    };

    RunMNISTConvolution(mnistSample, edgeKernel, 1, 0);

    return 0;
}
