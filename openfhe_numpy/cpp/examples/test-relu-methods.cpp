#include "numpy_enc_matrix.h"
#include "openfhe.h"
#include "binfhecontext.h"
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

/**
 * @brief Test ReLU using Chebyshev approximation in CKKS
 *
 * This method approximates ReLU using polynomial approximation.
 * Pros: Stays within CKKS, faster for simple operations
 * Cons: Approximation error, especially near x=0
 */
void TestReLUApproximation1() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  TEST 1: ReLU via Chebyshev Approximation (CKKS)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Setup CryptoContext
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(8192);

#if NATIVEINT == 128
    uint32_t scalingModSize = 78;
    uint32_t firstModSize = 89;
#else
    uint32_t scalingModSize = 50;
    uint32_t firstModSize = 60;
#endif

    parameters.SetScalingModSize(scalingModSize);
    parameters.SetFirstModSize(firstModSize);

    // Higher degree = better approximation but more computation
    uint32_t polyDegree = 127;
    uint32_t multDepth = 17;  // Based on polynomial degree

    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetBatchSize(16);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    std::cout << "CKKS scheme using ring dimension " << cc->GetRingDimension() << std::endl;
    std::cout << "Polynomial degree for approximation: " << polyDegree << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl << std::endl;

    // Key generation
    TimeVar t;
    TIC(t);
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    std::cout << "Key generation: " << TOC(t) << " ms" << std::endl;

    // Test inputs: range from negative to positive
    std::vector<double> input = {-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    size_t encodedLength = input.size();

    std::cout << "Input values: ";
    for (const auto& val : input) {
        std::cout << std::fixed << std::setprecision(2) << val << " ";
    }
    std::cout << std::endl;

    // Expected ReLU output
    std::vector<double> expected(encodedLength);
    for (size_t i = 0; i < encodedLength; ++i) {
        expected[i] = std::max(0.0, input[i]);
    }
    std::cout << "Expected ReLU: ";
    for (const auto& val : expected) {
        std::cout << std::fixed << std::setprecision(2) << val << " ";
    }
    std::cout << std::endl << std::endl;

    // Encode and encrypt
    TIC(t);
    Plaintext plaintext = cc->MakeCKKSPackedPlaintext(input);
    auto ciphertext = cc->Encrypt(keyPair.publicKey, plaintext);
    std::cout << "Encryption: " << TOC(t) << " ms" << std::endl;

    // Evaluate ReLU using Chebyshev approximation
    double lowerBound = -10.0;
    double upperBound = 10.0;

    TIC(t);
    auto reluResult = cc->EvalChebyshevFunction(
        [](double x) -> double { return std::max(0.0, x); },
        ciphertext,
        lowerBound,
        upperBound,
        polyDegree
    );
    std::cout << "ReLU computation: " << TOC(t) << " ms" << std::endl;

    // Decrypt and verify
    TIC(t);
    Plaintext plaintextDec;
    cc->Decrypt(keyPair.secretKey, reluResult, &plaintextDec);
    plaintextDec->SetLength(encodedLength);
    std::cout << "Decryption: " << TOC(t) << " ms" << std::endl;

    std::vector<std::complex<double>> result = plaintextDec->GetCKKSPackedValue();
    std::cout << "\nActual output: ";
    for (size_t i = 0; i < encodedLength; ++i) {
        std::cout << std::fixed << std::setprecision(4) << result[i].real() << " ";
    }
    std::cout << std::endl;

    // Compute error
    std::cout << "\nError analysis:" << std::endl;
    double maxError = 0.0;
    double avgError = 0.0;
    for (size_t i = 0; i < encodedLength; ++i) {
        double error = std::abs(expected[i] - result[i].real());
        avgError += error;
        maxError = std::max(maxError, error);
        std::cout << "  Input " << std::setw(6) << input[i]
                  << " | Expected " << std::setw(6) << expected[i]
                  << " | Got " << std::setw(8) << result[i].real()
                  << " | Error " << std::setw(8) << error << std::endl;
    }
    avgError /= encodedLength;
    std::cout << "\nMax error: " << maxError << std::endl;
    std::cout << "Avg error: " << avgError << std::endl;
    std::cout << "\n✓ ReLU Approximation Test Complete!\n" << std::endl;
}

/**
 * @brief Test ReLU using Minimax approximation in CKKS
 *
 * This method approximates ReLU using polynomial approximation.
 * Pros: Stays within CKKS, faster for simple operations
 * Cons: Approximation error, especially near x=0
 */
void TestReLUApproximation2() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  TEST 2: ReLU via Minimax Approximation (CKKS)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Setup CryptoContext
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(8192);

#if NATIVEINT == 128
    uint32_t scalingModSize = 78;
    uint32_t firstModSize = 89;
#else
    uint32_t scalingModSize = 50;
    uint32_t firstModSize = 60;
#endif

    parameters.SetScalingModSize(scalingModSize);
    parameters.SetFirstModSize(firstModSize);

    // Higher degree = better approximation but more computation
    uint32_t polyDegree1 = 15;
    uint32_t polyDegree2 = 15;
    uint32_t polyDegree3 = 27;
    uint32_t multDepth = 18;  // Based on polynomial degree

    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetBatchSize(16);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    std::cout << "CKKS scheme using ring dimension " << cc->GetRingDimension() << std::endl;
    std::cout << "Polynomial degree for approximation: " 
    << polyDegree1 << ", " << polyDegree2 << ", " << polyDegree3 << ", " << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl << std::endl;

    // Key generation
    TimeVar t;
    TIC(t);
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    std::cout << "Key generation: " << TOC(t) << " ms" << std::endl;

    // Test inputs: range from negative to positive
    std::vector<double> input = {-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    size_t encodedLength = input.size();

    std::cout << "Input values: ";
    for (const auto& val : input) {
        std::cout << std::fixed << std::setprecision(2) << val << " ";
    }
    std::cout << std::endl;

    // Expected ReLU output
    std::vector<double> expected(encodedLength);
    for (size_t i = 0; i < encodedLength; ++i) {
        expected[i] = std::max(0.0, input[i]);
    }
    std::cout << "Expected ReLU: ";
    for (const auto& val : expected) {
        std::cout << std::fixed << std::setprecision(2) << val << " ";
    }
    std::cout << std::endl << std::endl;

    // Encode and encrypt
    TIC(t);
    Plaintext plaintext = cc->MakeCKKSPackedPlaintext(input);
    auto ciphertext = cc->Encrypt(keyPair.publicKey, plaintext);
    std::cout << "Encryption: " << TOC(t) << " ms" << std::endl;

    // Evaluate ReLU using Minimax approximation
    double lowerBound = -10.0;
    double upperBound = 10.0;

    TIC(t);
    auto reluResult1 = cc->EvalChebyshevFunction(
        [](double x) -> double { return std::max(0.0, x); },
        ciphertext,
        lowerBound,
        upperBound,
        polyDegree1
    );
    std::cout << "ReLU computation: " << TOC(t) << " ms" << std::endl;
    auto reluResult2 = cc->EvalChebyshevFunction(
        [](double x) -> double { return std::max(0.0, x); },
        reluResult1,
        -1.0,
        upperBound,
        polyDegree2
    );
    std::cout << "ReLU computation: " << TOC(t) << " ms" << std::endl;
    auto reluResult3 = cc->EvalChebyshevFunction(
        [](double x) -> double { return std::max(0.0, x); },
        reluResult2,
        -0.1,
        upperBound,
        polyDegree3
    );
    std::cout << "ReLU computation: " << TOC(t) << " ms" << std::endl;

    // Decrypt and verify
    TIC(t);
    Plaintext plaintextDec;
    cc->Decrypt(keyPair.secretKey, reluResult3, &plaintextDec);
    plaintextDec->SetLength(encodedLength);
    std::cout << "Decryption: " << TOC(t) << " ms" << std::endl;

    std::vector<std::complex<double>> result = plaintextDec->GetCKKSPackedValue();
    std::cout << "\nActual output: ";
    for (size_t i = 0; i < encodedLength; ++i) {
        std::cout << std::fixed << std::setprecision(4) << result[i].real() << " ";
    }
    std::cout << std::endl;

    // Compute error
    std::cout << "\nError analysis:" << std::endl;
    double maxError = 0.0;
    double avgError = 0.0;
    for (size_t i = 0; i < encodedLength; ++i) {
        double error = std::abs(expected[i] - result[i].real());
        avgError += error;
        maxError = std::max(maxError, error);
        std::cout << "  Input " << std::setw(6) << input[i]
                  << " | Expected " << std::setw(6) << expected[i]
                  << " | Got " << std::setw(8) << result[i].real()
                  << " | Error " << std::setw(8) << error << std::endl;
    }
    avgError /= encodedLength;
    std::cout << "\nMax error: " << maxError << std::endl;
    std::cout << "Avg error: " << avgError << std::endl;
    std::cout << "\n✓ ReLU Approximation Test Complete!\n" << std::endl;
}

/**
 * @brief Test ReLU using scheme switching between CKKS and FHEW
 *
 * This method uses comparison in FHEW to get exact ReLU behavior.
 * Pros: Exact computation (no approximation error)
 * Cons: Slower due to scheme switching overhead
 */
void TestReLUSchemeSwitching() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  TEST 2: ReLU via Scheme Switching" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Setup CryptoContext for CKKS
    ScalingTechnique scTech = FLEXIBLEAUTO;
    uint32_t multDepth = 17;
    if (scTech == FLEXIBLEAUTOEXT)
        multDepth += 1;

    uint32_t scaleModSize = 50;
    uint32_t firstModSize = 60;
    uint32_t ringDim = 8192;
    SecurityLevel sl = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE = 25;
    uint32_t slots = 16;
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
    std::cout << "Multiplicative depth: " << multDepth << std::endl << std::endl;

    // Generate encryption keys
    TimeVar t;
    TIC(t);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // Prepare FHEW cryptocontext for scheme switching
    SchSwchParams params;
    params.SetSecurityLevelCKKS(sl);
    params.SetSecurityLevelFHEW(slBin);
    params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE);
    params.SetNumSlotsCKKS(slots);
    params.SetNumValues(slots);

    auto privateKeyFHEW = cc->EvalSchemeSwitchingSetup(params);
    auto ccLWE = cc->GetBinCCForSchemeSwitch();

    ccLWE->BTKeyGen(privateKeyFHEW);
    cc->EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW);

    std::cout << "Key generation: " << TOC(t) << " ms" << std::endl;

    std::cout << "FHEW lattice parameter: " << ccLWE->GetParams()->GetLWEParams()->Getn() << std::endl;
    std::cout << "FHEW logQ: " << logQ_ccLWE << std::endl;
    std::cout << "FHEW modulus q: " << ccLWE->GetParams()->GetLWEParams()->Getq() << std::endl << std::endl;

    // Set scaling for comparison
    auto modulus_LWE = 1 << logQ_ccLWE;
    auto beta = ccLWE->GetBeta().ConvertToInt();
    auto pLWE = modulus_LWE / (2 * beta);
    double scaleSignFHEW = 8.0;
    cc->EvalCompareSwitchPrecompute(pLWE, scaleSignFHEW);

    // Test inputs
    std::vector<double> input = {-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    std::cout << "Input values: ";
    for (const auto& val : input) {
        std::cout << std::fixed << std::setprecision(2) << val << " ";
    }
    std::cout << std::endl;

    // Expected ReLU output
    std::vector<double> expected(slots);
    for (size_t i = 0; i < slots; ++i) {
        expected[i] = std::max(0.0, input[i]);
    }
    std::cout << "Expected ReLU: ";
    for (const auto& val : expected) {
        std::cout << std::fixed << std::setprecision(2) << val << " ";
    }
    std::cout << std::endl << std::endl;

    // Encode and encrypt
    TIC(t);
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(input, 1, 0, nullptr, slots);
    auto ctInput = cc->Encrypt(keys.publicKey, ptxt);
    std::cout << "Encryption: " << TOC(t) << " ms" << std::endl;

    // Create ciphertext with zeros for comparison
    std::vector<double> zeros(slots, 0.0);
    Plaintext ptxtZero = cc->MakeCKKSPackedPlaintext(zeros, 1, 0, nullptr, slots);
    auto ctZero = cc->Encrypt(keys.publicKey, ptxtZero);

    // ReLU(x) = x * (x > 0)
    // Step 1: Compute comparison result (x > 0)
    std::cout << "\nComputing ReLU = x * (x > 0)..." << std::endl;
    TIC(t);
    auto ctComparison = cc->EvalCompareSchemeSwitching(ctInput, ctZero, slots, slots, 0, scaleSignFHEW);
    double compTime = TOC(t);
    std::cout << "Comparison (x > 0): " << compTime << " ms" << std::endl;

    // Step 2: Multiply input by comparison result
    TIC(t);
    auto ctReLU = cc->EvalMult(ctInput, cc->EvalAdd(cc->EvalMult(ctComparison, -1), 1));
    double multTime = TOC(t);
    std::cout << "Multiplication: " << multTime << " ms" << std::endl;
    std::cout << "Total ReLU computation: " << (compTime + multTime) << " ms" << std::endl;

    // Decrypt and verify
    TIC(t);
    Plaintext plaintextDec;
    cc->Decrypt(keys.secretKey, ctReLU, &plaintextDec);
    plaintextDec->SetLength(slots);
    std::cout << "Decryption: " << TOC(t) << " ms" << std::endl;

    std::vector<std::complex<double>> result = plaintextDec->GetCKKSPackedValue();
    std::cout << "\nActual output: ";
    for (size_t i = 0; i < slots; ++i) {
        std::cout << std::fixed << std::setprecision(4) << result[i].real() << " ";
    }
    std::cout << std::endl;

    // Compute error
    std::cout << "\nError analysis:" << std::endl;
    double maxError = 0.0;
    double avgError = 0.0;
    for (size_t i = 0; i < slots; ++i) {
        double error = std::abs(expected[i] - result[i].real());
        avgError += error;
        maxError = std::max(maxError, error);
        std::cout << "  Input " << std::setw(6) << input[i]
                  << " | Expected " << std::setw(6) << expected[i]
                  << " | Got " << std::setw(8) << result[i].real()
                  << " | Error " << std::setw(8) << error << std::endl;
    }
    avgError /= slots;
    std::cout << "\nMax error: " << maxError << std::endl;
    std::cout << "Avg error: " << avgError << std::endl;
    std::cout << "\n✓ ReLU Scheme Switching Test Complete!\n" << std::endl;
}

/**
 * @brief Main function
 */
int main(int argc, char* argv[]) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  OPENFHE: ReLU ACTIVATION FUNCTION TEST" << std::endl;
    std::cout << "  Testing Two Methods: Approximation vs Scheme Switching" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    int choice = 0;

    if (argc > 1) {
        choice = atoi(argv[1]);
    } else {
        std::cout << "\nSelect test to run:" << std::endl;
        std::cout << "1. ReLU via Approximation (Chebyshev)" << std::endl;
        std::cout << "2. ReLU via Approximation (Minimax)" << std::endl;
        std::cout << "3. ReLU via Scheme Switching (CKKS-FHEW-CKKS)" << std::endl;
        std::cout << "4. Run Both Tests + Comparison" << std::endl;
        std::cout << "Enter choice (default=4): ";
        std::cin >> choice;
    }

    try {
        switch (choice) {
            case 1:
                TestReLUApproximation1();
                break;
            case 2:
                TestReLUApproximation2();
                break;
            case 3:
                TestReLUSchemeSwitching();
                break;
            case 4:
            default:
                TestReLUApproximation1();
                TestReLUApproximation2();
                TestReLUSchemeSwitching();
                break;
        }

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  ALL TESTS COMPLETED SUCCESSFULLY!" << std::endl;
        std::cout << std::string(70, '=') << "\n" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
