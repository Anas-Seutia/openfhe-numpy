#include "openfhe.h"
#include <chrono>
#include <iomanip>

using namespace lbcrypto;

void ComparisonViaSchemeSwitching() {
    std::cout << "\n-----ComparisonViaSchemeSwitching-----\n" << std::endl;
    std::cout << "Output precision is only wrt the operations in CKKS after switching back.\n" << std::endl;

    // Timer for setup
    auto setup_start = std::chrono::high_resolution_clock::now();

    // Step 1: Setup CryptoContext for CKKS
    ScalingTechnique scTech = FLEXIBLEAUTO;
    uint32_t multDepth      = 17;
    if (scTech == FLEXIBLEAUTOEXT)
        multDepth += 1;

    uint32_t scaleModSize = 50;
    uint32_t firstModSize = 60;
    uint32_t ringDim      = 8192;
    SecurityLevel sl      = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE   = 25;
    uint32_t slots        = 16;  // sparsely-packed
    uint32_t batchSize    = slots;

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

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(SCHEMESWITCH);

    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension();
    std::cout << ", number of slots " << slots << ", and supports a multiplicative depth of " << multDepth << std::endl
              << std::endl;

    // Generate encryption keys
    auto keys = cc->KeyGen();

    // Step 2: Prepare the FHEW cryptocontext and keys for FHEW and scheme switching
    SchSwchParams params;
    params.SetSecurityLevelCKKS(sl);
    params.SetSecurityLevelFHEW(slBin);
    params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE);
    params.SetNumSlotsCKKS(slots);
    params.SetNumValues(slots);
    auto privateKeyFHEW = cc->EvalSchemeSwitchingSetup(params);
    auto ccLWE          = cc->GetBinCCForSchemeSwitch();

    ccLWE->BTKeyGen(privateKeyFHEW);
    cc->EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW);

    std::cout << "FHEW scheme is using lattice parameter " << ccLWE->GetParams()->GetLWEParams()->Getn();
    std::cout << ", logQ " << logQ_ccLWE;
    std::cout << ", and modulus q " << ccLWE->GetParams()->GetLWEParams()->Getq() << std::endl << std::endl;

    // Set the scaling factor to be able to decrypt; the LWE mod switch is performed on the ciphertext at the last level
    auto pLWE1           = ccLWE->GetMaxPlaintextSpace().ConvertToInt();  // Small precision
    auto modulus_LWE     = 1 << logQ_ccLWE;
    auto beta            = ccLWE->GetBeta().ConvertToInt();
    auto pLWE2           = modulus_LWE / (2 * beta);  // Large precision

    auto setup_end = std::chrono::high_resolution_clock::now();
    auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start);
    std::cout << "Setup time: " << setup_duration.count() << " ms" << std::endl << std::endl;

    // Step 3: Encoding and encryption of inputs
    auto encrypt_start = std::chrono::high_resolution_clock::now();

    // Inputs
    std::vector<double> x1 = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    std::vector<double> x2(slots, 5.25);

    // Encoding as plaintexts
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2, 1, 0, nullptr, slots);

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    // Compute the difference to compare to zero
    auto cDiff = cc->EvalSub(c1, c2);

    auto encrypt_end = std::chrono::high_resolution_clock::now();
    auto encrypt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(encrypt_end - encrypt_start);
    std::cout << "Encryption and difference computation time: " << encrypt_duration.count() << " ms" << std::endl << std::endl;

    // Step 4: CKKS to FHEW switching and sign evaluation to test correctness
    Plaintext pDiff;
    cc->Decrypt(keys.secretKey, cDiff, &pDiff);
    pDiff->SetLength(slots);
    std::cout << "Difference of inputs: ";
    auto pDiff_rpv = pDiff->GetRealPackedValue();
    for (uint32_t i = 0; i < slots; ++i) {
        std::cout << pDiff_rpv[i] << " ";
    }

    const double eps = 0.0001;
    std::cout << "\nExpected sign result from CKKS: ";
    for (uint32_t i = 0; i < slots; ++i) {
        std::cout << int(std::round(pDiff_rpv[i] / eps) * eps < 0) << " ";
    }
    std::cout << "\n";

    // ============================================================
    // Test 1: scaleSignFHEW = 1.0, pLWE2 (large precision)
    // ============================================================
    std::cout << "\n========== Test 1: scaleSignFHEW = 1.0, pLWE2 (large precision) ==========\n" << std::endl;

    auto test1_precompute_start = std::chrono::high_resolution_clock::now();
    double scaleSignFHEW = 1.0;
    cc->EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW);
    auto test1_precompute_end = std::chrono::high_resolution_clock::now();
    auto test1_precompute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test1_precompute_end - test1_precompute_start);
    std::cout << "Precompute time: " << test1_precompute_duration.count() << " ms" << std::endl;

    auto test1_switch_start = std::chrono::high_resolution_clock::now();
    auto LWECiphertexts = cc->EvalCKKStoFHEW(cDiff, slots);
    auto test1_switch_end = std::chrono::high_resolution_clock::now();
    auto test1_switch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test1_switch_end - test1_switch_start);
    std::cout << "CKKS to FHEW switch time: " << test1_switch_duration.count() << " ms" << std::endl;

    LWEPlaintext plainLWE;
    std::cout << "\nFHEW decryption with plaintext modulus " << NativeInteger(pLWE2) << ": ";
    for (uint32_t i = 0; i < LWECiphertexts.size(); ++i) {
        ccLWE->Decrypt(privateKeyFHEW, LWECiphertexts[i], &plainLWE, pLWE2);
        std::cout << plainLWE << " ";
    }

    std::cout << "\nExpected sign result in FHEW with plaintext modulus " << NativeInteger(pLWE2) << " and scale "
              << scaleSignFHEW << ": ";
    for (uint32_t i = 0; i < slots; ++i) {
        std::cout << (static_cast<int>(std::round(pDiff_rpv[i] * scaleSignFHEW)) % pLWE2 - pLWE2 / 2.0 >= 0) << " ";
    }
    std::cout << "\n";

    auto test1_sign_start = std::chrono::high_resolution_clock::now();
    std::cout << "Obtained sign result in FHEW with plaintext modulus " << NativeInteger(pLWE2) << " and scale "
              << scaleSignFHEW << ": ";
    std::vector<LWECiphertext> LWESign(LWECiphertexts.size());
    for (uint32_t i = 0; i < LWECiphertexts.size(); ++i) {
        LWESign[i] = ccLWE->EvalSign(LWECiphertexts[i]);
        ccLWE->Decrypt(privateKeyFHEW, LWESign[i], &plainLWE, 2);
        std::cout << plainLWE << " ";
    }
    std::cout << "\n";
    auto test1_sign_end = std::chrono::high_resolution_clock::now();
    auto test1_sign_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test1_sign_end - test1_sign_start);
    std::cout << "Sign evaluation time: " << test1_sign_duration.count() << " ms" << std::endl;

    // Step 5: Direct comparison via CKKS->FHEW->CKKS
    auto test1_compare_start = std::chrono::high_resolution_clock::now();
    auto cResult = cc->EvalCompareSchemeSwitching(c1, c2, slots, slots);
    auto test1_compare_end = std::chrono::high_resolution_clock::now();
    auto test1_compare_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test1_compare_end - test1_compare_start);
    std::cout << "Direct comparison (CKKS->FHEW->CKKS) time: " << test1_compare_duration.count() << " ms" << std::endl;

    Plaintext plaintextDec3;
    cc->Decrypt(keys.secretKey, cResult, &plaintextDec3);
    plaintextDec3->SetLength(slots);
    std::cout << "Decrypted switched result: " << plaintextDec3 << std::endl;

    auto test1_total_duration = test1_precompute_duration + test1_switch_duration + test1_sign_duration + test1_compare_duration;
    std::cout << "\nTest 1 Total time: " << test1_total_duration.count() << " ms" << std::endl;

    // ============================================================
    // Test 2: scaleSignFHEW = 8.0, pLWE2 (large precision)
    // ============================================================
    std::cout << "\n========== Test 2: scaleSignFHEW = 8.0, pLWE2 (large precision) ==========\n" << std::endl;

    auto test2_precompute_start = std::chrono::high_resolution_clock::now();
    scaleSignFHEW = 8.0;
    cc->EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW);
    auto test2_precompute_end = std::chrono::high_resolution_clock::now();
    auto test2_precompute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test2_precompute_end - test2_precompute_start);
    std::cout << "Precompute time: " << test2_precompute_duration.count() << " ms" << std::endl;

    auto test2_switch_start = std::chrono::high_resolution_clock::now();
    LWECiphertexts = cc->EvalCKKStoFHEW(cDiff, slots);
    auto test2_switch_end = std::chrono::high_resolution_clock::now();
    auto test2_switch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test2_switch_end - test2_switch_start);
    std::cout << "CKKS to FHEW switch time: " << test2_switch_duration.count() << " ms" << std::endl;

    std::cout << "\nFHEW decryption with plaintext modulus " << NativeInteger(pLWE2) << " and scale " << scaleSignFHEW
              << ": ";
    for (uint32_t i = 0; i < LWECiphertexts.size(); ++i) {
        ccLWE->Decrypt(privateKeyFHEW, LWECiphertexts[i], &plainLWE, pLWE2);
        std::cout << plainLWE << " ";
    }
    std::cout << "\nExpected sign result in FHEW with plaintext modulus " << NativeInteger(pLWE2) << " and scale "
              << scaleSignFHEW << ": ";
    for (uint32_t i = 0; i < slots; ++i) {
        std::cout << (static_cast<int>(std::round(pDiff_rpv[i] * scaleSignFHEW)) % pLWE2 - pLWE2 / 2.0 >= 0) << " ";
    }
    std::cout << "\n";

    auto test2_sign_start = std::chrono::high_resolution_clock::now();
    std::cout << "Obtained sign result in FHEW with plaintext modulus " << NativeInteger(pLWE2) << " and scale "
              << scaleSignFHEW << ": ";
    for (uint32_t i = 0; i < LWECiphertexts.size(); ++i) {
        LWESign[i] = ccLWE->EvalSign(LWECiphertexts[i]);
        ccLWE->Decrypt(privateKeyFHEW, LWESign[i], &plainLWE, 2);
        std::cout << plainLWE << " ";
    }
    std::cout << "\n";
    auto test2_sign_end = std::chrono::high_resolution_clock::now();
    auto test2_sign_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test2_sign_end - test2_sign_start);
    std::cout << "Sign evaluation time: " << test2_sign_duration.count() << " ms" << std::endl;

    auto test2_compare_start = std::chrono::high_resolution_clock::now();
    cResult = cc->EvalCompareSchemeSwitching(c1, c2, slots, slots);
    auto test2_compare_end = std::chrono::high_resolution_clock::now();
    auto test2_compare_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test2_compare_end - test2_compare_start);
    std::cout << "Direct comparison (CKKS->FHEW->CKKS) time: " << test2_compare_duration.count() << " ms" << std::endl;

    cc->Decrypt(keys.secretKey, cResult, &plaintextDec3);
    plaintextDec3->SetLength(slots);
    std::cout << "Decrypted switched result: " << plaintextDec3 << std::endl;

    auto test2_total_duration = test2_precompute_duration + test2_switch_duration + test2_sign_duration + test2_compare_duration;
    std::cout << "\nTest 2 Total time: " << test2_total_duration.count() << " ms" << std::endl;

    // ============================================================
    // Test 3: scaleSignFHEW = 1.0, pLWE1 (small precision)
    // ============================================================
    std::cout << "\n========== Test 3: scaleSignFHEW = 1.0, pLWE1 (small precision) ==========\n" << std::endl;
    std::cout
        << "For very small LWE plaintext modulus and initial fractional inputs, the sign does not always behave properly close to the boundaries at 0 and p/2."
        << std::endl;

    auto test3_precompute_start = std::chrono::high_resolution_clock::now();
    scaleSignFHEW = 1.0;
    cc->EvalCompareSwitchPrecompute(pLWE1, scaleSignFHEW);
    auto test3_precompute_end = std::chrono::high_resolution_clock::now();
    auto test3_precompute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test3_precompute_end - test3_precompute_start);
    std::cout << "Precompute time: " << test3_precompute_duration.count() << " ms" << std::endl;

    auto test3_switch_start = std::chrono::high_resolution_clock::now();
    LWECiphertexts = cc->EvalCKKStoFHEW(cDiff, slots);
    auto test3_switch_end = std::chrono::high_resolution_clock::now();
    auto test3_switch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test3_switch_end - test3_switch_start);
    std::cout << "CKKS to FHEW switch time: " << test3_switch_duration.count() << " ms" << std::endl;

    std::cout << "\nFHEW decryption with plaintext modulus " << NativeInteger(pLWE1) << ": ";
    for (uint32_t i = 0; i < LWECiphertexts.size(); ++i) {
        ccLWE->Decrypt(privateKeyFHEW, LWECiphertexts[i], &plainLWE, pLWE1);
        std::cout << plainLWE << " ";
    }
    std::cout << "\nExpected sign result in FHEW with plaintext modulus " << NativeInteger(pLWE1) << " and scale "
              << scaleSignFHEW << ": ";
    for (uint32_t i = 0; i < slots; ++i) {
        std::cout << (static_cast<int>(std::round(pDiff_rpv[i] * scaleSignFHEW)) % pLWE1 - pLWE1 / 2.0 >= 0) << " ";
    }
    std::cout << "\n";

    auto test3_sign_start = std::chrono::high_resolution_clock::now();
    std::cout << "Obtained sign result in FHEW with plaintext modulus " << NativeInteger(pLWE1) << " and scale "
              << scaleSignFHEW << ": ";
    for (uint32_t i = 0; i < LWECiphertexts.size(); ++i) {
        LWESign[i] = ccLWE->EvalSign(LWECiphertexts[i]);
        ccLWE->Decrypt(privateKeyFHEW, LWESign[i], &plainLWE, 2);
        std::cout << plainLWE << " ";
    }
    std::cout << "\n";
    auto test3_sign_end = std::chrono::high_resolution_clock::now();
    auto test3_sign_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test3_sign_end - test3_sign_start);
    std::cout << "Sign evaluation time: " << test3_sign_duration.count() << " ms" << std::endl;

    auto test3_compare_start = std::chrono::high_resolution_clock::now();
    cResult = cc->EvalCompareSchemeSwitching(c1, c2, slots, slots, 0, scaleSignFHEW);
    auto test3_compare_end = std::chrono::high_resolution_clock::now();
    auto test3_compare_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test3_compare_end - test3_compare_start);
    std::cout << "Direct comparison (CKKS->FHEW->CKKS) time: " << test3_compare_duration.count() << " ms" << std::endl;

    cc->Decrypt(keys.secretKey, cResult, &plaintextDec3);
    plaintextDec3->SetLength(slots);
    std::cout << "Decrypted switched result: " << plaintextDec3 << std::endl;

    auto test3_total_duration = test3_precompute_duration + test3_switch_duration + test3_sign_duration + test3_compare_duration;
    std::cout << "\nTest 3 Total time: " << test3_total_duration.count() << " ms" << std::endl;

    // ============================================================
    // Summary
    // ============================================================
    std::cout << "\n========== TIMING SUMMARY ==========\n" << std::endl;
    std::cout << std::setw(40) << "Operation" << std::setw(15) << "Time (ms)" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    std::cout << std::setw(40) << "Setup" << std::setw(15) << setup_duration.count() << std::endl;
    std::cout << std::setw(40) << "Encryption" << std::setw(15) << encrypt_duration.count() << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(40) << "Test 1 (scale=1.0, pLWE2)" << std::setw(15) << test1_total_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - Precompute" << std::setw(15) << test1_precompute_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - CKKS->FHEW switch" << std::setw(15) << test1_switch_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - Sign evaluation" << std::setw(15) << test1_sign_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - Direct comparison" << std::setw(15) << test1_compare_duration.count() << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(40) << "Test 2 (scale=8.0, pLWE2)" << std::setw(15) << test2_total_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - Precompute" << std::setw(15) << test2_precompute_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - CKKS->FHEW switch" << std::setw(15) << test2_switch_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - Sign evaluation" << std::setw(15) << test2_sign_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - Direct comparison" << std::setw(15) << test2_compare_duration.count() << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(40) << "Test 3 (scale=1.0, pLWE1)" << std::setw(15) << test3_total_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - Precompute" << std::setw(15) << test3_precompute_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - CKKS->FHEW switch" << std::setw(15) << test3_switch_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - Sign evaluation" << std::setw(15) << test3_sign_duration.count() << std::endl;
    std::cout << std::setw(40) << "  - Direct comparison" << std::setw(15) << test3_compare_duration.count() << std::endl;
    std::cout << std::endl;
    std::cout << std::string(55, '=') << std::endl;
}

int main() {
    ComparisonViaSchemeSwitching();
    return 0;
}
