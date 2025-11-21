#include "openfhe.h"
#include <chrono>
#include <iomanip>

using namespace lbcrypto;

void Test1_ConsecutiveSchemeSwitching() {
    std::cout << "\n========== Test 1: Consecutive Scheme Switching WITHOUT Bootstrapping ==========\n" << std::endl;

    auto setup_start = std::chrono::high_resolution_clock::now();

    // Setup CryptoContext for CKKS with enough depth for two scheme switchings
    // First scheme switching uses 13 depth, second uses 1 depth
    // We need: 13 (first switch) + 1 (2x operation) + 1 (subtraction) + 1 (second switch) = 16 depth minimum
    ScalingTechnique scTech = FLEXIBLEAUTO;
    uint32_t multDepth      = 20;  // Extra buffer for safety

    uint32_t scaleModSize = 50;
    uint32_t firstModSize = 60;
    uint32_t ringDim      = 8192;
    SecurityLevel sl      = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE   = 25;
    uint32_t slots        = 16;
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

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(SCHEMESWITCH);

    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension();
    std::cout << ", number of slots " << slots << ", and supports a multiplicative depth of " << multDepth << std::endl;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // Setup FHEW for scheme switching
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

    auto modulus_LWE = 1 << logQ_ccLWE;
    auto beta        = ccLWE->GetBeta().ConvertToInt();
    auto pLWE2       = modulus_LWE / (2 * beta);
    double scaleSignFHEW = 8.0;
    cc->EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW);

    auto setup_end = std::chrono::high_resolution_clock::now();
    auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start);
    std::cout << "Setup time: " << setup_duration.count() << " ms" << std::endl << std::endl;

    // Prepare input data
    std::vector<double> x1 = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    std::vector<double> x_zero(slots, 0.0);

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);
    Plaintext ptxt_zero = cc->MakeCKKSPackedPlaintext(x_zero, 1, 0, nullptr, slots);

    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c_zero = cc->Encrypt(keys.publicKey, ptxt_zero);

    std::cout << "Initial levels remaining: " << multDepth - c1->GetLevel() << std::endl;

    // First scheme switching: Compare with 0 (SignEval)
    auto compare1_start = std::chrono::high_resolution_clock::now();
    auto cResult1 = cc->EvalCompareSchemeSwitching(c1, c_zero, slots, slots);
    auto compare1_end = std::chrono::high_resolution_clock::now();
    auto compare1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(compare1_end - compare1_start);

    std::cout << "Levels remaining after first scheme switching: " << multDepth - cResult1->GetLevel() << std::endl;

    Plaintext result1;
    cc->Decrypt(keys.secretKey, cResult1, &result1);
    result1->SetLength(slots);
    std::cout << "First comparison result (sign of x1): " << result1 << std::endl;

    // Perform 2x - 1 operation
    auto arith_start = std::chrono::high_resolution_clock::now();
    auto cDoubled = cc->EvalMult(cResult1, 2.0);  // 2x
    auto cTransformed = cc->EvalSub(cDoubled, 1.0);  // 2x - 1
    auto arith_end = std::chrono::high_resolution_clock::now();
    auto arith_duration = std::chrono::duration_cast<std::chrono::milliseconds>(arith_end - arith_start);

    std::cout << "Levels remaining after 2x - 1: " << multDepth - cTransformed->GetLevel() << std::endl;

    // Perform second 2x - 1 operation
    auto arith_start2 = std::chrono::high_resolution_clock::now();
    auto cDoubled2 = cc->EvalMult(cTransformed, 2.0);  // 2x
    auto cTransformed2 = cc->EvalSub(cDoubled2, 1.0);  // 2x - 1
    auto arith_end2 = std::chrono::high_resolution_clock::now();
    auto arith_duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(arith_end2 - arith_start2);

    std::cout << "Levels remaining after second 2x - 1: " << multDepth - cTransformed2->GetLevel() << std::endl;

    Plaintext transformed2;
    cc->Decrypt(keys.secretKey, cTransformed2, &transformed2);
    transformed2->SetLength(slots);
    std::cout << "After 2x - 1: " << transformed2 << std::endl;

    Plaintext transformed;
    cc->Decrypt(keys.secretKey, cTransformed, &transformed);
    transformed->SetLength(slots);
    std::cout << "After 2x - 1: " << transformed << std::endl;

    // Second scheme switching: Compare with 0 again
    auto compare2_start = std::chrono::high_resolution_clock::now();
    auto cResult2 = cc->EvalCompareSchemeSwitching(cTransformed, c_zero, slots, slots);
    auto compare2_end = std::chrono::high_resolution_clock::now();
    auto compare2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(compare2_end - compare2_start);

    std::cout << "Levels remaining after second scheme switching: " << multDepth - cResult2->GetLevel() << std::endl;

    Plaintext result2;
    cc->Decrypt(keys.secretKey, cResult2, &result2);
    result2->SetLength(slots);
    std::cout << "Second comparison result (sign of 2x - 1): " << result2 << std::endl;

    auto total_duration = compare1_duration + arith_duration + compare2_duration;

    std::cout << "\n----- Test 1 Timing Summary -----" << std::endl;
    std::cout << "First scheme switching:  " << compare1_duration.count() << " ms" << std::endl;
    std::cout << "Arithmetic (2x - 1):     " << arith_duration.count() << " ms" << std::endl;
    std::cout << "Second scheme switching: " << compare2_duration.count() << " ms" << std::endl;
    std::cout << "Total computation time:  " << total_duration.count() << " ms" << std::endl;
}

void Test2_ConsecutiveSchemeSwitchingWithBootstrapping() {
    std::cout << "\n========== Test 2: Consecutive Scheme Switching WITH Bootstrapping ==========\n" << std::endl;

    auto setup_start = std::chrono::high_resolution_clock::now();

    // Setup CryptoContext with bootstrapping support
    ScalingTechnique scTech = FLEXIBLEAUTO;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;

    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;
    uint32_t ringDim      = 8192;
    SecurityLevel sl      = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE   = 25;
    uint32_t slots        = 16;
    uint32_t batchSize    = slots;

    // Calculate depth for bootstrapping
    std::vector<uint32_t> levelBudget = {4, 4};
    uint32_t levelsAvailableAfterBootstrap = 15;
    uint32_t approxBootstrapDepth = FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    uint32_t multDepth = levelsAvailableAfterBootstrap + approxBootstrapDepth;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetScalingTechnique(scTech);
    parameters.SetSecurityLevel(sl);
    parameters.SetRingDim(ringDim);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetNumLargeDigits(3);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(SCHEMESWITCH);
    cc->Enable(FHE);

    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension();
    std::cout << ", number of slots " << slots << ", and supports a multiplicative depth of " << multDepth << std::endl;

    // Setup bootstrapping with bsgsDim and numSlots parameters
    std::vector<uint32_t> bsgsDim = {0, 0};
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, slots);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    // Setup FHEW for scheme switching
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

    auto modulus_LWE = 1 << logQ_ccLWE;
    auto beta        = ccLWE->GetBeta().ConvertToInt();
    auto pLWE2       = modulus_LWE / (2 * beta);
    double scaleSignFHEW = 8.0;
    cc->EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW);

    auto setup_end = std::chrono::high_resolution_clock::now();
    auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start);
    std::cout << "Setup time (including bootstrapping keys): " << setup_duration.count() << " ms" << std::endl << std::endl;

    // Prepare input data
    std::vector<double> x1 = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    std::vector<double> x_zero(slots, 0.0);

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);
    Plaintext ptxt_zero = cc->MakeCKKSPackedPlaintext(x_zero, 1, 0, nullptr, slots);

    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c_zero = cc->Encrypt(keys.publicKey, ptxt_zero);

    std::cout << "Initial levels remaining: " << multDepth - c1->GetLevel() << std::endl;

    // First scheme switching: Compare with 0 (SignEval)
    auto compare1_start = std::chrono::high_resolution_clock::now();
    auto cResult1 = cc->EvalCompareSchemeSwitching(c1, c_zero, slots, slots);
    auto compare1_end = std::chrono::high_resolution_clock::now();
    auto compare1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(compare1_end - compare1_start);

    std::cout << "Levels remaining after first scheme switching: " << multDepth - cResult1->GetLevel() << std::endl;

    Plaintext result1;
    cc->Decrypt(keys.secretKey, cResult1, &result1);
    result1->SetLength(slots);
    std::cout << "First comparison result (sign of x1): " << result1 << std::endl;

    // Perform 2x - 1 operation
    auto arith_start = std::chrono::high_resolution_clock::now();
    auto cDoubled = cc->EvalMult(cResult1, 2.0);  // 2x
    auto cTransformed = cc->EvalSub(cDoubled, 1.0);  // 2x - 1
    auto arith_end = std::chrono::high_resolution_clock::now();
    auto arith_duration = std::chrono::duration_cast<std::chrono::milliseconds>(arith_end - arith_start);

    std::cout << "Levels remaining after 2x - 1: " << multDepth - cTransformed->GetLevel() << std::endl;

    // Perform second 2x - 1 operation
    auto arith_start2 = std::chrono::high_resolution_clock::now();
    auto cDoubled2 = cc->EvalMult(cTransformed, 2.0);  // 2x
    auto cTransformed2 = cc->EvalSub(cDoubled2, 1.0);  // 2x - 1
    auto arith_end2 = std::chrono::high_resolution_clock::now();
    auto arith_duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(arith_end2 - arith_start2);

    std::cout << "Levels remaining after second 2x - 1: " << multDepth - cTransformed2->GetLevel() << std::endl;

    // Consume more levels to force bootstrapping to be necessary
    std::cout << "\n--- Consuming more levels to test bootstrapping ---" << std::endl;
    auto cTemp = cTransformed2;
    for (int i = 0; i < 16; ++i) {
        cTemp = cc->EvalMult(cTemp, 1.001);  // Small multiplications to consume levels
        std::cout << "After multiplication " << (i+1) << ": " << (multDepth - cTemp->GetLevel()) << " levels remaining" << std::endl;
    }

    Plaintext transformed2;
    cc->Decrypt(keys.secretKey, cTemp, &transformed2);
    transformed2->SetLength(slots);
    std::cout << "After consuming levels: " << transformed2 << std::endl;

    // Perform bootstrapping
    std::cout << "\n--- Bootstrapping ---" << std::endl;
    std::cout << "Before bootstrap:" << std::endl;
    std::cout << "  - Ciphertext level: " << cTemp->GetLevel() << std::endl;
    std::cout << "  - Levels remaining: " << multDepth - cTemp->GetLevel() << std::endl;

    auto bootstrap_start = std::chrono::high_resolution_clock::now();
    auto cBootstrapped = cc->EvalBootstrap(cTemp);
    auto bootstrap_end = std::chrono::high_resolution_clock::now();
    auto bootstrap_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bootstrap_end - bootstrap_start);

    std::cout << "After bootstrap:" << std::endl;
    std::cout << "  - Ciphertext level: " << cBootstrapped->GetLevel() << std::endl;
    std::cout << "  - Levels remaining: " << multDepth - cBootstrapped->GetLevel() << std::endl;
    std::cout << "  - Expected refresh target: ~" << levelsAvailableAfterBootstrap << " levels" << std::endl;

    Plaintext bootstrapped;
    cc->Decrypt(keys.secretKey, cBootstrapped, &bootstrapped);
    bootstrapped->SetLength(slots);
    std::cout << "After bootstrapping: " << bootstrapped << std::endl;

    // Second scheme switching: Compare with 0 again
    auto compare2_start = std::chrono::high_resolution_clock::now();
    auto cResult2 = cc->EvalCompareSchemeSwitching(cBootstrapped, c_zero, slots, slots);
    auto compare2_end = std::chrono::high_resolution_clock::now();
    auto compare2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(compare2_end - compare2_start);

    std::cout << "Levels remaining after second scheme switching: " << multDepth - cResult2->GetLevel() << std::endl;

    Plaintext result2;
    cc->Decrypt(keys.secretKey, cResult2, &result2);
    result2->SetLength(slots);
    std::cout << "Second comparison result (sign of 2x - 1): " << result2 << std::endl;

    auto total_duration = compare1_duration + arith_duration + bootstrap_duration + compare2_duration;

    std::cout << "\n----- Test 2 Timing Summary -----" << std::endl;
    std::cout << "First scheme switching:  " << compare1_duration.count() << " ms" << std::endl;
    std::cout << "Arithmetic (2x - 1):     " << arith_duration.count() << " ms" << std::endl;
    std::cout << "Bootstrapping:           " << bootstrap_duration.count() << " ms" << std::endl;
    std::cout << "Second scheme switching: " << compare2_duration.count() << " ms" << std::endl;
    std::cout << "Total computation time:  " << total_duration.count() << " ms" << std::endl;
}

int main() {
    std::cout << "Testing consecutive scheme switching operations with depth consumption analysis\n";
    std::cout << "First scheme switching consumes 13 levels, subsequent ones consume 1 level\n";
    std::cout << std::string(80, '=') << std::endl;

    try {
        Test1_ConsecutiveSchemeSwitching();
    } catch (const std::exception& e) {
        std::cerr << "\nTest 1 failed with error: " << e.what() << std::endl;
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;

    try {
        Test2_ConsecutiveSchemeSwitchingWithBootstrapping();
    } catch (const std::exception& e) {
        std::cerr << "\nTest 2 failed with error: " << e.what() << std::endl;
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "\n========== COMPARISON SUMMARY ==========\n" << std::endl;
    std::cout << "Test 1 shows depth consumption without bootstrapping" << std::endl;
    std::cout << "Test 2 shows how bootstrapping refreshes levels for additional operations" << std::endl;
    std::cout << "Note: Bootstrapping adds significant latency but enables unlimited depth" << std::endl;

    return 0;
}
