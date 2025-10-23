#ifndef __RELU_HELPER_FUNCTION_H__
#define __RELU_HELPER_FUNCTION_H__

#include "openfhe.h"
#include <cstdint>
#include <vector>
#include <functional>

using namespace lbcrypto;

/**
 * @brief Generate MiniMax coefficients for composite sign function approximation
 * 
 * Following the multi-interval Remez approach from https://eprint.iacr.org/2020/834.pdf
 * 
 * The function approximates sign(x) using a composite polynomial approach where
 * each stage refines the approximation on narrowing intervals.
 * 
 * Stage 0: Approximates on [-1-ε, -α] ∪ [α, 1+ε] (excludes near-zero region)
 * Stage k: Approximates on narrower intervals based on previous stage errors
 * Final stage: Maps to [0, 1] for ReLU
 * 
 * @param degrees Polynomial degrees for each stage [deg(p0), deg(p1), ..., deg(pk)]
 * @param prec Precision in bits (unused in this simplified version)
 * @param logalpha log₂(α) where α is the distinguishing precision (default: 6 → α=2^-6)
 * @param logerr log₂(ε) where ε is scheme error upperbound (default: 12 → ε=2^-12)
 * @param debug Enable debug output (default: false)
 * @return std::vector<std::vector<double>> Coefficients for each stage
 * 
 * Example usage:
 * ```cpp
 * // Generate coefficients for 3-stage approximation
 * std::vector<uint32_t> degrees = {15, 15, 27};
 * auto coeffs = GenerateMiniMaxSignCoeffs(degrees, 128, 6, 12, true);
 * ```
 * 
 * @note This implementation uses Chebyshev approximation as a substitute for
 *       the full Remez algorithm. For optimal MiniMax polynomials, consider
 *       using external tools like Sollya or pre-computed coefficients.
 */
std::vector<std::vector<double>> GenerateMiniMaxSignCoefficients(
    const std::vector<uint32_t>& degrees,
    uint32_t logalpha = 6,
    uint32_t logerr = 12
);

/**
 * @brief Evaluate homomorphic ReLU: ReLU(x) ≈ x · sign(x)
 * 
 * @param cc Crypto context
 * @param ct Input ciphertext
 * @param signCoeffs Coefficients from GenerateMiniMaxSignCoeffs
 * @param lowerBound Used for lower bound scalling before/after sign evaluation (default: No Rescaling)
 * @param upperBound Used for upper bound scalling before/after sign evaluation (default: No Rescaling)
 * @return Ciphertext<DCRTPoly> Encrypted ReLU(ct)
 * 
 * Example:
 * ```cpp
 * auto coeffs = GenerateMiniMaxSignCoefficients({15, 15, 27});
 * auto ctReLU = EvalMiniMaxSign(cc, ctInput, coeffs);
 * ```
 */
Ciphertext<DCRTPoly> EvalMiniMaxSign(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct,
    const std::vector<std::vector<double>>& signCoeffs,
    double upperBound = 0.0,
    double lowerBound = 0.0
);

#endif  // __RELU_HELPER_FUNCTION_H__