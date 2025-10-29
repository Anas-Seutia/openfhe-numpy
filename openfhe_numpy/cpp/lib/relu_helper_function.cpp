#include "relu_helper_function.h"
#include "math/chebyshev.h"
#include "utils/exception.h"

#include <openfhe.h>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

using namespace lbcrypto; 

std::vector<std::vector<double>> GenerateMiniMaxSignCoefficients(
    const std::vector<uint32_t>& degrees,
    uint32_t logalpha,
    uint32_t logerr) {
    
    if (degrees.empty())
        OPENFHE_THROW("Degrees vector cannot be empty");
    
    std::vector<std::vector<double>> allCoeffs;
    allCoeffs.reserve(degrees.size());

    // Parameters for MiniMax approximation
    // double alpha = std::pow(2.0, logalpha);  // Transition sharpness (unused in simplified version)
    double epsilon = std::pow(2.0, -logerr);  // Target error
    (void)logalpha;  // Suppress unused parameter warning

    {
        uint32_t degree = degrees[0];

        // Approximate sign(x) = -1 if x < 0, +1 if x > 0
        std::function<double(double)> sign_func = [](double x) -> double {
            return (x < 0) ? -1.0 : 1.0;
        };

        // Generate Chebyshev approximation on [-1, 1]
        // Note: This is a simplified version. Full Remez would be more accurate.
        std::vector<double> coeffs = EvalChebyshevCoefficients(sign_func, -1.0, 1.0, degree);
    
        allCoeffs.push_back(coeffs);
    }
    
    // Intermediate stages: Iterative refinement
    // Each stage works on a narrower interval based on previous approximation errors
    // double prev_max_error = 0.1;  // Initial conservative estimate (unused in simplified version)

    for (size_t stage = 1; stage < degrees.size(); ++stage) {
        uint32_t degree = degrees[stage];
        bool is_last_stage = (stage == degrees.size() - 1);

        // New interval = [-(1+max_err+ε), -(1-ε)] ∪ [1-ε, 1+max_err+ε]
        // Simplified: we use conservative bounds
        double max_interval = 1.0 + epsilon;
        // double min_interval = 1.0 - epsilon;  // Reserved for future use

        std::function<double(double)> func;

        if (!is_last_stage) {
            // Intermediate: sign(x) ∈ {-1, 1}
            func = [](double x) -> double {
                return (x < 0) ? -1.0 : 1.0;
            };
        } else {
            // Final stage: map to [0, 1] for ReLU
            // (sign(x) + 1) / 2 ∈ {0, 1}
            func = [](double x) -> double {
                return (x < 0) ? 0.0 : 1.0;
            };
        }
        
        // Generate Chebyshev approximation on [-1, 1]
        std::vector<double> coeffs = EvalChebyshevCoefficients(func, -1.0, 1.0, degree);

        // Divide coefficients by max_interval to scale to actual working range
        if (stage < degrees.size() - 1) {
            for (auto& coeff : coeffs) {
                coeff /= max_interval;
            }
        }

        allCoeffs.push_back(coeffs);
    }

    return allCoeffs;
}

Ciphertext<DCRTPoly> EvalMiniMaxSign(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ct,
    const std::vector<std::vector<double>>& signCoeffs,
    double lowerBound, double upperBound) {
    
    // Step 1: Prescale input if needed (update to use lower/upper bounds)
    double absmax = std::max(std::abs(lowerBound), std::abs(upperBound)) * 1.05;
    auto ctScaled = ct;
    if (absmax > 1.0) {
        ctScaled = cc->EvalMult(ct, 1.0/absmax);
    }
    
    // Step 2: Evaluate sign(x) using composite polynomials
    auto ctSign = ctScaled;
    for (const auto& stageCoeffs : signCoeffs) {
        ctSign = cc->EvalChebyshevSeries(ctSign, stageCoeffs, -1.0, 1.0);
    }

    // Step 3: ReLU(x) ≈ x · sign(x)
    auto ctResult = cc->EvalMult(ctScaled, ctSign);
    cc->ModReduceInPlace(ctResult);
    
    // Step 4: Post-scale if needed
    if (absmax > 1.0) {
        ctResult = cc->EvalMult(ctResult, absmax);
    }
    
    return ctResult;
}