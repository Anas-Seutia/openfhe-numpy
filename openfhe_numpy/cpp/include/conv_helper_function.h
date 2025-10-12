#ifndef __CONV_HELPER_FUNCTION_H__
#define __CONV_HELPER_FUNCTION_H__

#include <cstdint>
#include <vector>

#include "lattice/lat-hal.h"

/**
 * @brief Diagonalize a matrix for FHE operations
 * 
 * @param matrix Input matrix (H x W)
 * @param num_slots Total number of slots to fill (must be power of two)
 * @return Toeplitz matrix representation as 2D vector
 */
template <typename T>
std::map<uint32_t, std::vector<T>> PackMatDiagWise(
    const std::vector<std::vector<T>> &matrix,
    const std::size_t &num_slots
);

/**
 * @brief Structure to hold diagonalization results
 */
template <typename T>
struct DiagonalResult {
    // Map of (block_row, block_col) -> {diagonal_index -> diagonal_values}
    std::map<std::pair<uint32_t, uint32_t>, std::map<uint32_t, std::vector<T>>> diagonals_by_block;
    uint32_t output_rotations;
};

/**
 * @brief Diagonalize a blocked matrix for FHE operations
 * 
 * @param matrix Input matrix (H x W)
 * @param block_width Width size of SIMD plaintext blocks
 * @param embed_method Embedding method ["hybrid"(plaintext), "standard" (ciphertext)], use "standard" for fully connected
 * @param num_slots Total number of slots to fill (must be power of two)
 * @return Toeplitz matrix representation as 2D vector
 */
template <typename T>
DiagonalResult<T> PackBlockMatDiagWise(
    std::vector<std::vector<T>> matrix,
    const std::size_t &block_width,
    const std::string& embed_method,
    const std::size_t &num_slots
);

/**
 * @brief Construct Toeplitz matrix for 2D convolution
 * 
 * @param kernel Input kernel (Co x Ci x kH x kW)
 * @param input_height Input height (Hi)
 * @param input_width Input width (Wi)
 * @param stride Stride (assumes square stride)
 * @param padding Padding (assumes square padding)
 * @param dilation Dilation (assumes square dilation)
 * @param batch_size Batch size (N)
 * @param input_gap Multiplexing gap for input (iG)
 * @param output_gap Multiplexing gap for output (oG)
 * @return Toeplitz matrix representation as 2D vector
 */
template <typename T>
std::vector<std::vector<T>> ConstructConv2DToeplitz(
    const std::vector<std::vector<std::vector<std::vector<T>>>>& kernel,
    const uint32_t &input_height,
    const uint32_t &input_width,
    const uint32_t &stride,
    const uint32_t &padding,
    const uint32_t &dilation,
    const uint32_t &batch_size,
    const uint32_t &input_gap,
    const uint32_t &output_gap
);

#endif  // __CONV_HELPER_FUNCTION_H__