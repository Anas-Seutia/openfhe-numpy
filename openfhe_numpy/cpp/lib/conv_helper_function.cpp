#include "conv_helper_function.h"
#include "numpy_utils.h"
#include "utils/exception.h"

#include <openfhe.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <map>

template <typename T>
std::vector<std::vector<T>> PackMatDiagWise(
    const std::vector<std::vector<T>> &matrix,
    const std::size_t &num_slots) {
    
    // Check input parameters
    uint32_t matrix_height = matrix.size();
    uint32_t matrix_width = matrix.empty() ? 0 : matrix[0].size();

    std::vector<std::vector<T>> diagonals;
    
    if (!IsPowerOfTwo(num_slots)) {
        OPENFHE_THROW("NumSlots must be a power of two");
    }
    
    // Check size constraints
    if (num_slots < matrix_width) {
        OPENFHE_THROW("size is bigger than total slots");
    }
    
    if (matrix_height == 1) {
        std::vector<std::vector<T>> vector = {matrix[0]};
        return vector;
    }
    
    for (uint32_t diag_idx = 0; diag_idx < matrix_width; ++diag_idx) {
        std::vector<T> diagonal(num_slots, 0.0);
        
        // Extract diagonal values
        for (uint32_t n = 0; n < matrix_height; ++n) {
            // Compute row and column indices with wrapping
            uint32_t row = n;
            uint32_t col = (diag_idx + n) % matrix_width;
            
            if (row < matrix_height && col < matrix_width) {
                diagonal[n] = matrix[row][col];
            }
        }
        
        diagonals.push_back(diagonal);
    }

    return diagonals;
}
template std::vector<std::vector<double>> PackMatDiagWise(const std::vector<std::vector<double>> &matrix, const std::size_t &num_slots);

std::vector<int32_t> getOptimalRots(const std::vector<std::vector<double>> &matrix, bool BSGSmode, uint32_t babyStep) {
    // Check input parameters
    
    uint32_t matrix_height = matrix.size();
    uint32_t matrix_width = matrix.empty() ? 0 : matrix[0].size();
    std::vector<int32_t> rotations;
    
    switch (BSGSmode) {
        case true: {
            if (babyStep == 0) {
                babyStep = ceil(sqrt(matrix_height));
            } 
            for (uint32_t diag_idx = 0; diag_idx < babyStep; ++diag_idx) {
                // std::cout << diag_idx << std::endl;
                rotations.push_back(diag_idx);
            }
            for (uint32_t diag_idx = babyStep; diag_idx < matrix_height; diag_idx+=babyStep) {
                // std::cout << diag_idx << std::endl;
                rotations.push_back(diag_idx);
            }
            break;
        }
        case false: {
            for (uint32_t diag_idx = 0; diag_idx < matrix_height; ++diag_idx) {
                bool is_nonzero = false;
                for (uint32_t n = 0; n < matrix_width; ++n) {
                    if (std::abs(matrix[diag_idx][n]) > 1e-10) {
                        is_nonzero = true;
                        break;
                    }
                }
                // Only store non-zero diagonals
                if (is_nonzero) {
                    rotations.push_back(diag_idx);
                }
            }
            break;
        }
    }

    // std::cout << (int32_t)matrix_height * -1 << std::endl;

    rotations.push_back((int32_t)matrix_height * -1);

    return rotations;
}

template <typename T>
DiagonalResult<T> PackBlockMatDiagWise(
    std::vector<std::vector<T>> matrix,
    const std::size_t &block_width,
    const std::string& embed_method,
    const std::size_t &num_slots) {
    
    // Check input parameters
    uint32_t matrix_height = matrix.size();
    uint32_t matrix_width = matrix.empty() ? 0 : matrix[0].size();
    
    DiagonalResult<T> result;
    
    // Check power of two constraints
    if (!IsPowerOfTwo(block_width)) {
        OPENFHE_THROW("BlockSize must be a power of two");
    }
    
    if (!IsPowerOfTwo(num_slots)) {
        OPENFHE_THROW("NumSlots must be a power of two");
    }
    
    // Check size constraints
    if (num_slots < matrix_height * matrix_width) {
        OPENFHE_THROW("size is bigger than total slots");
    }
    
    if (matrix_height == 1) {
        if (num_slots / block_width > 1) {
            OPENFHE_THROW("vector is too long, can't duplicate");
        }
        result.diagonals_by_block[{0,0}][0] = matrix[0];
        result.output_rotations = 0;
        return result;
    }
    
    if (num_slots % (block_width * block_width) != 0) {
        OPENFHE_THROW("num_slots % block_size must equal 0");
    }
    
    uint32_t num_block_rows = std::ceil((double)matrix_height / block_width);
    uint32_t num_block_cols = std::ceil((double)matrix_width / block_width);
    
    
    // Determine block height and output rotations
    uint32_t block_height;
    if (num_block_rows == 1 && embed_method == "hybrid") {
        block_height = NextPow2(matrix_height);
        result.output_rotations = static_cast<uint32_t>(std::log2(block_width / block_height));
    } else {
        block_height = block_width;
        result.output_rotations = 0;
    }

    // Inflate dimensions of the matrix (resize)
    uint32_t new_height = num_block_rows * block_height;
    uint32_t new_width = num_block_cols * block_width;

    // Copy original matrix data
    std::vector<std::vector<T>> resized_matrix(new_height, std::vector<T>(new_width, 0.0));
    if (new_height > matrix_height || new_width > matrix_width) {
        for (uint32_t i = 0; i < matrix_height; ++i) {
            for (uint32_t j = 0; j < matrix_width; ++j) {
                resized_matrix[i][j] = matrix[i][j];
            }
        }
        matrix = std::move(resized_matrix);
    }


    // Process each block
    uint32_t total_diagonals = 0;

    for (uint32_t block_row = 0; block_row < num_block_rows; ++block_row) {
        for (uint32_t block_col = 0; block_col < num_block_cols; ++block_col) {
            uint32_t row_start = block_height * block_row;
            uint32_t col_start = block_width * block_col;
            
            // Extract a block from the matrix
            std::vector<std::vector<T>> block(block_height, std::vector<T>(block_width, 0.0));
            for (uint32_t i = 0; i < block_height; ++i) {
                for (uint32_t j = 0; j < block_width; ++j) {
                    uint32_t row = row_start + i;
                    uint32_t col = col_start + j;
                    
                    if (row < matrix.size() && col < matrix[0].size()) {
                        block[i][j] = matrix[row][col];
                    }
                }
            }
            
            
            // Extract generalized diagonals from a block (row)
            std::vector<uint32_t> row_idx;
            uint32_t repeat_count = block_width / block_height;
            for (uint32_t rep = 0; rep < repeat_count; ++rep) {
                for (uint32_t i = 0; i < block_height; ++i) {
                    row_idx.push_back(i);
                }
            }
            
            // Extract generalized diagonals from a block (col)
            std::vector<std::vector<uint32_t>> col_idx(block_height, std::vector<uint32_t>(block_width));
            for (uint32_t i = 0; i < block_height; ++i) {
                for (uint32_t j = 0; j < block_width; ++j) {
                    uint32_t idx = i + j;
                    col_idx[i][j] = (idx >= block_width) ? (idx - block_width) : idx;
                }
            }
            
            // Extract diagonals using the computed indices
            std::vector<std::vector<T>> block_diagonals(block_height, std::vector<T>(block_width));
            for (uint32_t i = 0; i < block_height; ++i) {
                for (uint32_t j = 0; j < block_width; ++j) {
                    uint32_t row = row_idx[j];
                    uint32_t col = col_idx[i][j];
                    
                    if (row < block.size() && col < block[0].size()) {
                        block_diagonals[i][j] = block[row][col];
                    } else {
                        block_diagonals[i][j] = 0.0;
                    }
                }
            }
            
            // Collect non-zero diagonals
            std::map<uint32_t, std::vector<T>> nonzero_diagonals;
            for (uint32_t i = 0; i < block_height; ++i) {
                bool is_nonzero = false;
                for (uint32_t j = 0; j < block_width; ++j) {
                    if (std::abs(block_diagonals[i][j]) > 1e-10) {
                        is_nonzero = true;
                        break;
                    }
                }
                
                if (is_nonzero) {
                    nonzero_diagonals[i] = block_diagonals[i];
                }
            }
            
            // If no non-zero diagonals, add a zero diagonal
            if (nonzero_diagonals.empty()) {
                nonzero_diagonals[0] = std::vector<T>(block_width, 0.0);
            }
            
            total_diagonals += nonzero_diagonals.size();
            
            // Store in result
            for (const auto& i : nonzero_diagonals) {
                result.diagonals_by_block[{block_row, block_col}][i.first] = i.second;
            }
        }
    }

    return result;
}
template DiagonalResult<double> PackBlockMatDiagWise(std::vector<std::vector<double>> matrix, const std::size_t &block_size, const std::string& embed_method, const std::size_t &num_slots);

template <typename T>
std::vector<std::vector<T>> ConstructConv2DToeplitz(
    const std::vector<std::vector<std::vector<std::vector<T>>>>& kernel,
    const uint32_t &input_height, 
    const uint32_t &input_width,  
    const uint32_t &stride,
    const uint32_t &padding,
    const uint32_t &dilation,
    const uint32_t &input_gap,
    const uint32_t &output_gap) {

    const uint32_t out_channels = kernel.size();
    const uint32_t in_channels = kernel[0].size();
    const uint32_t kernel_height = kernel[0][0].size();
    const uint32_t kernel_width = kernel[0][0][0].size();
    
    // Compute output dimensions after convolution
    uint32_t output_height, output_width;
    output_height = (input_height + 2*padding -
                    dilation*(kernel_height-1) - 1) / stride + 1;
    output_width = (input_width + 2*padding -
                    dilation*(kernel_width-1) - 1) / stride + 1;
    
    // Compute multiplexed input dimensions
    // With input_gap, input channels are packed into spatial dimensions
    uint32_t input_gap_squared = input_gap * input_gap;
    uint32_t super_in_channels = (in_channels + input_gap_squared - 1) / input_gap_squared;  // ceil division
    uint32_t final_input_height = input_height * input_gap;
    uint32_t final_input_width = input_width * input_gap;

    // Padded input dimensions (after multiplexing)
    uint32_t input_height_padding = final_input_height + 2 * padding;
    uint32_t input_width_padding = final_input_width + 2 * padding;

    // Compute multiplexed output dimensions
    // With output_gap, we pack output_gap×output_gap channels into spatial dimensions
    uint32_t output_gap_squared = output_gap * output_gap;
    uint32_t super_channels = (out_channels + output_gap_squared - 1) / output_gap_squared;  // ceil division
    uint32_t final_output_height = output_height * output_gap;
    uint32_t final_output_width = output_width * output_gap;

    // Initialize Toeplitz matrix with multiplexed dimensions
    uint32_t n_rows = super_channels * final_output_height * final_output_width;

    // Using map for sparse representation (row -> col -> value)
    std::map<uint32_t, std::map<uint32_t, T>> sparse_matrix;
    
    // Create index grid for the multiplexed padded input image
    std::vector<std::vector<std::vector<uint32_t>>> valid_image_indices(super_in_channels);
    uint32_t idx = 0;

    for (uint32_t c = 0; c < super_in_channels; ++c) {
        valid_image_indices[c].resize(input_height_padding);
        for (uint32_t h = 0; h < input_height_padding; ++h) {
            valid_image_indices[c][h].resize(input_width_padding);
            for (uint32_t w = 0; w < input_width_padding; ++w) {
                valid_image_indices[c][h][w] = idx++;
            }
        }
    }
    
    // Flatten kernel for easier access
    std::vector<std::vector<T>> flat_kernel(
        out_channels,
        std::vector<T>(in_channels * kernel_height * kernel_width, 0.0)
    );

    // Copy kernel values
    for (uint32_t co = 0; co < out_channels; ++co) {
        for (uint32_t ci = 0; ci < in_channels; ++ci) {
            for (uint32_t kh = 0; kh < kernel_height; ++kh) {
                for (uint32_t kw = 0; kw < kernel_width; ++kw) {
                    uint32_t flat_idx = ci * kernel_height * kernel_width +
                                       kh * kernel_width + kw;
                    flat_kernel[co][flat_idx] = kernel[co][ci][kh][kw];
                }
            }
        }
    }
    
    // Compute initial kernel position (which input pixels the kernel initially touches)
    // Map from logical input channels to multiplexed physical positions
    std::vector<uint32_t> initial_kernel_position;

    // For each logical input channel, compute kernel positions
    for (uint32_t ci = 0; ci < in_channels; ++ci) {
        // Map logical channel to multiplexed position
        uint32_t super_ch = ci / input_gap_squared;
        uint32_t in_block = ci % input_gap_squared;
        uint32_t block_h = in_block / input_gap;
        uint32_t block_w = in_block % input_gap;

        // For each kernel position, compute the physical column index
        for (uint32_t kh = 0; kh < kernel_height; ++kh) {
            for (uint32_t kw = 0; kw < kernel_width; ++kw) {
                // Logical position with dilation
                uint32_t logical_h = kh * dilation;
                uint32_t logical_w = kw * dilation;

                // Physical position in multiplexed layout
                uint32_t physical_h = logical_h * input_gap + block_h;
                uint32_t physical_w = logical_w * input_gap + block_w;

                initial_kernel_position.push_back(valid_image_indices[super_ch][physical_h][physical_w]);
            }
        }
    }
    
    // Compute input starting positions for each output position
    // Map logical output positions to physical input positions in multiplexed layout
    std::vector<uint32_t> input_start_indices;
    for (uint32_t h = 0; h < output_height; ++h) {
        for (uint32_t w = 0; w < output_width; ++w) {
            // Logical input position for this output
            // Physical position in multiplexed layout (channel 0, block position 0,0)
            uint32_t physical_h = h * stride * input_gap;
            uint32_t physical_w = w * stride * input_gap;
            input_start_indices.push_back(valid_image_indices[0][physical_h][physical_w]);
        }
    }
    
    // Populate the Toeplitz matrix
    uint32_t pos_idx = 0;
    for (uint32_t h = 0; h < output_height; ++h) {
        for (uint32_t w = 0; w < output_width; ++w) {
            uint32_t start_idx = input_start_indices[pos_idx];

            for (uint32_t co = 0; co < out_channels; ++co) {
                // Compute multiplexed row index
                // Determine which super-channel this output channel belongs to
                uint32_t super_ch = co / output_gap_squared;
                uint32_t in_block = co % output_gap_squared;

                // Position within the output_gap × output_gap channel block
                uint32_t block_h = in_block / output_gap;
                uint32_t block_w = in_block % output_gap;

                // Final multiplexed position
                uint32_t final_h = h * output_gap + block_h;
                uint32_t final_w = w * output_gap + block_w;
                uint32_t row = super_ch * final_output_height * final_output_width +
                               final_h * final_output_width + final_w;

                for (size_t k = 0; k < initial_kernel_position.size(); ++k) {
                    uint32_t col = initial_kernel_position[k] + start_idx;
                    T value = flat_kernel[co][k];

                    if (std::abs(value) > 1e-10) {  // Only store non-zero values
                        sparse_matrix[row][col] = value;
                    }
                }
            }
            pos_idx++;
        }
    }
    
    // Remove padding columns (keep only non-padded input positions in multiplexed layout)
    std::vector<uint32_t> image_indices_flat;
    for (uint32_t c = 0; c < super_in_channels; ++c) {
        for (uint32_t h = padding; h < padding + final_input_height; ++h) {
            for (uint32_t w = padding; w < padding + final_input_width; ++w) {
                image_indices_flat.push_back(valid_image_indices[c][h][w]);
            }
        }
    }

    // Build final dense matrix with unpadded columns (multiplexed dimensions)
    uint32_t final_n_cols = super_in_channels * final_input_height * final_input_width;
    std::vector<std::vector<T>> toeplitz(n_rows, std::vector<T>(final_n_cols, 0.0));
    
    for (const auto& row_entry : sparse_matrix) {
        uint32_t row = row_entry.first;
        for (const auto& col_entry : row_entry.second) {
            uint32_t original_col = col_entry.first;
            T value = col_entry.second;
            
            // Find new column index in unpadded matrix
            auto it = std::find(image_indices_flat.begin(), image_indices_flat.end(), original_col);
            if (it != image_indices_flat.end()) {
                uint32_t new_col = std::distance(image_indices_flat.begin(), it);
                toeplitz[row][new_col] = value;
            }
        }
    }
    
    return toeplitz;
}
template std::vector<std::vector<double>> ConstructConv2DToeplitz(
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
    const uint32_t &input_height,
    const uint32_t &input_width,
    const uint32_t &stride,
    const uint32_t &padding,
    const uint32_t &dilation,
    const uint32_t &input_gap,
    const uint32_t &output_gap);

template <typename T>
std::vector<std::vector<T>> MultiplexDenseMatrix(
    const std::vector<std::vector<T>>& matrix,
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t input_gap,
    const uint32_t output_height,
    const uint32_t output_width,
    const uint32_t output_gap) {

    uint32_t out_features = matrix.size();
    uint32_t in_features = matrix.empty() ? 0 : matrix[0].size();

    // Derive channel counts from features and spatial dimensions
    uint32_t in_channels = in_features / (input_height * input_width);
    uint32_t out_channels = out_features / (output_height * output_width);

    // Compute multiplexed dimensions
    uint32_t input_gap_squared = input_gap * input_gap;
    uint32_t output_gap_squared = output_gap * output_gap;

    uint32_t super_in_channels = (in_channels + input_gap_squared - 1) / input_gap_squared;
    uint32_t super_out_channels = (out_channels + output_gap_squared - 1) / output_gap_squared;

    uint32_t final_input_height = input_height * input_gap;
    uint32_t final_input_width = input_width * input_gap;
    uint32_t final_output_height = output_height * output_gap;
    uint32_t final_output_width = output_width * output_gap;

    // Create column mapping: logical input index -> multiplexed input index
    std::vector<uint32_t> col_mapping(in_features);

    for (uint32_t ci = 0; ci < in_channels; ++ci) {
        for (uint32_t h = 0; h < input_height; ++h) {
            for (uint32_t w = 0; w < input_width; ++w) {
                // Logical flat index (channel-major order)
                uint32_t logical_idx = ci * input_height * input_width + h * input_width + w;

                // Map to multiplexed position
                uint32_t super_ch = ci / input_gap_squared;
                uint32_t in_block = ci % input_gap_squared;
                uint32_t block_h = in_block / input_gap;
                uint32_t block_w = in_block % input_gap;

                uint32_t physical_h = h * input_gap + block_h;
                uint32_t physical_w = w * input_gap + block_w;

                // Multiplexed flat index (channel-major order)
                uint32_t multiplexed_idx = super_ch * final_input_height * final_input_width +
                                          physical_h * final_input_width + physical_w;

                col_mapping[logical_idx] = multiplexed_idx;
            }
        }
    }

    // Create row mapping: logical output index -> multiplexed output index
    std::vector<uint32_t> row_mapping(out_features);

    for (uint32_t co = 0; co < out_channels; ++co) {
        for (uint32_t h = 0; h < output_height; ++h) {
            for (uint32_t w = 0; w < output_width; ++w) {
                // Logical flat index (channel-major order)
                uint32_t logical_idx = co * output_height * output_width + h * output_width + w;

                // Map to multiplexed position
                uint32_t super_ch = co / output_gap_squared;
                uint32_t in_block = co % output_gap_squared;
                uint32_t block_h = in_block / output_gap;
                uint32_t block_w = in_block % output_gap;

                uint32_t physical_h = h * output_gap + block_h;
                uint32_t physical_w = w * output_gap + block_w;

                // Multiplexed flat index (channel-major order)
                uint32_t multiplexed_idx = super_ch * final_output_height * final_output_width +
                                          physical_h * final_output_width + physical_w;

                row_mapping[logical_idx] = multiplexed_idx;
            }
        }
    }

    // Create multiplexed matrix with reordered rows and columns
    uint32_t multiplexed_out_features = super_out_channels * final_output_height * final_output_width;
    uint32_t multiplexed_in_features = super_in_channels * final_input_height * final_input_width;

    std::vector<std::vector<T>> multiplexed_matrix(
        multiplexed_out_features,
        std::vector<T>(multiplexed_in_features, 0.0)
    );

    // Reorder the matrix
    for (uint32_t row = 0; row < out_features; ++row) {
        for (uint32_t col = 0; col < in_features; ++col) {
            uint32_t new_row = row_mapping[row];
            uint32_t new_col = col_mapping[col];
            multiplexed_matrix[new_row][new_col] = matrix[row][col];
        }
    }

    return multiplexed_matrix;
}
template std::vector<std::vector<double>> MultiplexDenseMatrix(
    const std::vector<std::vector<double>>& matrix,
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t input_gap,
    const uint32_t output_height,
    const uint32_t output_width,
    const uint32_t output_gap);

template <typename T>

Ciphertext<DCRTPoly> EvalMultMatVecDiag(const Ciphertext<DCRTPoly>& ctVector,
                                        const std::vector<T>& diagonals,
                                        uint32_t hoistingMode,
                                        std::vector<int32_t>& rotations,
                                        uint32_t babyStep) { 

    if (rotations.empty()) {
        rotations.reserve(diagonals.size());
        for (size_t k = 0; k < diagonals.size(); ++k) {
            rotations.push_back(static_cast<int32_t>(k));
        }
    }

    auto cryptoContext = ctVector->GetCryptoContext();
    Ciphertext<DCRTPoly> ctResult;
    Ciphertext<DCRTPoly> ctRotated;
    Ciphertext<DCRTPoly> ctProduct;
    bool first = true;

    switch (hoistingMode) {
        case 0: {
            for (const int32_t rotation : rotations) {     
                ctRotated = cryptoContext->EvalRotate(ctVector, static_cast<int32_t>(rotation));
                ctProduct = cryptoContext->EvalMult(ctRotated, diagonals[rotation]);
                if (first) {
                    ctResult = ctProduct;
                    first = false;
                } else {
                    cryptoContext->EvalAddInPlace(ctResult, ctProduct);
                }
            }
            break;
        }
        case 1: {
            uint32_t M = 2 * cryptoContext->GetRingDimension();
            auto precomp = cryptoContext->EvalFastRotationPrecompute(ctVector);
            for (const int32_t rotation : rotations) {
                if (rotation < 0) break;
                ctRotated = cryptoContext->EvalFastRotation(ctVector, rotation, M, precomp);
                ctProduct = cryptoContext->EvalMult(ctRotated, diagonals[rotation]);
                if (first) {
                    ctResult = ctProduct;
                    first = false;
                } else {
                    cryptoContext->EvalAddInPlace(ctResult, ctProduct);
                }
            }
            break;
        }
        case 2: {
            if (babyStep == 0) {
                babyStep = static_cast<uint32_t>(ceil(sqrt(diagonals.size())));
            }
            uint32_t giantStep = static_cast<uint32_t>(ceil(static_cast<double>(diagonals.size()) / babyStep));

            uint32_t M = 2 * cryptoContext->GetRingDimension();
            auto digits = cryptoContext->EvalFastRotationPrecompute(ctVector);

            std::vector<Ciphertext<DCRTPoly>> fastRotation(babyStep);
            #pragma omp parallel for
            for (uint32_t j = 0; j < babyStep; j++) {
                // std::cout << j << std::endl;
                fastRotation[j] = cryptoContext->EvalFastRotation(ctVector, j, M, digits);
            }

            for (uint32_t i = 0; i < giantStep; i++) {
                Ciphertext<DCRTPoly> inner;

                for (uint32_t j = 0; j < babyStep; j++) {
                    uint32_t idx = i * babyStep + j;
                    if (idx >= diagonals.size()) break;  // Handle incomplete last giant step
                    
                    // remove for after pre rotation diagonals by -babyStep * i
                    int32_t jdx = static_cast<int32_t>(babyStep * i * -1);
                    Ciphertext<DCRTPoly> ctProduct;
                    if constexpr (std::is_same<T, Ciphertext<DCRTPoly>>::value) {
                        auto preRotated = cryptoContext->EvalRotate(diagonals[idx], jdx);
                        ctProduct = cryptoContext->EvalMult(fastRotation[j], preRotated);
                    } else {
                        std::vector<std::complex<double>> vecDiag = diagonals[idx]->GetCKKSPackedValue();
                        auto rotatedVec = lbcrypto::Rotate(vecDiag, jdx);
                        auto preRotated = cryptoContext->MakeCKKSPackedPlaintext(rotatedVec);
                        ctProduct = cryptoContext->EvalMult(fastRotation[j], preRotated);
                    }
                    
                    if (j == 0) {
                        inner = ctProduct;
                    } else {
                        cryptoContext->EvalAddInPlace(inner, ctProduct);
                    }
                }

                // Step 2.5: Apply giant-step rotation with SECOND HOISTING
                auto innerDigits = cryptoContext->EvalFastRotationPrecompute(inner);
                if (i == 0) {
                    ctResult = cryptoContext->EvalFastRotation(inner, 0, M, innerDigits);
                } else {
                    // std::cout << i * babyStep << std::endl;
                    auto rotated = cryptoContext->EvalFastRotation(inner, i * babyStep, M, innerDigits);
                    cryptoContext->EvalAddInPlace(ctResult, rotated);
                }
            }
            break;
        }
        default:
            std::cerr << "invalid hoistingMode" << std::endl;
    }
    return ctResult;
}
template Ciphertext<DCRTPoly> EvalMultMatVecDiag(const Ciphertext<DCRTPoly>& ctVector, const std::vector<Ciphertext<DCRTPoly>>& diagonals, uint32_t hoistingMode, std::vector<int32_t>& rotations, uint32_t babyStep);
template Ciphertext<DCRTPoly> EvalMultMatVecDiag(const Ciphertext<DCRTPoly>& ctVector, const std::vector<Plaintext>& diagonals, uint32_t hoistingMode, std::vector<int32_t>& rotations, uint32_t babyStep);

std::vector<Plaintext> MakeCKKSPackedPlaintextVectors(const CryptoContextCKKSRNS::ContextType &cc, const std::vector<std::vector<double>>& vectors) {
    std::vector<Plaintext> ctVectors;
    for (uint32_t i = 0; i < vectors.size(); i++) {
        ctVectors.push_back(cc->MakeCKKSPackedPlaintext(vectors[i]));
    }
    return ctVectors;
}

std::vector<Ciphertext<DCRTPoly>> EncryptVectors(const CryptoContextCKKSRNS::ContextType &cc, const PublicKey<DCRTPoly>& publicKey, const std::vector<Plaintext>& vectors) {
    std::vector<Ciphertext<DCRTPoly>> ctVectors;
    for (uint32_t i = 0; i < vectors.size(); i++) {
        ctVectors.push_back(cc->Encrypt(publicKey, vectors[i]));
    }
    return ctVectors;
}