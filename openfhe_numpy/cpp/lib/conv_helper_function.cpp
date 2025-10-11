#include "conv_helper_function.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <map> 

std::vector<std::vector<double>> ConstructConv2DToeplitz(
    const std::vector<std::vector<std::vector<std::vector<double>>>>& kernel,
    const uint32_t &in_channels,  
    const uint32_t &out_channels, 
    const uint32_t &input_height, 
    const uint32_t &input_width,  
    const uint32_t &kernel_height,
    const uint32_t &kernel_width, 
    const uint32_t &stride,
    const uint32_t &padding,
    const uint32_t &dilation,
    const uint32_t &batch_size,
    const uint32_t &input_gap,
    const uint32_t &output_gap) {
    
    std::cout << "\n=== Constructing Toeplitz Matrix ===" << std::endl;
    
    // Compute output dimensions after convolution
    uint32_t output_height, output_width;
    output_height = (input_height + 2*padding - 
                    dilation*(kernel_height-1) - 1) / stride + 1;
    output_width = (input_width + 2*padding - 
                    dilation*(kernel_width-1) - 1) / stride + 1;
    std::cout << "Output dimensions (output_height, output_width): (" << output_height << ", " << output_width << ")" << std::endl;
    
    // Compute FHE output dimensions with multiplexing
    uint32_t ctx_input_channel = std::ceil((double)in_channels / 
                               (input_gap * input_gap));
    uint32_t ctx_output_channel, ctx_output_height, ctx_output_width;
    ctx_output_channel = std::ceil((double)out_channels / (output_gap * output_gap));
    ctx_output_height = std::max(input_height, output_height * output_gap);
    ctx_output_width = std::max(input_width, output_width * output_gap);
        
    std::cout << "FHE input shape: (N=" << batch_size 
              << ", Ci=" << ctx_input_channel 
              << ", Hi=" << input_height * input_gap
              << ", Wi=" << input_width * input_gap << ")" << std::endl;
    std::cout << "FHE output shape: (N=" << batch_size 
              << ", Co=" << ctx_output_channel << ", output_height=" << ctx_output_height << ", output_width=" << ctx_output_width << ")" << std::endl;
    
    // Padded input dimensions
    uint32_t input_height_padding = input_height * input_gap + 
                      2 * padding * input_gap;
    uint32_t input_width_padding = input_width * input_gap + 
                      2 * padding * input_gap;
    
    // Initialize Toeplitz matrix
    uint32_t n_rows = ctx_output_channel * ctx_output_height * ctx_output_width;
    uint32_t n_cols = ctx_input_channel * input_height_padding * input_width_padding;
    
    std::cout << "Toeplitz matrix dimensions: " << n_rows << " x " << n_cols << std::endl;
    
    // Using map for sparse representation (row -> col -> value)
    std::map<uint32_t, std::map<uint32_t, double>> sparse_matrix;
    
    // Create index grid for the padded input image
    std::vector<std::vector<std::vector<uint32_t>>> valid_image_indices(ctx_input_channel);
    uint32_t idx = 0;
        
    for (uint32_t c = 0; c < ctx_input_channel; ++c) {
        valid_image_indices[c].resize(input_height_padding);
        for (uint32_t h = 0; h < input_height_padding; ++h) {
            valid_image_indices[c][h].resize(input_width_padding);
            for (uint32_t w = 0; w < input_width_padding; ++w) {
                valid_image_indices[c][h][w] = idx++;
            }
        }
    }
    
    // Pad kernel to match multiplexing requirements
    uint32_t padded_out_channels = ctx_output_channel * output_gap * output_gap;
    uint32_t padded_in_channels = ctx_input_channel * input_gap * input_gap;
    
    std::vector<std::vector<double>> padded_kernel(
        padded_out_channels,
        std::vector<double>(padded_in_channels * kernel_height * kernel_width, 0.0)
    );
    
    // Copy kernel values
    for (uint32_t co = 0; co < out_channels; ++co) {
        for (uint32_t ci = 0; ci < in_channels; ++ci) {
            for (uint32_t kh = 0; kh < kernel_height; ++kh) {
                for (uint32_t kw = 0; kw < kernel_width; ++kw) {
                    uint32_t flat_idx = ci * kernel_height * kernel_width + 
                                       kh * kernel_width + kw;
                    padded_kernel[co][flat_idx] = kernel[co][ci][kh][kw];
                }
            }
        }
    }
    
    // Compute initial kernel position (which input pixels the kernel initially touches)
    std::vector<uint32_t> initial_kernel_position;
    
    // Extract multiplex anchors (top-left iG x iG region)
    for (uint32_t h = 0; h < input_gap; ++h) {
        for (uint32_t w = 0; w < input_gap; ++w) {
            uint32_t anchor = valid_image_indices[0][h][w];
            
            // For each anchor, compute all kernel offsets
            for (uint32_t kh = 0; kh < kernel_height; ++kh) {
                for (uint32_t kw = 0; kw < kernel_width; ++kw) {
                    uint32_t row_idx = kh * dilation * input_gap;
                    uint32_t col_idx = kw * dilation * input_gap;
                    uint32_t offset = valid_image_indices[0][row_idx][col_idx];
                    initial_kernel_position.push_back(anchor + offset);
                }
            }
        }
    }
    
    // Compute row interchange map for optimal packing     
    std::vector<std::vector<uint32_t>> row_map;
    
    // Create output indices grid
    std::vector<std::vector<uint32_t>> output_indices(ctx_output_height, std::vector<uint32_t>(ctx_output_width));
    idx = 0;
    for (uint32_t h = 0; h < ctx_output_height; ++h) {
        for (uint32_t w = 0; w < ctx_output_width; ++w) {
            output_indices[h][w] = idx++;
        }
    }
    
    // Extract start indices (top-left output_gap x output_gap region)
    std::vector<uint32_t> start_indices;
    for (uint32_t h = 0; h < output_gap; ++h) {
        for (uint32_t w = 0; w < output_gap; ++w) {
            start_indices.push_back(output_indices[h][w]);
        }
    }
    
    // Extract corner indices (strided positions)
    for (uint32_t h = 0; h < output_height * output_gap; h += output_gap) {
        for (uint32_t w = 0; w < output_width * output_gap; w += output_gap) {
            std::vector<uint32_t> positions;
            uint32_t corner = output_indices[h][w];
            for (uint32_t start : start_indices) {
                positions.push_back(corner + start);
            }
            row_map.push_back(positions);
        }
    }
    
    // Compute corner indices for iteration
    std::vector<uint32_t> corner_indices;  
    for (uint32_t h = 0; h < output_height * output_gap; h += output_gap) {
        for (uint32_t w = 0; w < output_width * output_gap; w += output_gap) {
            corner_indices.push_back(valid_image_indices[0][h][w]);
        }
    }
    
    // Create output channel offsets
    std::vector<uint32_t> out_channel_offsets(ctx_output_channel);
    for (uint32_t co = 0; co < ctx_output_channel; ++co) {
        out_channel_offsets[co] = co * ctx_output_height * ctx_output_width;
    }
    
    std::cout << "Populating Toeplitz matrix..." << std::endl;
    
    // Populate the Toeplitz matrix
    for (size_t i = 0; i < corner_indices.size(); ++i) {
        uint32_t start_idx = corner_indices[i];
        
        for (uint32_t co = 0; co < ctx_output_channel; ++co) {
            for (size_t j = 0; j < row_map[i].size(); ++j) {
                uint32_t row = row_map[i][j] + out_channel_offsets[co];
                
                for (size_t k = 0; k < initial_kernel_position.size(); ++k) {
                    uint32_t col = initial_kernel_position[k] + start_idx;
                    double value = padded_kernel[co * output_gap * output_gap + j][k];
                    
                    if (std::abs(value) > 1e-10) {  // Only store non-zero values
                        sparse_matrix[row][col] = value;
                    }
                }
            }
        }
    }
    
    // Remove padding columns (keep only non-padded input positions)
    std::vector<uint32_t> image_indices_flat;
    for (uint32_t c = 0; c < ctx_input_channel; ++c) {
        for (uint32_t h = padding * input_gap; 
             h < padding * input_gap + input_height * input_gap; ++h) {
            for (uint32_t w = padding * input_gap; 
                 w < padding * input_gap + input_width * input_gap; ++w) {
                image_indices_flat.push_back(valid_image_indices[c][h][w]);
            }
        }
    }
    
    // Build final dense matrix with unpadded columns
    uint32_t final_n_cols = ctx_input_channel * input_height * input_gap * 
                            input_width * input_gap;
    std::vector<std::vector<double>> toeplitz(n_rows, std::vector<double>(final_n_cols, 0.0));
    
    std::cout << "Converting to dense matrix..." << std::endl;
    
    for (const auto& row_entry : sparse_matrix) {
        uint32_t row = row_entry.first;
        for (const auto& col_entry : row_entry.second) {
            uint32_t original_col = col_entry.first;
            double value = col_entry.second;
            
            // Find new column index in unpadded matrix
            auto it = std::find(image_indices_flat.begin(), image_indices_flat.end(), original_col);
            if (it != image_indices_flat.end()) {
                uint32_t new_col = std::distance(image_indices_flat.begin(), it);
                toeplitz[row][new_col] = value;
            }
        }
    }
    
    std::cout << "Final Toeplitz matrix: " << toeplitz.size() << " x " 
              << (toeplitz.empty() ? 0 : toeplitz[0].size()) << std::endl;
    std::cout << "=== Toeplitz Construction Complete ===" << std::endl;
    
    return toeplitz;
}