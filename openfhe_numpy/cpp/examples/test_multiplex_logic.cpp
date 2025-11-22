#include "conv_helper_function.h"
#include "numpy_utils.h"
#include "numpy_helper_functions.h"
#include <iostream>
#include <vector>
#include <cmath>

// Simple matrix-vector multiplication
std::vector<double> MatVecMul(const std::vector<std::vector<double>>& matrix,
                              const std::vector<double>& vector) {
    std::vector<double> result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

int main() {
    std::cout << "Testing multiplexing logic..." << std::endl;

    // Small test case: 2 channels, 3x3 input, 2x2 kernel
    uint32_t in_channels = 2;
    uint32_t out_channels = 3;
    uint32_t kernel_h = 2, kernel_w = 2;
    uint32_t input_h = 4, input_w = 4;
    uint32_t stride = 1, padding = 0, dilation = 1;

    // Create a simple kernel filled with incremental values
    std::vector<std::vector<std::vector<std::vector<double>>>> kernel(
        out_channels,
        std::vector<std::vector<std::vector<double>>>(
            in_channels,
            std::vector<std::vector<double>>(
                kernel_h,
                std::vector<double>(kernel_w, 0.0)
            )
        )
    );

    double val = 0.1;
    for (uint32_t oc = 0; oc < out_channels; ++oc) {
        for (uint32_t ic = 0; ic < in_channels; ++ic) {
            for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                    kernel[oc][ic][kh][kw] = val;
                    val += 0.1;
                }
            }
        }
    }

    // Create simple input
    std::vector<std::vector<double>> input_2d(input_h, std::vector<double>(input_w, 0.0));
    val = 1.0;
    for (uint32_t h = 0; h < input_h; ++h) {
        for (uint32_t w = 0; w < input_w; ++w) {
            input_2d[h][w] = val;
            val += 1.0;
        }
    }

    // Output dimensions
    uint32_t output_h = (input_h + 2*padding - dilation*(kernel_h-1) - 1) / stride + 1;
    uint32_t output_w = (input_w + 2*padding - dilation*(kernel_w-1) - 1) / stride + 1;

    std::cout << "\nTest configuration:" << std::endl;
    std::cout << "  Input: " << in_channels << " x " << input_h << " x " << input_w << std::endl;
    std::cout << "  Kernel: " << out_channels << " x " << in_channels << " x " << kernel_h << " x " << kernel_w << std::endl;
    std::cout << "  Output (logical): " << out_channels << " x " << output_h << " x " << output_w << std::endl;

    // Test 1: No multiplexing (gap=1)
    std::cout << "\n=== Test 1: No multiplexing (input_gap=1, output_gap=1) ===" << std::endl;
    auto toeplitz_gap1 = ConstructConv2DToeplitz(kernel, input_h, input_w, stride, padding, dilation, 1, 1);
    std::vector<double> input_flat_gap1 = EncodeMatrix(input_2d, in_channels * input_h * input_w);
    std::vector<double> output_gap1 = MatVecMul(toeplitz_gap1, input_flat_gap1);

    std::cout << "  Toeplitz size: " << toeplitz_gap1.size() << " x " << toeplitz_gap1[0].size() << std::endl;
    std::cout << "  Input size: " << input_flat_gap1.size() << std::endl;
    std::cout << "  Output size: " << output_gap1.size() << std::endl;
    std::cout << "  First 5 output values: ";
    for (int i = 0; i < std::min(5, (int)output_gap1.size()); ++i) {
        std::cout << output_gap1[i] << " ";
    }
    std::cout << std::endl;

    // Test 2: With output multiplexing (gap=2)
    std::cout << "\n=== Test 2: With output multiplexing (input_gap=1, output_gap=2) ===" << std::endl;
    auto toeplitz_gap2 = ConstructConv2DToeplitz(kernel, input_h, input_w, stride, padding, dilation, 1, 2);

    // Same input (no input multiplexing)
    std::vector<double> output_gap2_multiplexed = MatVecMul(toeplitz_gap2, input_flat_gap1);

    std::cout << "  Toeplitz size: " << toeplitz_gap2.size() << " x " << toeplitz_gap2[0].size() << std::endl;
    std::cout << "  Output size (multiplexed): " << output_gap2_multiplexed.size() << std::endl;

    // Unmultiplex the output for comparison
    uint32_t output_gap = 2;
    uint32_t output_gap_squared = output_gap * output_gap;
    // uint32_t super_channels = (out_channels + output_gap_squared - 1) / output_gap_squared;
    uint32_t final_output_h = output_h * output_gap;
    uint32_t final_output_w = output_w * output_gap;

    std::vector<double> output_gap2_unmultiplexed(out_channels * output_h * output_w, 0.0);
    for (uint32_t co = 0; co < out_channels; ++co) {
        for (uint32_t h = 0; h < output_h; ++h) {
            for (uint32_t w = 0; w < output_w; ++w) {
                // Logical index
                uint32_t logical_idx = co * output_h * output_w + h * output_w + w;

                // Multiplexed position
                uint32_t super_ch = co / output_gap_squared;
                uint32_t in_block = co % output_gap_squared;
                uint32_t block_h = in_block / output_gap;
                uint32_t block_w = in_block % output_gap;
                uint32_t final_h = h * output_gap + block_h;
                uint32_t final_w = w * output_gap + block_w;
                uint32_t multiplexed_idx = super_ch * final_output_h * final_output_w +
                                          final_h * final_output_w + final_w;

                output_gap2_unmultiplexed[logical_idx] = output_gap2_multiplexed[multiplexed_idx];
            }
        }
    }

    std::cout << "  First 5 output values (after unmultiplexing): ";
    for (int i = 0; i < std::min(5, (int)output_gap2_unmultiplexed.size()); ++i) {
        std::cout << output_gap2_unmultiplexed[i] << " ";
    }
    std::cout << std::endl;

    // Compare outputs
    std::cout << "\n=== Comparison ===" << std::endl;
    double max_error = 0.0;
    double sum_error = 0.0;
    int count = std::min(output_gap1.size(), output_gap2_unmultiplexed.size());

    for (int i = 0; i < count; ++i) {
        double error = std::abs(output_gap1[i] - output_gap2_unmultiplexed[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
    }

    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Avg error: " << sum_error / count << std::endl;

    bool conv_test_passed = (max_error < 1e-6);
    if (conv_test_passed) {
        std::cout << "  ✓ PASS - Multiplexing produces correct results!" << std::endl;
    } else {
        std::cout << "  ✗ FAIL - Multiplexing has errors!" << std::endl;
        std::cout << "\n  Detailed comparison (first 10 values):" << std::endl;
        for (int i = 0; i < std::min(10, count); ++i) {
            std::cout << "    [" << i << "] gap=1: " << output_gap1[i]
                     << ", gap=2: " << output_gap2_unmultiplexed[i]
                     << ", error: " << std::abs(output_gap1[i] - output_gap2_unmultiplexed[i]) << std::endl;
        }
    }

    // Test 3: MultiplexDenseMatrix
    std::cout << "\n=== Test 3: MultiplexDenseMatrix ===" << std::endl;

    // Create a simple dense layer weight matrix
    uint32_t dense_output = 5;
    uint32_t dense_input = out_channels * output_h * output_w;  // 27

    std::vector<std::vector<double>> dense_weights(dense_output, std::vector<double>(dense_input, 0.0));
    val = 0.01;
    for (uint32_t i = 0; i < dense_output; ++i) {
        for (uint32_t j = 0; j < dense_input; ++j) {
            dense_weights[i][j] = val;
            val += 0.01;
        }
    }

    // Test without multiplexing
    std::cout << "  Original dense weights: " << dense_weights.size() << " x " << dense_weights[0].size() << std::endl;
    std::vector<double> dense_output_gap1 = MatVecMul(dense_weights, output_gap1);
    std::cout << "  Output with gap=1: ";
    for (int i = 0; i < std::min(5, (int)dense_output_gap1.size()); ++i) {
        std::cout << dense_output_gap1[i] << " ";
    }
    std::cout << std::endl;

    // Test with multiplexing
    auto dense_weights_multiplexed = MultiplexDenseMatrix(dense_weights, output_h, output_w, output_gap);
    std::cout << "  Multiplexed dense weights: " << dense_weights_multiplexed.size() << " x " << dense_weights_multiplexed[0].size() << std::endl;
    std::vector<double> dense_output_gap2 = MatVecMul(dense_weights_multiplexed, output_gap2_multiplexed);
    std::cout << "  Output with gap=2 (multiplexed input): ";
    for (int i = 0; i < std::min(5, (int)dense_output_gap2.size()); ++i) {
        std::cout << dense_output_gap2[i] << " ";
    }
    std::cout << std::endl;

    // Compare dense outputs
    std::cout << "\n=== Dense Layer Comparison ===" << std::endl;
    max_error = 0.0;
    sum_error = 0.0;
    count = std::min(dense_output_gap1.size(), dense_output_gap2.size());

    for (int i = 0; i < count; ++i) {
        double error = std::abs(dense_output_gap1[i] - dense_output_gap2[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
    }

    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Avg error: " << sum_error / count << std::endl;

    bool dense_test_passed = (max_error < 1e-6);
    if (dense_test_passed) {
        std::cout << "  ✓ PASS - MultiplexDenseMatrix works correctly!" << std::endl;
    } else {
        std::cout << "  ✗ FAIL - MultiplexDenseMatrix has errors!" << std::endl;
        std::cout << "\n  Detailed comparison:" << std::endl;
        for (int i = 0; i < count; ++i) {
            std::cout << "    [" << i << "] gap=1: " << dense_output_gap1[i]
                     << ", gap=2: " << dense_output_gap2[i]
                     << ", error: " << std::abs(dense_output_gap1[i] - dense_output_gap2[i]) << std::endl;
        }
    }

    return (conv_test_passed && dense_test_passed) ? 0 : 1;
}
