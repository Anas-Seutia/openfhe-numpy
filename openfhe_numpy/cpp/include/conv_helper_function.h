#ifndef __CONV_HELPER_FUNCTION_H__
#define __CONV_HELPER_FUNCTION_H__

#include <cstdint>
#include <vector>

/**
 * @brief Construct Toeplitz matrix for 2D convolution
 * 
 * @param kernel Input kernel (Co x Ci x kH x kW)
 * @param in_channels Input channels (Ci)
 * @param out_channels Output channels (Co)
 * @param input_height Input height (Hi)
 * @param input_width Input width (Wi)
 * @param kernel_height Kernel height (kH)
 * @param kernel_width Kernel width (kW)
 * @param stride Stride (assumes square stride)
 * @param padding Padding (assumes square padding)
 * @param dilation Dilation (assumes square dilation)
 * @param batch_size Batch size (N)
 * @param input_gap Multiplexing gap for input (iG)
 * @param output_gap Multiplexing gap for output (oG)
 * @return Toeplitz matrix representation as 2D vector
 */
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
    const uint32_t &output_gap
);

#endif  // __CONV_HELPER_FUNCTION_H__