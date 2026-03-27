#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace openfhe_numpy {

/**
 * @brief Load shape information from .shape file
 */
inline std::vector<size_t> LoadShape(const std::string& shapePath) {
    std::ifstream shapeFile(shapePath);
    if (!shapeFile.is_open()) {
        throw std::runtime_error("Failed to open shape file: " + shapePath);
    }

    std::vector<size_t> shape;
    size_t dim;
    while (shapeFile >> dim) {
        shape.push_back(dim);
    }
    shapeFile.close();

    return shape;
}

/**
 * @brief Load 1D weights (biases or dense weights flattened)
 */
inline std::vector<double> Load1DWeights(const std::string& binPath, const std::string& shapePath) {
    auto shape = LoadShape(shapePath);

    size_t totalSize = 1;
    for (size_t dim : shape) {
        totalSize *= dim;
    }

    std::vector<double> weights(totalSize);
    std::ifstream binFile(binPath, std::ios::binary);
    if (!binFile.is_open()) {
        throw std::runtime_error("Failed to open binary file: " + binPath);
    }

    binFile.read(reinterpret_cast<char*>(weights.data()), totalSize * sizeof(double));
    binFile.close();

    return weights;
}

/**
 * @brief Load 2D weights (Dense layer: [out_features, in_features])
 */
inline std::vector<std::vector<double>> Load2DWeights(const std::string& binPath, const std::string& shapePath) {
    auto shape = LoadShape(shapePath);

    if (shape.size() != 2) {
        throw std::runtime_error("Expected 2D shape for dense weights, got " + std::to_string(shape.size()) + "D");
    }

    size_t out_features = shape[0];
    size_t in_features = shape[1];
    size_t totalSize = out_features * in_features;

    std::vector<double> flat_weights(totalSize);
    std::ifstream binFile(binPath, std::ios::binary);
    if (!binFile.is_open()) {
        throw std::runtime_error("Failed to open binary file: " + binPath);
    }

    binFile.read(reinterpret_cast<char*>(flat_weights.data()), totalSize * sizeof(double));
    binFile.close();

    // Convert to 2D
    std::vector<std::vector<double>> weights(out_features, std::vector<double>(in_features));
    for (size_t i = 0; i < out_features; i++) {
        for (size_t j = 0; j < in_features; j++) {
            weights[i][j] = flat_weights[i * in_features + j];
        }
    }

    return weights;
}

/**
 * @brief Load 4D convolutional weights (Conv2D: [out_channels, in_channels, kernel_h, kernel_w])
 */
inline std::vector<std::vector<std::vector<std::vector<double>>>> Load4DWeights(
    const std::string& binPath,
    const std::string& shapePath
) {
    auto shape = LoadShape(shapePath);

    if (shape.size() != 4) {
        throw std::runtime_error("Expected 4D shape for conv weights, got " + std::to_string(shape.size()) + "D");
    }

    size_t out_channels = shape[0];
    size_t in_channels = shape[1];
    size_t kernel_h = shape[2];
    size_t kernel_w = shape[3];
    size_t totalSize = out_channels * in_channels * kernel_h * kernel_w;

    std::vector<double> flat_weights(totalSize);
    std::ifstream binFile(binPath, std::ios::binary);
    if (!binFile.is_open()) {
        throw std::runtime_error("Failed to open binary file: " + binPath);
    }

    binFile.read(reinterpret_cast<char*>(flat_weights.data()), totalSize * sizeof(double));
    binFile.close();

    // Convert to 4D
    std::vector<std::vector<std::vector<std::vector<double>>>> weights(
        out_channels, std::vector<std::vector<std::vector<double>>>(
            in_channels, std::vector<std::vector<double>>(
                kernel_h, std::vector<double>(kernel_w)
            )
        )
    );

    for (size_t oc = 0; oc < out_channels; oc++) {
        for (size_t ic = 0; ic < in_channels; ic++) {
            for (size_t kh = 0; kh < kernel_h; kh++) {
                for (size_t kw = 0; kw < kernel_w; kw++) {
                    size_t idx = oc * (in_channels * kernel_h * kernel_w) +
                                ic * (kernel_h * kernel_w) +
                                kh * kernel_w +
                                kw;
                    weights[oc][ic][kh][kw] = flat_weights[idx];
                }
            }
        }
    }

    return weights;
}

/**
 * @brief Load LeNet-5 ReLU model weights from directory
 */
struct LeNet5Weights {
    std::vector<std::vector<std::vector<std::vector<double>>>> conv1_weight;
    std::vector<std::vector<std::vector<std::vector<double>>>> conv2_weight;
    std::vector<std::vector<double>> fc1_weight;
    std::vector<std::vector<double>> fc2_weight;
    std::vector<std::vector<double>> fc3_weight;

    // Optional biases (if your model has them)
    std::vector<double> conv1_bias;
    std::vector<double> conv2_bias;
    std::vector<double> fc1_bias;
    std::vector<double> fc2_bias;
    std::vector<double> fc3_bias;

    // Optional scaling factors (for square activation model)
    double scale1 = 1.0;
    double scale2 = 1.0;
    double scale3 = 1.0;
    double scale4 = 1.0;
    bool has_scales = false;
};

inline LeNet5Weights LoadLeNet5Weights(const std::string& weightsDir) {
    LeNet5Weights weights;

    // Load convolutional layers
    weights.conv1_weight = Load4DWeights(
        weightsDir + "/conv1.weight.bin",
        weightsDir + "/conv1.weight.shape"
    );

    weights.conv2_weight = Load4DWeights(
        weightsDir + "/conv2.weight.bin",
        weightsDir + "/conv2.weight.shape"
    );

    // Load fully connected layers
    weights.fc1_weight = Load2DWeights(
        weightsDir + "/fc1.weight.bin",
        weightsDir + "/fc1.weight.shape"
    );

    weights.fc2_weight = Load2DWeights(
        weightsDir + "/fc2.weight.bin",
        weightsDir + "/fc2.weight.shape"
    );

    weights.fc3_weight = Load2DWeights(
        weightsDir + "/fc3.weight.bin",
        weightsDir + "/fc3.weight.shape"
    );

    // Load biases (if they exist)
    try {
        weights.conv1_bias = Load1DWeights(
            weightsDir + "/conv1.bias.bin",
            weightsDir + "/conv1.bias.shape"
        );
        weights.conv2_bias = Load1DWeights(
            weightsDir + "/conv2.bias.bin",
            weightsDir + "/conv2.bias.shape"
        );
        weights.fc1_bias = Load1DWeights(
            weightsDir + "/fc1.bias.bin",
            weightsDir + "/fc1.bias.shape"
        );
        weights.fc2_bias = Load1DWeights(
            weightsDir + "/fc2.bias.bin",
            weightsDir + "/fc2.bias.shape"
        );
        weights.fc3_bias = Load1DWeights(
            weightsDir + "/fc3.bias.bin",
            weightsDir + "/fc3.bias.shape"
        );
    } catch (const std::exception& e) {
        std::cout << "Note: Biases not found or error loading them (this is okay if model has no biases)" << std::endl;
    }

    // Load scale factors (if they exist - for square activation model)
    try {
        auto scale1_vec = Load1DWeights(weightsDir + "/scale1.bin", weightsDir + "/scale1.shape");
        auto scale2_vec = Load1DWeights(weightsDir + "/scale2.bin", weightsDir + "/scale2.shape");
        auto scale3_vec = Load1DWeights(weightsDir + "/scale3.bin", weightsDir + "/scale3.shape");
        auto scale4_vec = Load1DWeights(weightsDir + "/scale4.bin", weightsDir + "/scale4.shape");

        if (!scale1_vec.empty()) weights.scale1 = scale1_vec[0];
        if (!scale2_vec.empty()) weights.scale2 = scale2_vec[0];
        if (!scale3_vec.empty()) weights.scale3 = scale3_vec[0];
        if (!scale4_vec.empty()) weights.scale4 = scale4_vec[0];
        weights.has_scales = true;

    } catch (const std::exception& e) {
        // No scales - this is normal for ReLU models
        weights.has_scales = false;
    }

    return weights;
}

/**
 * @brief Load MNIST image from binary file
 */
inline std::vector<std::vector<double>> LoadMNISTImage(const std::string& imagePath) {
    // Read binary file
    std::ifstream binFile(imagePath, std::ios::binary);
    if (!binFile.is_open()) {
        throw std::runtime_error("Failed to open MNIST image file: " + imagePath);
    }

    // MNIST images are always 28x28
    const size_t height = 28;
    const size_t width = 28;
    const size_t totalSize = height * width;

    std::vector<double> flat_image(totalSize);
    binFile.read(reinterpret_cast<char*>(flat_image.data()), totalSize * sizeof(double));
    binFile.close();

    // Convert to 2D
    std::vector<std::vector<double>> image(height, std::vector<double>(width));
    for (size_t h = 0; h < height; h++) {
        for (size_t w = 0; w < width; w++) {
            image[h][w] = flat_image[h * width + w];
        }
    }

    return image;
}

/**
 * @brief Load LoLa model weights from directory
 */
struct LoLaWeights {
    std::vector<std::vector<std::vector<std::vector<double>>>> conv1_weight;
    std::vector<std::vector<double>> fc1_weight;
    std::vector<std::vector<double>> fc2_weight;

    // Biases
    std::vector<double> conv1_bias;
    std::vector<double> fc1_bias;
    std::vector<double> fc2_bias;

    // Optional scaling factors (for square activation model)
    double scale1 = 1.0;
    double scale2 = 1.0;
    bool has_scales = false;
};

inline LoLaWeights LoadLoLaWeights(const std::string& weightsDir) {
    LoLaWeights weights;

    // Load convolutional layer
    weights.conv1_weight = Load4DWeights(
        weightsDir + "/conv1.weight.bin",
        weightsDir + "/conv1.weight.shape"
    );

    // Load fully connected layers
    weights.fc1_weight = Load2DWeights(
        weightsDir + "/fc1.weight.bin",
        weightsDir + "/fc1.weight.shape"
    );

    weights.fc2_weight = Load2DWeights(
        weightsDir + "/fc2.weight.bin",
        weightsDir + "/fc2.weight.shape"
    );

    // Load biases
    try {
        weights.conv1_bias = Load1DWeights(
            weightsDir + "/conv1.bias.bin",
            weightsDir + "/conv1.bias.shape"
        );
        weights.fc1_bias = Load1DWeights(
            weightsDir + "/fc1.bias.bin",
            weightsDir + "/fc1.bias.shape"
        );
        weights.fc2_bias = Load1DWeights(
            weightsDir + "/fc2.bias.bin",
            weightsDir + "/fc2.bias.shape"
        );
    } catch (const std::exception& e) {
        std::cout << "Note: Biases not found or error loading them" << std::endl;
    }

    // Load scale factors (if they exist - for square activation model)
    try {
        auto scale1_vec = Load1DWeights(weightsDir + "/scale1.bin", weightsDir + "/scale1.shape");
        auto scale2_vec = Load1DWeights(weightsDir + "/scale2.bin", weightsDir + "/scale2.shape");

        if (!scale1_vec.empty()) weights.scale1 = scale1_vec[0];
        if (!scale2_vec.empty()) weights.scale2 = scale2_vec[0];
        weights.has_scales = true;

    } catch (const std::exception& e) {
        // No scales - this is normal for ReLU models
        weights.has_scales = false;
    }

    return weights;
}

/**
 * @brief Load CIFAR-10 image from binary file (3x32x32, CHW, normalized, float64)
 */
inline std::vector<std::vector<std::vector<double>>> LoadCIFAR10Image(const std::string& imagePath) {
    std::ifstream binFile(imagePath, std::ios::binary);
    if (!binFile.is_open()) {
        throw std::runtime_error("Failed to open CIFAR-10 image file: " + imagePath);
    }

    const size_t channels = 3, height = 32, width = 32;
    const size_t totalSize = channels * height * width;

    std::vector<double> flat_image(totalSize);
    binFile.read(reinterpret_cast<char*>(flat_image.data()), totalSize * sizeof(double));
    binFile.close();

    // Convert to 3D [C, H, W]
    std::vector<std::vector<std::vector<double>>> image(channels,
        std::vector<std::vector<double>>(height, std::vector<double>(width)));
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                image[c][h][w] = flat_image[c * height * width + h * width + w];
            }
        }
    }

    return image;
}

/**
 * @brief ResNet-20 weights (no BatchNorm)
 *
 * Architecture: conv1 -> [layer1 x3] -> [layer2 x3] -> [layer3 x3] -> avgpool -> fc
 * Each block: conv1->relu->conv2 + shortcut -> relu
 */
struct ResNet20BlockWeights {
    std::vector<std::vector<std::vector<std::vector<double>>>> conv1_weight;
    std::vector<double> conv1_bias;
    std::vector<std::vector<std::vector<std::vector<double>>>> conv2_weight;
    std::vector<double> conv2_bias;

    bool has_shortcut = false;
    std::vector<std::vector<std::vector<std::vector<double>>>> shortcut_weight;
    std::vector<double> shortcut_bias;
};

struct ResNet20Weights {
    // Initial conv
    std::vector<std::vector<std::vector<std::vector<double>>>> conv1_weight;
    std::vector<double> conv1_bias;

    // 3 layers x 3 blocks each = 9 blocks
    ResNet20BlockWeights layer1[3];
    ResNet20BlockWeights layer2[3];
    ResNet20BlockWeights layer3[3];

    // FC layer
    std::vector<std::vector<double>> fc_weight;
    std::vector<double> fc_bias;
};

inline ResNet20BlockWeights LoadResNet20Block(const std::string& weightsDir,
                                               const std::string& prefix,
                                               bool expectShortcut) {
    ResNet20BlockWeights block;

    block.conv1_weight = Load4DWeights(
        weightsDir + "/" + prefix + ".conv1.weight.bin",
        weightsDir + "/" + prefix + ".conv1.weight.shape");
    block.conv1_bias = Load1DWeights(
        weightsDir + "/" + prefix + ".conv1.bias.bin",
        weightsDir + "/" + prefix + ".conv1.bias.shape");

    block.conv2_weight = Load4DWeights(
        weightsDir + "/" + prefix + ".conv2.weight.bin",
        weightsDir + "/" + prefix + ".conv2.weight.shape");
    block.conv2_bias = Load1DWeights(
        weightsDir + "/" + prefix + ".conv2.bias.bin",
        weightsDir + "/" + prefix + ".conv2.bias.shape");

    block.has_shortcut = expectShortcut;
    if (expectShortcut) {
        block.shortcut_weight = Load4DWeights(
            weightsDir + "/" + prefix + ".shortcut.weight.bin",
            weightsDir + "/" + prefix + ".shortcut.weight.shape");
        block.shortcut_bias = Load1DWeights(
            weightsDir + "/" + prefix + ".shortcut.bias.bin",
            weightsDir + "/" + prefix + ".shortcut.bias.shape");
    }

    return block;
}

inline ResNet20Weights LoadResNet20Weights(const std::string& weightsDir) {
    ResNet20Weights w;

    // Initial conv
    w.conv1_weight = Load4DWeights(
        weightsDir + "/conv1.weight.bin",
        weightsDir + "/conv1.weight.shape");
    w.conv1_bias = Load1DWeights(
        weightsDir + "/conv1.bias.bin",
        weightsDir + "/conv1.bias.shape");

    // Layer 1: 3 blocks, all 16->16, no shortcut
    for (int i = 0; i < 3; i++) {
        std::string prefix = "layer1." + std::to_string(i);
        w.layer1[i] = LoadResNet20Block(weightsDir, prefix, false);
    }

    // Layer 2: block 0 has shortcut (16->32, stride=2), blocks 1-2 no shortcut
    for (int i = 0; i < 3; i++) {
        std::string prefix = "layer2." + std::to_string(i);
        w.layer2[i] = LoadResNet20Block(weightsDir, prefix, i == 0);
    }

    // Layer 3: block 0 has shortcut (32->64, stride=2), blocks 1-2 no shortcut
    for (int i = 0; i < 3; i++) {
        std::string prefix = "layer3." + std::to_string(i);
        w.layer3[i] = LoadResNet20Block(weightsDir, prefix, i == 0);
    }

    // FC layer
    w.fc_weight = Load2DWeights(
        weightsDir + "/fc.weight.bin",
        weightsDir + "/fc.weight.shape");
    w.fc_bias = Load1DWeights(
        weightsDir + "/fc.bias.bin",
        weightsDir + "/fc.bias.shape");

    return w;
}

/**
 * @brief Extract label from MNIST filename
 * Format: mnist_X_label_Y.bin -> returns Y
 */
inline int GetLabelFromFilename(const std::string& filename) {
    size_t label_pos = filename.find("_label_");
    if (label_pos == std::string::npos) {
        return -1;
    }

    size_t start = label_pos + 7;  // length of "_label_"
    size_t end = filename.find(".bin", start);
    if (end == std::string::npos) {
        return -1;
    }

    std::string label_str = filename.substr(start, end - start);
    return std::stoi(label_str);
}

} // namespace openfhe_numpy

#endif // WEIGHT_LOADER_H
