#pragma once

#include "openfhe.h"

// CRITICAL: Include serialization headers for proper CEREAL type registration
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <stdexcept>

using namespace lbcrypto;

namespace openfhe_numpy {

/**
 * @brief Utility functions for layer-wise rotation key serialization and ciphertext checkpointing
 *
 * Purpose: Reduce memory footprint by:
 * 1. Serializing rotation keys per layer (load only when needed)
 * 2. Checkpointing intermediate ciphertexts to disk
 * 3. Clearing keys/ciphertexts from memory after each layer
 *
 * Memory savings: ~700MB → ~100-250MB for LeNet-5
 */

/**
 * @brief Create directory if it doesn't exist
 */
inline bool CreateDirectoryIfNotExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        #ifdef _WIN32
            return _mkdir(path.c_str()) == 0;
        #else
            return mkdir(path.c_str(), 0755) == 0;
        #endif
    } else if (info.st_mode & S_IFDIR) {
        // Directory already exists
        return true;
    }
    return false;
}

/**
 * @brief Save rotation keys for a specific layer
 * @param cc CryptoContext
 * @param filepath Path to save the rotation keys
 * @return true if successful, false otherwise
 */
inline bool SaveRotationKeys(const CryptoContext<DCRTPoly>& cc, const std::string& filepath) {
    try {
        std::ofstream keyFile(filepath, std::ios::out | std::ios::binary);
        if (!keyFile.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << filepath << std::endl;
            return false;
        }

        if (!cc->SerializeEvalAutomorphismKey(keyFile, SerType::BINARY)) {
            std::cerr << "Error: Failed to serialize rotation keys to " << filepath << std::endl;
            keyFile.close();
            return false;
        }

        keyFile.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in SaveRotationKeys: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Load rotation keys for a specific layer
 * @param cc CryptoContext
 * @param filepath Path to load the rotation keys from
 * @return true if successful, false otherwise
 */
inline bool LoadRotationKeys(CryptoContext<DCRTPoly>& cc, const std::string& filepath) {
    try {
        std::ifstream keyFile(filepath, std::ios::in | std::ios::binary);
        if (!keyFile.is_open()) {
            std::cerr << "Error: Cannot open file for reading: " << filepath << std::endl;
            return false;
        }

        if (!cc->DeserializeEvalAutomorphismKey(keyFile, SerType::BINARY)) {
            std::cerr << "Error: Failed to deserialize rotation keys from " << filepath << std::endl;
            keyFile.close();
            return false;
        }

        keyFile.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in LoadRotationKeys: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Clear rotation keys from memory
 * @param cc CryptoContext
 */
inline void ClearRotationKeys(CryptoContext<DCRTPoly>& cc) {
    cc->ClearEvalAutomorphismKeys();
}

/**
 * @brief Save ciphertext checkpoint to disk
 * @param ct Ciphertext to save
 * @param filepath Path to save the ciphertext
 * @return true if successful, false otherwise
 */
inline bool SaveCiphertext(const Ciphertext<DCRTPoly>& ct, const std::string& filepath) {
    try {
        if (!Serial::SerializeToFile(filepath, ct, SerType::BINARY)) {
            std::cerr << "Error: Failed to serialize ciphertext to " << filepath << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in SaveCiphertext: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Load ciphertext checkpoint from disk
 * @param filepath Path to load the ciphertext from
 * @param ct Output ciphertext (will be populated)
 * @return true if successful, false otherwise
 */
inline bool LoadCiphertext(const std::string& filepath, Ciphertext<DCRTPoly>& ct) {
    try {
        if (!Serial::DeserializeFromFile(filepath, ct, SerType::BINARY)) {
            std::cerr << "Error: Failed to deserialize ciphertext from " << filepath << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in LoadCiphertext: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Layer-wise key and checkpoint manager
 *
 * Usage pattern:
 * 1. Setup phase: Generate and save all keys
 * 2. Inference phase: Load keys per layer, checkpoint outputs, clear memory
 */
struct LayerKeyManager {
    std::string keysDir;
    std::string checkpointsDir;

    LayerKeyManager(const std::string& keysPath = "./keys",
                   const std::string& checkpointsPath = "./checkpoints")
        : keysDir(keysPath), checkpointsDir(checkpointsPath) {
        CreateDirectoryIfNotExists(keysDir);
        CreateDirectoryIfNotExists(checkpointsDir);
    }

    // Generate and save rotation keys for a specific layer
    bool GenerateAndSaveLayerKeys(
        CryptoContext<DCRTPoly>& cc,
        const PrivateKey<DCRTPoly>& secretKey,
        const std::vector<int32_t>& rotations,
        const std::string& layerName
    ) {
        // Clear any existing keys first
        cc->ClearEvalAutomorphismKeys();

        // Generate keys for this layer
        cc->EvalRotateKeyGen(secretKey, rotations);

        // Save to file
        std::string filepath = keysDir + "/" + layerName + "_rotkeys.bin";
        bool success = SaveRotationKeys(cc, filepath);

        // Clear keys from memory after saving
        cc->ClearEvalAutomorphismKeys();

        return success;
    }

    // Load rotation keys for a specific layer
    bool LoadLayerKeys(CryptoContext<DCRTPoly>& cc, const std::string& layerName) {
        std::string filepath = keysDir + "/" + layerName + "_rotkeys.bin";
        return LoadRotationKeys(cc, filepath);
    }

    // Save layer output checkpoint
    bool SaveLayerCheckpoint(const Ciphertext<DCRTPoly>& ct, const std::string& layerName) {
        std::string filepath = checkpointsDir + "/" + layerName + "_output.bin";
        return SaveCiphertext(ct, filepath);
    }

    // Load layer checkpoint
    bool LoadLayerCheckpoint(const std::string& layerName, Ciphertext<DCRTPoly>& ct) {
        std::string filepath = checkpointsDir + "/" + layerName + "_output.bin";
        return LoadCiphertext(filepath, ct);
    }
};

} // namespace openfhe_numpy
