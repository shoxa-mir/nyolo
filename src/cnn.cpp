#include "cnn.h"
#include "cnn_impl.h"
#include <iostream>

namespace cnn {

    CNN::CNN() : impl_(std::make_unique<CNNImpl>()) {}

    CNN::~CNN() = default;

    CNN::CNN(CNN&&) noexcept = default;
    CNN& CNN::operator=(CNN&&) noexcept = default;

    bool CNN::LoadModel(const std::string& model_path, float conf_threshold, float iou_threshold) {
        return impl_->LoadModel(model_path, conf_threshold, iou_threshold);
    }

    std::vector<Detection> CNN::Infer(const std::vector<uint8_t>& image_data, 
                                     int width, int height, int channels) {
        return impl_->Infer(image_data, width, height, channels);
    }

    std::vector<Detection> CNN::InferFromFile(const std::string& image_path) {
        return impl_->InferFromFile(image_path);
    }

    bool CNN::IsModelLoaded() const {
        return impl_->IsModelLoaded();
    }

    std::string CNN::GetLastError() const {
        return impl_->GetLastError();
    }

    int CNN::GetInputSize() const {
        return impl_->GetInputSize();
    }

    void CNN::SetConfidenceThreshold(float threshold) {
        impl_->SetConfidenceThreshold(threshold);
    }

    void CNN::SetIouThreshold(float threshold) {
        impl_->SetIouThreshold(threshold);
    }

    float CNN::GetConfidenceThreshold() const {
        return impl_->GetConfidenceThreshold();
    }

    float CNN::GetIouThreshold() const {
        return impl_->GetIouThreshold();
    }

} // namespace cnn

// C-style wrapper functions
extern "C" {

    void* CreateCNNInstance() {
        try {
            return new cnn::CNN();
        } catch (const std::exception& e) {
            std::cerr << "Error creating CNN instance: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void DeleteCNNInstance(void* instance) {
        if (instance) {
            delete static_cast<cnn::CNN*>(instance);
        }
    }

    bool CNN_LoadModel(void* instance, const char* model_path, float conf_threshold, float iou_threshold) {
        if (!instance || !model_path) return false;
        try {
            auto* cnn_instance = static_cast<cnn::CNN*>(instance);
            return cnn_instance->LoadModel(std::string(model_path), conf_threshold, iou_threshold);
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    }

    bool CNN_IsModelLoaded(void* instance) {
        if (!instance) return false;
        try {
            auto* cnn_instance = static_cast<cnn::CNN*>(instance);
            return cnn_instance->IsModelLoaded();
        } catch (const std::exception& e) {
            std::cerr << "Error checking model status: " << e.what() << std::endl;
            return false;
        }
    }

    const char* CNN_GetLastError(void* instance) {
        if (!instance) return "Invalid instance";
        try {
            auto* cnn_instance = static_cast<cnn::CNN*>(instance);
            static std::string error = cnn_instance->GetLastError();
            return error.c_str();
        } catch (const std::exception& e) {
            return "Error getting last error";
        }
    }

    int CNN_GetInputSize(void* instance) {
        if (!instance) return -1;
        try {
            auto* cnn_instance = static_cast<cnn::CNN*>(instance);
            return cnn_instance->GetInputSize();
        } catch (const std::exception& e) {
            return -1;
        }
    }

    void CNN_SetConfidenceThreshold(void* instance, float threshold) {
        if (!instance) return;
        try {
            auto* cnn_instance = static_cast<cnn::CNN*>(instance);
            cnn_instance->SetConfidenceThreshold(threshold);
        } catch (const std::exception& e) {
            std::cerr << "Error setting confidence threshold: " << e.what() << std::endl;
        }
    }

    void CNN_SetIouThreshold(void* instance, float threshold) {
        if (!instance) return;
        try {
            auto* cnn_instance = static_cast<cnn::CNN*>(instance);
            cnn_instance->SetIouThreshold(threshold);
        } catch (const std::exception& e) {
            std::cerr << "Error setting IoU threshold: " << e.what() << std::endl;
        }
    }

    float CNN_GetConfidenceThreshold(void* instance) {
        if (!instance) return -1.0f;
        try {
            auto* cnn_instance = static_cast<cnn::CNN*>(instance);
            return cnn_instance->GetConfidenceThreshold();
        } catch (const std::exception& e) {
            return -1.0f;
        }
    }

    float CNN_GetIouThreshold(void* instance) {
        if (!instance) return -1.0f;
        try {
            auto* cnn_instance = static_cast<cnn::CNN*>(instance);
            return cnn_instance->GetIouThreshold();
        } catch (const std::exception& e) {
            return -1.0f;
        }
    }

    int CNN_Infer(void* instance, const unsigned char* image_data, int width, int height, int channels,
                 int* x_coords, int* y_coords, int* widths, int* heights,
                 float* confidences, int* class_ids, int max_detections) {
        if (!instance || !image_data || !x_coords || !y_coords || !widths || !heights || 
            !confidences || !class_ids) return 0;
        
        try {
            auto* cnn_instance = static_cast<cnn::CNN*>(instance);
            
            // Convert raw data to vector
            std::vector<uint8_t> img_vec(image_data, image_data + (width * height * channels));
            
            auto detections = cnn_instance->Infer(img_vec, width, height, channels);
            
            int num_detections = std::min(static_cast<int>(detections.size()), max_detections);
            
            for (int i = 0; i < num_detections; ++i) {
                x_coords[i] = detections[i].x;
                y_coords[i] = detections[i].y;
                widths[i] = detections[i].width;
                heights[i] = detections[i].height;
                confidences[i] = detections[i].confidence;
                class_ids[i] = detections[i].class_id;
            }
            
            return num_detections;
        } catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            return 0;
        }
    }

    int CNN_InferFromFile(void* instance, const char* image_path,
                         int* x_coords, int* y_coords, int* widths, int* heights,
                         float* confidences, int* class_ids, int max_detections) {
        if (!instance || !image_path || !x_coords || !y_coords || !widths || !heights || 
            !confidences || !class_ids) return 0;
        
        try {
            auto* cnn_instance = static_cast<cnn::CNN*>(instance);
            
            auto detections = cnn_instance->InferFromFile(std::string(image_path));
            
            int num_detections = std::min(static_cast<int>(detections.size()), max_detections);
            
            for (int i = 0; i < num_detections; ++i) {
                x_coords[i] = detections[i].x;
                y_coords[i] = detections[i].y;
                widths[i] = detections[i].width;
                heights[i] = detections[i].height;
                confidences[i] = detections[i].confidence;
                class_ids[i] = detections[i].class_id;
            }
            
            return num_detections;
        } catch (const std::exception& e) {
            std::cerr << "Error during file inference: " << e.what() << std::endl;
            return 0;
        }
    }

}