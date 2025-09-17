#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

// Define export macros for DLL
#ifdef CNN_EXPORTS
    #define CNN_API __declspec(dllexport)
#else
    #define CNN_API __declspec(dllimport)
#endif

namespace cnn {

    // Structure to hold detection results
    struct Detection {
        int x;          // Left coordinate
        int y;          // Top coordinate
        int width;      // Width of bounding box
        int height;     // Height of bounding box
        float confidence; // Detection confidence score
        int class_id;   // Class ID

        Detection(int x, int y, int w, int h, float conf, int cls)
            : x(x), y(y), width(w), height(h), confidence(conf), class_id(cls) {}
    };

    // Forward declaration of implementation class
    class CNNImpl;

    // Main CNN class for YOLO inference
    class CNN_API CNN {
    private:
        std::unique_ptr<CNNImpl> impl_;

    public:
        CNN();
        ~CNN();

        // Disable copy constructor and assignment (we'll use move semantics)
        CNN(const CNN&) = delete;
        CNN& operator=(const CNN&) = delete;
        CNN(CNN&&) noexcept;
        CNN& operator=(CNN&&) noexcept;

        // Load ONNX YOLO model
        bool LoadModel(const std::string& model_path, 
                      float conf_threshold = 0.3f, 
                      float iou_threshold = 0.4f);

        // Perform inference on image data
        std::vector<Detection> Infer(const std::vector<uint8_t>& image_data, 
                                   int width, int height, int channels);

        // Perform inference on image file
        std::vector<Detection> InferFromFile(const std::string& image_path);

        // Utility functions
        bool IsModelLoaded() const;
        std::string GetLastError() const;
        int GetInputSize() const;
        void SetConfidenceThreshold(float threshold);
        void SetIouThreshold(float threshold);
        float GetConfidenceThreshold() const;
        float GetIouThreshold() const;
    };
}

// C-style wrapper functions for easier DLL interoperability
extern "C" {
    CNN_API void* CreateCNNInstance();
    CNN_API void DeleteCNNInstance(void* instance);
    CNN_API bool CNN_LoadModel(void* instance, const char* model_path, float conf_threshold, float iou_threshold);
    CNN_API bool CNN_IsModelLoaded(void* instance);
    CNN_API const char* CNN_GetLastError(void* instance);
    CNN_API int CNN_GetInputSize(void* instance);
    CNN_API void CNN_SetConfidenceThreshold(void* instance, float threshold);
    CNN_API void CNN_SetIouThreshold(void* instance, float threshold);
    CNN_API float CNN_GetConfidenceThreshold(void* instance);
    CNN_API float CNN_GetIouThreshold(void* instance);
    
    // Inference functions
    CNN_API int CNN_Infer(void* instance, const unsigned char* image_data, int width, int height, int channels, 
                         int* x_coords, int* y_coords, int* widths, int* heights, 
                         float* confidences, int* class_ids, int max_detections);
    CNN_API int CNN_InferFromFile(void* instance, const char* image_path,
                                 int* x_coords, int* y_coords, int* widths, int* heights,
                                 float* confidences, int* class_ids, int max_detections);
}