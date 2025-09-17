#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Define export macros for DLL
#ifdef CNN_EXPORTS
    #define CNN_API __declspec(dllexport)
#else
    #define CNN_API __declspec(dllimport)
#endif

namespace cnn {

class CNN_API CnnModel {
public:
    // Initialize model - exactly like yolo8onnx
    CnnModel(const std::string& path,
             float conf_th = 0.3f,
             float iou_th = 0.4f,
             const std::vector<std::string>& providers = {"CUDAExecutionProvider", "CPUExecutionProvider"});

    ~CnnModel() = default;

    // Run inference - returns [x, y, w, h, confidence, class_id] for each detection
    std::vector<std::vector<float>> infer(const cv::Mat& image);

    // Resize image with padding - returns resized image, padding info, and gain
    std::tuple<cv::Mat, std::pair<int, int>, float> resize(const cv::Mat& image, const cv::Size& shape);

private:
    std::string path_;
    float confidence_threshold_;
    float iou_threshold_;
    int input_size_;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;

    void warmup();
};

} // namespace cnn