#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#ifdef CNN_EXPORTS
    #define CNN_API __declspec(dllexport)
#else
    #define CNN_API __declspec(dllimport)
#endif

namespace cnn {

enum class OutputFormat {
    CONCATENATED,  // Single output [1, N, 8400] - YOLOv8 style
    SPLIT_HEAD     // Multiple outputs (reg/cls pairs) - YOLOv11 style with DFL
};

class CNN_API CnnModel {
public:
    CnnModel(const std::string& path, float conf_th = 0.3f, float iou_th = 0.4f);
    ~CnnModel() = default;

    std::vector<std::vector<float>> infer(const cv::Mat& image);
    std::tuple<cv::Mat, std::pair<int, int>, float> resize(const cv::Mat& image, cv::Size shape);

private:
    // Core parameters
    std::string path_;
    float confidence_threshold_;
    float iou_threshold_;
    int input_size_;

    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> output_shapes_;

    // Output format
    OutputFormat output_format_;
    int num_classes_;

    // Split-head specific (DFL)
    std::vector<std::vector<float>> meshgrid_;  // Pre-generated grid for each scale
    static constexpr int HEAD_NUM = 3;
    static constexpr int STRIDES[3] = {8, 16, 32};
    static constexpr int MAP_SIZES[3][2] = {{80, 80}, {40, 40}, {20, 20}};

    // Helper methods
    OutputFormat detectOutputFormat();
    void generateMeshgrid();
    std::vector<std::vector<float>> postprocessConcatenated(
        const std::vector<Ort::Value>& outputs, int img_h, int img_w);
    std::vector<std::vector<float>> postprocessSplitHead(
        const std::vector<Ort::Value>& outputs, int img_h, int img_w);
};

} // namespace cnn