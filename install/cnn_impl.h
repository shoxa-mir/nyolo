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

class CNN_API CnnModel {
public:
    CnnModel(const std::string& path, float conf_th = 0.3f, float iou_th = 0.4f);
    ~CnnModel() = default;

    std::vector<std::vector<float>> infer(const cv::Mat& image);
    std::tuple<cv::Mat, std::pair<int, int>, float> resize(const cv::Mat& image, cv::Size shape);

private:
    std::string path_;
    float confidence_threshold_;
    float iou_threshold_;
    int input_size_;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
};

} // namespace cnn