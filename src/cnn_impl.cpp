#include "cnn_impl.h"
#include <iostream>
#include <algorithm>
#include <tuple>

namespace cnn {

CnnModel::CnnModel(const std::string& path, float conf_th, float iou_th)
    : path_(path), confidence_threshold_(conf_th), iou_threshold_(iou_th) {

    std::cout << "Loading cnn model: " << path << std::endl;

    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CnnModel");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Try CUDA first, fallback to CPU
    try {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "CUDA provider configured" << std::endl;
    } catch (...) {
        std::cout << "CUDA failed, using CPU" << std::endl;
    }

    std::wstring wide_path(path.begin(), path.end());
    session_ = std::make_unique<Ort::Session>(*env_, wide_path.c_str(), session_options);

    auto input_name_ptr = session_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    input_name_ = input_name_ptr.get();

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_shape = tensor_info.GetShape();
    input_size_ = static_cast<int>(input_shape[2]);

    // Warmup
    std::vector<int64_t> shape = {1, 3, input_size_, input_size_};
    std::vector<float> image(3 * input_size_ * input_size_, 0.0f);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (int i = 0; i < 10; ++i) {
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, image.data(), image.size(), shape.data(), shape.size());
        std::vector<const char*> input_names = {input_name_.c_str()};

        // Get output names for warmup
        size_t num_outputs = session_->GetOutputCount();
        std::vector<Ort::AllocatedStringPtr> warmup_output_name_ptrs;
        std::vector<const char*> warmup_output_names;

        for (size_t j = 0; j < num_outputs; ++j) {
            auto name_ptr = session_->GetOutputNameAllocated(j, Ort::AllocatorWithDefaultOptions());
            warmup_output_names.push_back(name_ptr.get());
            warmup_output_name_ptrs.push_back(std::move(name_ptr));
        }

        session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                     warmup_output_names.data(), warmup_output_names.size());
    }

    std::cout << "Model initialized: " << input_size_ << "x" << input_size_ << std::endl;
}

std::tuple<cv::Mat, std::pair<int, int>, float> CnnModel::resize(const cv::Mat& image, cv::Size shape) {
    float r = std::min(static_cast<float>(input_size_) / shape.height,
                       static_cast<float>(input_size_) / shape.width);

    std::pair<int, int> pad = {static_cast<int>(std::round(shape.width * r)),
                               static_cast<int>(std::round(shape.height * r))};

    float w = (input_size_ - pad.first) / 2.0f;
    float h = (input_size_ - pad.second) / 2.0f;

    cv::Mat resized = image.clone();
    if (cv::Size(shape.width, shape.height) != cv::Size(pad.first, pad.second)) {
        cv::resize(image, resized, cv::Size(pad.first, pad.second), 0, 0, cv::INTER_LINEAR);
    }

    int top = static_cast<int>(std::round(h - 0.1));
    int bottom = static_cast<int>(std::round(h + 0.1));
    int left = static_cast<int>(std::round(w - 0.1));
    int right = static_cast<int>(std::round(w + 0.1));

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return std::make_tuple(padded, std::make_pair(top, left), r);
}

std::vector<std::vector<float>> CnnModel::infer(const cv::Mat& image) {
    // Resize with padding - exactly like Python version
    auto [x, pad, gain] = resize(image, image.size());

    // Convert to RGB and normalize - exactly like Python version
    cv::Mat rgb;
    cv::cvtColor(x, rgb, cv::COLOR_BGR2RGB);

    // Convert to CHW format and normalize - exactly like Python version
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> input_data(3 * input_size_ * input_size_);
    for (int c = 0; c < 3; ++c) {
        cv::Mat float_channel;
        channels[2-c].convertTo(float_channel, CV_32F, 1.0/255.0); // Reverse channel order like Python [::-1]
        std::memcpy(input_data.data() + c * input_size_ * input_size_,
                   float_channel.data, input_size_ * input_size_ * sizeof(float));
    }

    // Run inference
    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    std::vector<const char*> input_names = {input_name_.c_str()};

    // Get output names
    size_t num_outputs = session_->GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char*> output_names;

    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_names.push_back(name_ptr.get());
        output_name_ptrs.push_back(std::move(name_ptr));
    }

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                               output_names.data(), output_names.size());

    // Process outputs - exactly like Python version
    auto& output_tensor = outputs[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    const float* output_data = output_tensor.GetTensorData<float>();

    // Transpose [1, 84, N] -> [N, 84] like Python version
    int num_detections = static_cast<int>(output_shape[2]);
    int detection_size = static_cast<int>(output_shape[1]);

    std::vector<float> transposed_data(num_detections * detection_size);
    for (int n = 0; n < num_detections; ++n) {
        for (int d = 0; d < detection_size; ++d) {
            transposed_data[n * detection_size + d] = output_data[d * num_detections + n];
        }
    }

    // Remove padding offset - exactly like Python version
    for (int i = 0; i < num_detections; ++i) {
        float* detection = transposed_data.data() + i * detection_size;
        detection[0] -= pad.second; // pad[1] in Python
        detection[1] -= pad.first;  // pad[0] in Python
    }

    // Get class scores - exactly like Python version
    int num_classes = detection_size - 4;
    std::vector<float> max_scores;
    std::vector<int> class_indices;
    std::vector<int> valid_indices;

    for (int i = 0; i < num_detections; ++i) {
        const float* detection = transposed_data.data() + i * detection_size;
        const float* class_scores = detection + 4;

        auto max_iter = std::max_element(class_scores, class_scores + num_classes);
        float max_score = *max_iter;
        int class_id = static_cast<int>(max_iter - class_scores);

        if (max_score >= confidence_threshold_) {
            max_scores.push_back(max_score);
            class_indices.push_back(class_id);
            valid_indices.push_back(i);
        }
    }

    if (valid_indices.empty()) {
        return {};
    }

    // Convert to boxes format - exactly like Python version
    std::vector<cv::Rect> boxes;
    for (int idx : valid_indices) {
        const float* detection = transposed_data.data() + idx * detection_size;
        float cx = detection[0];
        float cy = detection[1];
        float w = detection[2];
        float h = detection[3];

        int left = static_cast<int>((cx - w / 2) / gain);
        int top = static_cast<int>((cy - h / 2) / gain);
        int width = static_cast<int>(w / gain);
        int height = static_cast<int>(h / gain);

        boxes.emplace_back(left, top, width, height);
    }

    // Apply NMS - exactly like Python version
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, max_scores, confidence_threshold_, iou_threshold_, nms_indices);

    // Format results - exactly like Python version
    std::vector<std::vector<float>> results;
    for (int idx : nms_indices) {
        std::vector<float> detection = {
            static_cast<float>(boxes[idx].x),
            static_cast<float>(boxes[idx].y),
            static_cast<float>(boxes[idx].width),
            static_cast<float>(boxes[idx].height),
            max_scores[idx],
            static_cast<float>(class_indices[idx])
        };
        results.push_back(detection);
    }

    return results;
}

} // namespace cnn