#include "cnn_impl.h"
#include <iostream>
#include <algorithm>
#include <tuple>

namespace cnn {

CnnModel::CnnModel(const std::string& path, float conf_th, float iou_th)
    : path_(path), confidence_threshold_(conf_th), iou_threshold_(iou_th) {

    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CnnModel");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Try CUDA first, fallback to CPU
    try {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    } catch (...) {
        // Silently fall back to CPU
    }

    std::wstring wide_path(path.begin(), path.end());
    session_ = std::make_unique<Ort::Session>(*env_, wide_path.c_str(), session_options);

    // Get input information
    auto input_name_ptr = session_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    input_name_ = input_name_ptr.get();

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_shape = tensor_info.GetShape();
    input_size_ = static_cast<int>(input_shape[2]);

    // Print input info - match Python format
    std::cout << "Model input name: " << input_name_ << std::endl;
    std::cout << "Model input shape: [";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Get output information
    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_names_.push_back(name_ptr.get());

        auto type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_shapes_.push_back(tensor_info.GetShape());
    }

    // Print output names - match Python format
    std::cout << "Model output names: [";
    for (size_t i = 0; i < output_names_.size(); ++i) {
        std::cout << "'" << output_names_[i] << "'";
        if (i < output_names_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Print output shapes - match Python format
    std::cout << "Model output shapes: [";
    for (size_t i = 0; i < output_shapes_.size(); ++i) {
        std::cout << "[";
        for (size_t j = 0; j < output_shapes_[i].size(); ++j) {
            std::cout << output_shapes_[i][j];
            if (j < output_shapes_[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (i < output_shapes_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Detect output format
    output_format_ = detectOutputFormat();
    std::cout << "Detected output format: " << (output_format_ == OutputFormat::CONCATENATED ? "concatenated" : "split-head") << std::endl;

    // Generate meshgrid for split-head format
    if (output_format_ == OutputFormat::SPLIT_HEAD) {
        generateMeshgrid();
    }

    // Warmup
    std::vector<int64_t> shape = {1, 3, input_size_, input_size_};
    std::vector<float> image(3 * input_size_ * input_size_, 0.0f);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<const char*> input_names = {input_name_.c_str()};
    std::vector<const char*> output_name_ptrs;
    for (const auto& name : output_names_) {
        output_name_ptrs.push_back(name.c_str());
    }

    for (int i = 0; i < 10; ++i) {
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, image.data(), image.size(), shape.data(), shape.size());

        session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                     output_name_ptrs.data(), output_name_ptrs.size());
    }
}

OutputFormat CnnModel::detectOutputFormat() {
    if (output_shapes_.size() == 1) {
        // Single output - concatenated format [1, N, 8400]
        auto& shape = output_shapes_[0];
        if (shape.size() == 3 && shape[1] >= 5) {
            num_classes_ = static_cast<int>(shape[1]) - 4;
            std::cout << "Detected concatenated format with " << num_classes_ << " classes" << std::endl;
            return OutputFormat::CONCATENATED;
        }
    } else if (output_shapes_.size() == 6) {
        // 6 outputs - split-head format with DFL
        // Expected: reg1, cls1, reg2, cls2, reg3, cls3
        // Get number of classes from first cls output (output index 1)
        auto& cls_shape = output_shapes_[1];  // [1, num_classes, h, w]
        num_classes_ = static_cast<int>(cls_shape[1]);
        std::cout << "Detected split-head format with " << num_classes_ << " classes" << std::endl;
        return OutputFormat::SPLIT_HEAD;
    }

    num_classes_ = 80;  // Default to COCO
    return OutputFormat::SPLIT_HEAD;
}

void CnnModel::generateMeshgrid() {
    meshgrid_.clear();
    for (int i = 0; i < HEAD_NUM; ++i) {
        int h = MAP_SIZES[i][0];
        int w = MAP_SIZES[i][1];
        std::vector<float> grid;
        grid.reserve(h * w * 2);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                grid.push_back(static_cast<float>(x) + 0.5f);  // grid_x
                grid.push_back(static_cast<float>(y) + 0.5f);  // grid_y
            }
        }
        meshgrid_.push_back(grid);
    }
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
    // Direct resize to input_size x input_size - exactly like Python (NO letterboxing!)
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_size_, input_size_), 0, 0, cv::INTER_LINEAR);

    // Convert to RGB - exactly like Python
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Convert to CHW format and normalize - exactly like Python
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> input_data(3 * input_size_ * input_size_);
    for (int c = 0; c < 3; ++c) {
        cv::Mat float_channel;
        channels[c].convertTo(float_channel, CV_32F, 1.0/255.0);  // FIX: channels[c] not channels[2-c]!
        std::memcpy(input_data.data() + c * input_size_ * input_size_,
                   float_channel.data, input_size_ * input_size_ * sizeof(float));
    }

    // Run inference
    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    std::vector<const char*> input_names = {input_name_.c_str()};
    std::vector<const char*> output_name_ptrs;
    for (const auto& name : output_names_) {
        output_name_ptrs.push_back(name.c_str());
    }

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                               output_name_ptrs.data(), output_name_ptrs.size());

    // Route to appropriate postprocessing
    if (output_format_ == OutputFormat::CONCATENATED) {
        return postprocessConcatenated(outputs, image.rows, image.cols);
    } else {
        return postprocessSplitHead(outputs, image.rows, image.cols);
    }
}

std::vector<std::vector<float>> CnnModel::postprocessConcatenated(
    const std::vector<Ort::Value>& outputs, int img_h, int img_w) {

    // Calculate scale factors - exactly like Python
    float scale_h = static_cast<float>(img_h) / input_size_;
    float scale_w = static_cast<float>(img_w) / input_size_;

    // Process outputs [1, N, 8400]
    auto& output_tensor = outputs[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    const float* output_data = output_tensor.GetTensorData<float>();

    // Transpose [1, N, num_anchors] -> [num_anchors, N]
    int num_detections = static_cast<int>(output_shape[2]);
    int detection_size = static_cast<int>(output_shape[1]);

    std::vector<float> transposed_data(num_detections * detection_size);
    for (int n = 0; n < num_detections; ++n) {
        for (int d = 0; d < detection_size; ++d) {
            transposed_data[n * detection_size + d] = output_data[d * num_detections + n];
        }
    }

    // Get class scores
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
    std::vector<cv::Rect2d> boxes;
    std::vector<std::array<float, 4>> boxes_xyxy;  // Store xyxy for final output

    for (int idx : valid_indices) {
        const float* detection = transposed_data.data() + idx * detection_size;
        float cx = detection[0];
        float cy = detection[1];
        float w = detection[2];
        float h = detection[3];

        // Convert xywh to xyxy - exactly like Python
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // Scale to original image size - exactly like Python
        float xmin = std::max(0.0f, std::min(x1 * scale_w, static_cast<float>(img_w)));
        float ymin = std::max(0.0f, std::min(y1 * scale_h, static_cast<float>(img_h)));
        float xmax = std::max(0.0f, std::min(x2 * scale_w, static_cast<float>(img_w)));
        float ymax = std::max(0.0f, std::min(y2 * scale_h, static_cast<float>(img_h)));

        boxes.emplace_back(xmin, ymin, xmax - xmin, ymax - ymin);
        boxes_xyxy.push_back({xmin, ymin, xmax, ymax});
    }

    // Apply NMS - exactly like Python version
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, max_scores, confidence_threshold_, iou_threshold_, nms_indices);

    // Format results - exactly like Python version (return xyxy format)
    std::vector<std::vector<float>> results;
    for (int idx : nms_indices) {
        auto& box = boxes_xyxy[idx];
        std::vector<float> detection = {
            box[0],  // xmin
            box[1],  // ymin
            box[2] - box[0],  // width
            box[3] - box[1],  // height
            max_scores[idx],
            static_cast<float>(class_indices[idx])
        };
        results.push_back(detection);
    }

    return results;
}

std::vector<std::vector<float>> CnnModel::postprocessSplitHead(
    const std::vector<Ort::Value>& outputs, int img_h, int img_w) {

    // Get scaling factors
    float scale_h = static_cast<float>(img_h) / input_size_;
    float scale_w = static_cast<float>(img_w) / input_size_;

    auto [x, pad, gain] = resize(cv::Mat(img_h, img_w, CV_8UC3), cv::Size(img_w, img_h));

    std::vector<cv::Rect2d> all_boxes;
    std::vector<float> all_scores;
    std::vector<int> all_class_ids;

    // Process each scale (3 heads)
    for (int head_idx = 0; head_idx < HEAD_NUM; ++head_idx) {
        int h = MAP_SIZES[head_idx][0];
        int w = MAP_SIZES[head_idx][1];
        int num_anchors = h * w;
        int stride = STRIDES[head_idx];

        // Get regression and classification outputs
        auto& reg_tensor = outputs[head_idx * 2 + 0];  // [1, 64, h, w] - DFL
        auto& cls_tensor = outputs[head_idx * 2 + 1];  // [1, num_classes, h, w]

        auto reg_shape = reg_tensor.GetTensorTypeAndShapeInfo().GetShape();
        auto cls_shape = cls_tensor.GetTensorTypeAndShapeInfo().GetShape();

        const float* reg_data = reg_tensor.GetTensorData<float>();
        const float* cls_data = cls_tensor.GetTensorData<float>();

        int reg_channels = static_cast<int>(reg_shape[1]);  // 64
        int cls_channels = static_cast<int>(cls_shape[1]);  // num_classes

        // Process each anchor
        for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx) {
            // Get max class score
            float max_score = -FLT_MAX;
            int max_class_id = 0;

            for (int c = 0; c < cls_channels; ++c) {
                float score = cls_data[c * num_anchors + anchor_idx];
                // Apply sigmoid
                score = 1.0f / (1.0f + std::exp(-score));
                if (score > max_score) {
                    max_score = score;
                    max_class_id = c;
                }
            }

            // Filter by confidence
            if (max_score < confidence_threshold_) {
                continue;
            }

            // DFL (Distribution Focal Loss) decoding
            // reg_data has 64 channels = 4 directions * 16 bins
            std::array<float, 4> distances = {0, 0, 0, 0};  // left, top, right, bottom

            for (int dir = 0; dir < 4; ++dir) {
                // Apply softmax over 16 bins for this direction
                float softmax_sum = 0.0f;
                std::array<float, 16> softmax_vals;

                // Find max for numerical stability
                float max_val = -FLT_MAX;
                for (int bin = 0; bin < 16; ++bin) {
                    int channel = dir * 16 + bin;
                    float val = reg_data[channel * num_anchors + anchor_idx];
                    if (val > max_val) max_val = val;
                }

                // Compute softmax
                for (int bin = 0; bin < 16; ++bin) {
                    int channel = dir * 16 + bin;
                    float val = reg_data[channel * num_anchors + anchor_idx];
                    softmax_vals[bin] = std::exp(val - max_val);
                    softmax_sum += softmax_vals[bin];
                }

                // Weighted sum
                for (int bin = 0; bin < 16; ++bin) {
                    distances[dir] += (softmax_vals[bin] / softmax_sum) * bin;
                }
            }

            // Get grid coordinates
            float grid_x = meshgrid_[head_idx][anchor_idx * 2 + 0];
            float grid_y = meshgrid_[head_idx][anchor_idx * 2 + 1];

            // Decode box coordinates
            float x1 = (grid_x - distances[0]) * stride - pad.second;
            float y1 = (grid_y - distances[1]) * stride - pad.first;
            float x2 = (grid_x + distances[2]) * stride - pad.second;
            float y2 = (grid_y + distances[3]) * stride - pad.first;

            // Scale to original image size
            x1 = std::max(0.0f, std::min(x1 / gain, static_cast<float>(img_w)));
            y1 = std::max(0.0f, std::min(y1 / gain, static_cast<float>(img_h)));
            x2 = std::max(0.0f, std::min(x2 / gain, static_cast<float>(img_w)));
            y2 = std::max(0.0f, std::min(y2 / gain, static_cast<float>(img_h)));

            // Store detection
            all_boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            all_scores.push_back(max_score);
            all_class_ids.push_back(max_class_id);
        }
    }

    if (all_boxes.empty()) {
        return {};
    }

    // Apply NMS
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(all_boxes, all_scores, confidence_threshold_, iou_threshold_, nms_indices);

    // Format results
    std::vector<std::vector<float>> results;
    for (int idx : nms_indices) {
        std::vector<float> detection = {
            static_cast<float>(all_boxes[idx].x),
            static_cast<float>(all_boxes[idx].y),
            static_cast<float>(all_boxes[idx].width),
            static_cast<float>(all_boxes[idx].height),
            all_scores[idx],
            static_cast<float>(all_class_ids[idx])
        };
        results.push_back(detection);
    }

    return results;
}

} // namespace cnn