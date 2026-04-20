# YOLO 缺陷检测上位机（WPF / .NET Framework）

这是一个基于 **WPF + C# + ONNX Runtime** 的缺陷检测上位机示例，适配 YOLO 系列导出的 ONNX 模型（默认解析 YOLOv5 常见输出格式）。

## 功能

- 加载 `.onnx` 模型。
- 选择本地图片并执行缺陷检测。
- 在界面上显示框选结果（类别、置信度、坐标）。
- 输出缺陷统计汇总。

## 运行环境

- Visual Studio 2022
- .NET Framework 4.8（`net48`）

## 快速开始

1. 打开 `YoloDefectInspector.sln`。
2. 还原 NuGet 包。
3. 启动后点击“浏览”选择 ONNX 模型并加载。
4. 点击“开始检测”选择待检图片。

## 说明

- 当前示例将输入固定 resize 到 `640x640`。
- 若你的模型输出维度与 YOLOv5 不一致，请修改 `Services/YoloOnnxDetector.cs` 中 `ParseOutput` 逻辑。
- 类别名称可在 `ViewModels/MainViewModel.cs` 中 `_labels` 数组修改。

## 模型下载建议（ONNX）

本项目默认按 **YOLOv5 检测头输出** 解析，建议先用官方预训练 ONNX 做联调：

- 推荐：`yolov5s.onnx`（速度和精度更平衡，适合先跑通流程）
- 版本线：`YOLOv5 v7.0`
- 下载链接（官方 Release 直链）：
  - https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx
  - https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx
  - https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.onnx

如果你后续要做实际缺陷检测，建议用你自己的缺陷数据训练后再导出 ONNX（保持 `640x640` 输入更省心）。
