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
