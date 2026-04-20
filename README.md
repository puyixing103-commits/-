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

## 常见问题

### 模型加载失败：“Microsoft.ML.OnnxRuntime.NativeMethods”的类型初始值设定项引发异常

这通常是 **ONNX Runtime 原生依赖未正确加载** 引起的，按下面顺序排查：

1. 使用 `x64` 运行程序（本项目已改为 `x64` 平台目标）。
2. 安装 **Microsoft Visual C++ 2015-2022 Redistributable (x64)**。
3. 确认未使用 `x86` 进程启动（`x86` 与 ONNX Runtime x64 原生库不兼容）。

若仍失败，可查看界面中的“原始错误”信息进一步定位缺失 DLL。

### 检测失败：[ErrorCode:InvalidArgument] ... expected: Float16

这是输入/输出张量精度不匹配：当前示例按 **Float32** 预处理与解析，但你加载的模型是 **FP16(Float16)**。

处理方式（二选一）：

1. **推荐**：重新导出 FP32 ONNX（不要 half/FP16）。
2. 代码侧改造为完整 FP16 预处理与后处理（本示例暂未实现）。
