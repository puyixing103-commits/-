using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using YoloDefectInspector.Models;

namespace YoloDefectInspector.Services;

public sealed class YoloOnnxDetector : IDisposable
{
    private const int InputWidth = 640;
    private const int InputHeight = 640;
    private readonly string[] _labels;
    private InferenceSession? _session;

    public YoloOnnxDetector(string[] labels)
    {
        _labels = labels;
    }

    public void LoadModel(string modelPath)
    {
        _session?.Dispose();
        try
        {
            _session = new InferenceSession(modelPath);
        }
        catch (TypeInitializationException ex)
        {
            throw BuildNativeRuntimeException(ex);
        }
        catch (DllNotFoundException ex)
        {
            throw BuildNativeRuntimeException(ex);
        }
        catch (BadImageFormatException ex)
        {
            throw BuildNativeRuntimeException(ex);
        }
    }

    private static InvalidOperationException BuildNativeRuntimeException(Exception ex)
    {
        var message =
            "ONNX Runtime 原生库加载失败。请确认：\n" +
            "1) 应用以 x64 运行（不要用 x86）；\n" +
            "2) 已安装 Microsoft Visual C++ 2015-2022 Redistributable (x64)；\n" +
            "3) ONNX Runtime NuGet 包与应用位数一致。\n" +
            $"原始错误：{GetInnermostMessage(ex)}";

        return new InvalidOperationException(message, ex);
    }

    private static string GetInnermostMessage(Exception ex)
    {
        var current = ex;
        while (current.InnerException is not null)
        {
            current = current.InnerException;
        }

        return current is Win32Exception win32
            ? $"{win32.Message} (Win32Error={win32.NativeErrorCode})"
            : current.Message;
    }

    public IReadOnlyList<DefectResult> Detect(string imagePath, float confThreshold, float iouThreshold)
    {
        if (_session is null)
        {
            throw new InvalidOperationException("请先加载 ONNX 模型。");
        }

        using var image = new Bitmap(imagePath);
        var scaleX = (float)image.Width / InputWidth;
        var scaleY = (float)image.Height / InputHeight;

        var inputName = _session.InputMetadata.Keys.First();
        var inputMeta = _session.InputMetadata[inputName];
        NamedOnnxValue inputValue = inputMeta.ElementDataType switch
        {
            TensorElementType.Float => NamedOnnxValue.CreateFromTensor(inputName, CreateInputTensor(image)),
            TensorElementType.Float16 => NamedOnnxValue.CreateFromTensor(inputName, CreateInputTensorFloat16(image)),
            _ => throw new InvalidOperationException(
                $"模型输入类型不受支持：{inputMeta.ElementDataType}。当前仅支持 Float/Float16。")
        };

        using var inputs = new List<NamedOnnxValue>
        {
            inputValue
        };

        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
        try
        {
            results = _session.Run(inputs);
        }
        catch (OnnxRuntimeException ex)
        {
            throw new InvalidOperationException($"ONNX 推理执行失败：{ex.Message}", ex);
        }

        using (results)
        {
            var outputMeta = _session.OutputMetadata.Values.FirstOrDefault();
            if (outputMeta is null)
            {
                throw new InvalidOperationException("模型未提供可用输出节点。");
            }

            return outputMeta.ElementDataType switch
            {
                TensorElementType.Float => ParseOutput(
                    results.First().AsTensor<float>(),
                    confThreshold,
                    iouThreshold,
                    scaleX,
                    scaleY),
                TensorElementType.Float16 => ParseOutput(
                    results.First().AsTensor<Float16>(),
                    confThreshold,
                    iouThreshold,
                    scaleX,
                    scaleY),
                _ => throw new InvalidOperationException(
                    $"模型输出类型不受支持：{outputMeta.ElementDataType}。当前仅支持 Float/Float16。")
            };
        }
    }

    private static DenseTensor<float> CreateInputTensor(Bitmap source)
    {
        using var resized = new Bitmap(InputWidth, InputHeight);
        using (var graphics = Graphics.FromImage(resized))
        {
            graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
            graphics.DrawImage(source, 0, 0, InputWidth, InputHeight);
        }

        var tensor = new DenseTensor<float>(new[] { 1, 3, InputHeight, InputWidth });

        for (var y = 0; y < InputHeight; y++)
        {
            for (var x = 0; x < InputWidth; x++)
            {
                var color = resized.GetPixel(x, y);
                tensor[0, 0, y, x] = color.R / 255f;
                tensor[0, 1, y, x] = color.G / 255f;
                tensor[0, 2, y, x] = color.B / 255f;
            }
        }

        return tensor;
    }

    private static DenseTensor<Float16> CreateInputTensorFloat16(Bitmap source)
    {
        using var resized = new Bitmap(InputWidth, InputHeight);
        using (var graphics = Graphics.FromImage(resized))
        {
            graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
            graphics.DrawImage(source, 0, 0, InputWidth, InputHeight);
        }

        var tensor = new DenseTensor<Float16>(new[] { 1, 3, InputHeight, InputWidth });

        for (var y = 0; y < InputHeight; y++)
        {
            for (var x = 0; x < InputWidth; x++)
            {
                var color = resized.GetPixel(x, y);
                tensor[0, 0, y, x] = (Float16)(color.R / 255f);
                tensor[0, 1, y, x] = (Float16)(color.G / 255f);
                tensor[0, 2, y, x] = (Float16)(color.B / 255f);
            }
        }

        return tensor;
    }

    private IReadOnlyList<DefectResult> ParseOutput(
        Tensor<float> output,
        float confThreshold,
        float iouThreshold,
        float scaleX,
        float scaleY)
    {
        return ParseOutputCore(
            output.Dimensions,
            (i, j) => output[0, i, j],
            confThreshold,
            iouThreshold,
            scaleX,
            scaleY);
    }

    private IReadOnlyList<DefectResult> ParseOutput(
        Tensor<Float16> output,
        float confThreshold,
        float iouThreshold,
        float scaleX,
        float scaleY)
    {
        return ParseOutputCore(
            output.Dimensions,
            (i, j) => output[0, i, j].ToFloat(),
            confThreshold,
            iouThreshold,
            scaleX,
            scaleY);
    }

    private IReadOnlyList<DefectResult> ParseOutputCore(
        IReadOnlyList<int> dimensions,
        Func<int, int, float> valueAt,
        float confThreshold,
        float iouThreshold,
        float scaleX,
        float scaleY)
    {
        // 针对常见 YOLOv5 输出维度 [1, 25200, 85]。
        if (dimensions.Count < 3)
        {
            throw new InvalidOperationException("YOLO 输出维度不符合预期。");
        }

        var candidates = new List<DefectResult>();
        var rows = dimensions[1];
        var features = dimensions[2];
        var classes = features - 5;

        for (var i = 0; i < rows; i++)
        {
            var objConfidence = valueAt(i, 4);
            if (objConfidence < confThreshold)
            {
                continue;
            }

            var bestClass = -1;
            var bestClassScore = 0f;
            for (var c = 0; c < classes; c++)
            {
                var classScore = valueAt(i, 5 + c);
                if (classScore > bestClassScore)
                {
                    bestClass = c;
                    bestClassScore = classScore;
                }
            }

            var confidence = objConfidence * bestClassScore;
            if (bestClass < 0 || confidence < confThreshold)
            {
                continue;
            }

            var centerX = valueAt(i, 0) * scaleX;
            var centerY = valueAt(i, 1) * scaleY;
            var width = valueAt(i, 2) * scaleX;
            var height = valueAt(i, 3) * scaleY;

            candidates.Add(new DefectResult
            {
                Label = bestClass < _labels.Length ? _labels[bestClass] : $"class_{bestClass}",
                Confidence = confidence,
                X = centerX - width / 2,
                Y = centerY - height / 2,
                Width = width,
                Height = height
            });
        }

        return ApplyNms(candidates, iouThreshold);
    }

    private static IReadOnlyList<DefectResult> ApplyNms(IReadOnlyList<DefectResult> boxes, float iouThreshold)
    {
        var sorted = boxes.OrderByDescending(d => d.Confidence).ToList();
        var selected = new List<DefectResult>();

        while (sorted.Count > 0)
        {
            var current = sorted[0];
            selected.Add(current);
            sorted.RemoveAt(0);

            sorted = sorted
                .Where(candidate => CalculateIoU(current, candidate) < iouThreshold)
                .ToList();
        }

        return selected;
    }

    private static float CalculateIoU(DefectResult a, DefectResult b)
    {
        var x1 = Math.Max(a.X, b.X);
        var y1 = Math.Max(a.Y, b.Y);
        var x2 = Math.Min(a.X + a.Width, b.X + b.Width);
        var y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

        var intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        if (intersection <= 0)
        {
            return 0;
        }

        var areaA = a.Width * a.Height;
        var areaB = b.Width * b.Height;
        return intersection / (areaA + areaB - intersection);
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
