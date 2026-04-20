using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
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
        _session = new InferenceSession(modelPath);
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

        var tensor = CreateInputTensor(image);
        var inputName = _session.InputMetadata.Keys.First();
        using var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };

        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();
        var detections = ParseOutput(output, confThreshold, iouThreshold, scaleX, scaleY);

        return detections;
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

    private IReadOnlyList<DefectResult> ParseOutput(
        Tensor<float> output,
        float confThreshold,
        float iouThreshold,
        float scaleX,
        float scaleY)
    {
        // 针对常见 YOLOv5 输出维度 [1, 25200, 85]。
        var dimensions = output.Dimensions;
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
            var objConfidence = output[0, i, 4];
            if (objConfidence < confThreshold)
            {
                continue;
            }

            var bestClass = -1;
            var bestClassScore = 0f;
            for (var c = 0; c < classes; c++)
            {
                var classScore = output[0, i, 5 + c];
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

            var centerX = output[0, i, 0] * scaleX;
            var centerY = output[0, i, 1] * scaleY;
            var width = output[0, i, 2] * scaleX;
            var height = output[0, i, 3] * scaleY;

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
