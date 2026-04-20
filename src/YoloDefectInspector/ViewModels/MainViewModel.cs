using Microsoft.Win32;
using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using YoloDefectInspector.Models;
using YoloDefectInspector.Services;
using YoloDefectInspector.Utilities;

namespace YoloDefectInspector.ViewModels;

public sealed class MainViewModel : ObservableObject
{
    private readonly string[] _labels = { "scratch", "dent", "crack", "stain", "bubble" };
    private readonly YoloOnnxDetector _detector;

    private string _modelPath = string.Empty;
    private string _status = "请先加载 YOLO ONNX 模型。";
    private string _emptyPreviewHint = "未选择待检测图像";
    private float _confidenceThreshold = 0.35f;
    private float _iouThreshold = 0.45f;
    private string? _currentImagePath;
    private ImageSource? _previewImage;

    public MainViewModel()
    {
        _detector = new YoloOnnxDetector(_labels);
        Defects = new ObservableCollection<DefectResult>();

        BrowseModelCommand = new RelayCommand(BrowseModel);
        LoadModelCommand = new RelayCommand(LoadModel, () => File.Exists(ModelPath));
        RunDetectCommand = new RelayCommand(RunDetect);
    }

    public ObservableCollection<DefectResult> Defects { get; }

    public RelayCommand BrowseModelCommand { get; }

    public RelayCommand LoadModelCommand { get; }

    public RelayCommand RunDetectCommand { get; }

    public string ModelPath
    {
        get => _modelPath;
        set
        {
            if (SetProperty(ref _modelPath, value))
            {
                LoadModelCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public float ConfidenceThreshold
    {
        get => _confidenceThreshold;
        set => SetProperty(ref _confidenceThreshold, value);
    }

    public float IouThreshold
    {
        get => _iouThreshold;
        set => SetProperty(ref _iouThreshold, value);
    }

    public string Status
    {
        get => _status;
        set => SetProperty(ref _status, value);
    }

    public string EmptyPreviewHint
    {
        get => _emptyPreviewHint;
        set => SetProperty(ref _emptyPreviewHint, value);
    }

    public ImageSource? PreviewImage
    {
        get => _previewImage;
        set => SetProperty(ref _previewImage, value);
    }

    public string DefectSummary => Defects.Count == 0
        ? "未发现缺陷"
        : $"检测到 {Defects.Count} 个缺陷：{string.Join("、", Defects.GroupBy(d => d.Label).Select(g => $"{g.Key}x{g.Count()}"))}";

    private void BrowseModel()
    {
        var dialog = new OpenFileDialog
        {
            Filter = "ONNX 模型 (*.onnx)|*.onnx"
        };

        if (dialog.ShowDialog() == true)
        {
            ModelPath = dialog.FileName;
            Status = "已选择模型，请点击“加载模型”。";
        }
    }

    private void LoadModel()
    {
        try
        {
            _detector.LoadModel(ModelPath);
            Status = $"模型加载成功：{Path.GetFileName(ModelPath)}";
        }
        catch (Exception ex)
        {
            Status = $"模型加载失败：{ex.Message}";
        }
    }

    private void RunDetect()
    {
        var imageDialog = new OpenFileDialog
        {
            Filter = "图像文件|*.jpg;*.jpeg;*.png;*.bmp"
        };

        if (imageDialog.ShowDialog() != true)
        {
            return;
        }

        _currentImagePath = imageDialog.FileName;
        EmptyPreviewHint = string.Empty;

        try
        {
            var results = _detector.Detect(_currentImagePath, ConfidenceThreshold, IouThreshold);

            Defects.Clear();
            foreach (var defect in results)
            {
                Defects.Add(defect);
            }

            PreviewImage = DetectionOverlayRenderer.Render(_currentImagePath, Defects);
            OnPropertyChanged(nameof(DefectSummary));
            Status = $"检测完成，耗时约 {DateTime.Now:HH:mm:ss}，结果数：{Defects.Count}";
        }
        catch (Exception ex)
        {
            Status = $"检测失败：{ex.Message}";
        }
    }
}

internal static class DetectionOverlayRenderer
{
    public static ImageSource Render(string imagePath, ObservableCollection<DefectResult> defects)
    {
        var bitmap = new BitmapImage();
        bitmap.BeginInit();
        bitmap.UriSource = new Uri(imagePath);
        bitmap.CacheOption = BitmapCacheOption.OnLoad;
        bitmap.EndInit();

        var drawingVisual = new DrawingVisual();
        using var context = drawingVisual.RenderOpen();

        context.DrawImage(bitmap, new System.Windows.Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight));

        foreach (var defect in defects)
        {
            var rect = new System.Windows.Rect(defect.X, defect.Y, defect.Width, defect.Height);
            context.DrawRectangle(null, new Pen(Brushes.Red, 2), rect);

            var text = new FormattedText(
                $"{defect.Label} {defect.Confidence.ToString("P0", CultureInfo.InvariantCulture)}",
                CultureInfo.CurrentCulture,
                System.Windows.FlowDirection.LeftToRight,
                new Typeface("Segoe UI"),
                18,
                Brushes.Yellow,
                1.25);

            context.DrawText(text, new System.Windows.Point(defect.X, Math.Max(0, defect.Y - 24)));
        }

        var target = new RenderTargetBitmap(bitmap.PixelWidth, bitmap.PixelHeight, 96, 96, PixelFormats.Pbgra32);
        target.Render(drawingVisual);
        target.Freeze();
        return target;
    }
}
