namespace YoloDefectInspector.Models;

public sealed class DefectResult
{
    public string Label { get; init; } = string.Empty;

    public float Confidence { get; init; }

    public float X { get; init; }

    public float Y { get; init; }

    public float Width { get; init; }

    public float Height { get; init; }
}
