//-------------------------------------------------------------------------------------
// Gpu/GpuTexture.cs
//
// Device-resident images. A GpuTexture is either packed RGBA8 (one uint per pixel,
// R in the low byte — the layout of an R8G8B8A8 Image's bytes) or linear RGBA32F
// (one Float4 per pixel). The pipeline uploads the source once, then resizes,
// generates mips, and encodes without round-tripping through host memory; only the
// final encoded blocks are downloaded (GpuBlocks).
//-------------------------------------------------------------------------------------

using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

namespace TexGen.Gpu;

public sealed class GpuTexture : IDisposable
{
    public int Width { get; }
    public int Height { get; }
    /// <summary>Format of the host Image this texture mirrors (RGBA8 family or R32G32B32A32_FLOAT).</summary>
    public DxgiFormat Format { get; }
    public bool IsFloat => Format == DxgiFormat.R32G32B32A32_FLOAT;

    internal MemoryBuffer1D<uint, Stride1D.Dense>? Rgba8 { get; }
    internal MemoryBuffer1D<Float4, Stride1D.Dense>? Rgba32F { get; }

    internal ArrayView<uint> Rgba8View => Rgba8!.View;
    internal ArrayView<Float4> Rgba32FView => Rgba32F!.View;

    private GpuTexture(int width, int height, DxgiFormat format,
        MemoryBuffer1D<uint, Stride1D.Dense>? rgba8, MemoryBuffer1D<Float4, Stride1D.Dense>? rgba32F)
    {
        Width = width;
        Height = height;
        Format = format;
        Rgba8 = rgba8;
        Rgba32F = rgba32F;
    }

    internal static GpuTexture Allocate(Accelerator acc, int width, int height, DxgiFormat format)
    {
        long n = (long)width * height;
        return format == DxgiFormat.R32G32B32A32_FLOAT
            ? new GpuTexture(width, height, format, null, acc.Allocate1D<Float4>(n))
            : new GpuTexture(width, height, format, acc.Allocate1D<uint>(n), null);
    }

    /// <summary>Upload an R8G8B8A8(_SRGB) or R32G32B32A32_FLOAT image.</summary>
    public static GpuTexture Upload(GpuDevice device, Image image)
    {
        if (!image.IsRgba8 && image.Format != DxgiFormat.R32G32B32A32_FLOAT)
            throw new ArgumentException($"GpuTexture: source must be R8G8B8A8(_SRGB) or R32G32B32A32_FLOAT; got {image.Format}");
        var tex = Allocate(device.Accelerator, image.Width, image.Height, image.Format);
        var bytes = image.Pixels.AsSpan(0, image.SlicePitch);
        if (tex.IsFloat) tex.Rgba32FView.CopyFromCPU(MemoryMarshal.Cast<byte, Float4>(bytes));
        else tex.Rgba8View.CopyFromCPU(MemoryMarshal.Cast<byte, uint>(bytes));
        return tex;
    }

    /// <summary>Copy the texture back into a host Image of the same format.</summary>
    public Image Download()
    {
        var img = Image.Create(Format, Width, Height);
        var bytes = img.Pixels.AsSpan(0, img.SlicePitch);
        if (IsFloat) Rgba32FView.CopyToCPU(MemoryMarshal.Cast<byte, Float4>(bytes));
        else Rgba8View.CopyToCPU(MemoryMarshal.Cast<byte, uint>(bytes));
        return img;
    }

    public void Dispose()
    {
        Rgba8?.Dispose();
        Rgba32F?.Dispose();
    }
}

/// <summary>Encoded BC blocks still resident on the device (block-raster order).</summary>
public sealed class GpuBlocks : IDisposable
{
    public int Width { get; }
    public int Height { get; }
    public DxgiFormat Format { get; }
    internal MemoryBuffer1D<uint, Stride1D.Dense> Buffer { get; }

    internal GpuBlocks(int width, int height, DxgiFormat format, MemoryBuffer1D<uint, Stride1D.Dense> buffer)
    {
        Width = width;
        Height = height;
        Format = format;
        Buffer = buffer;
    }

    /// <summary>Download into a compressed host Image (buffer pitch == image row pitch).</summary>
    public Image Download()
    {
        var img = Image.Create(Format, Width, Height);
        var words = MemoryMarshal.Cast<byte, uint>(img.Pixels.AsSpan(0, img.SlicePitch));
        ((ArrayView<uint>)Buffer.View).SubView(0, words.Length).CopyToCPU(words);
        return img;
    }

    public void Dispose() => Buffer.Dispose();
}
