//-------------------------------------------------------------------------------------
// Image.cs
//
// Image / ScratchImage types, mirroring DirectXTex's Image and TexMetadata.
//-------------------------------------------------------------------------------------

using System.Runtime.InteropServices;

namespace TexGen;

/// <summary>A single 2D subresource (one mip level / array slice).</summary>
public sealed class Image
{
    public int Width { get; }
    public int Height { get; }
    public DxgiFormat Format { get; }
    public int RowPitch { get; }
    public int SlicePitch { get; }
    /// <summary>Pixel data for this subresource (at least <see cref="SlicePitch"/> bytes).</summary>
    public byte[] Pixels { get; }

    public Image(int width, int height, DxgiFormat format, int rowPitch, int slicePitch, byte[] pixels)
    {
        if (pixels.Length < slicePitch) throw new ArgumentException("Image: pixel buffer too small");
        Width = width;
        Height = height;
        Format = format;
        RowPitch = rowPitch;
        SlicePitch = slicePitch;
        Pixels = pixels;
    }

    /// <summary>Allocate an Image of the given size/format with a zeroed pixel buffer.</summary>
    public static Image Create(DxgiFormat format, int width, int height, CpFlags flags = CpFlags.None)
    {
        var (rowPitch, slicePitch) = Dxgi.ComputePitch(format, width, height, flags);
        return new Image(width, height, format, rowPitch, slicePitch, new byte[slicePitch]);
    }

    /// <summary>Wrap existing tightly-packed pixel bytes as an Image of the given format.</summary>
    public static Image FromPixels(DxgiFormat format, int width, int height, byte[] pixels)
    {
        var (rowPitch, slicePitch) = Dxgi.ComputePitch(format, width, height);
        return new Image(width, height, format, rowPitch, slicePitch, pixels);
    }

    /// <summary>Same bytes, different format tag (e.g. UNORM vs UNORM_SRGB).</summary>
    public Image Reinterpret(DxgiFormat format) => new(Width, Height, format, RowPitch, SlicePitch, Pixels);

    /// <summary>The pixel data viewed as 32-bit floats (R32G32B32A32_FLOAT images).</summary>
    public Span<float> Floats => MemoryMarshal.Cast<byte, float>(Pixels.AsSpan(0, SlicePitch));

    public bool IsRgba8 => Format is DxgiFormat.R8G8B8A8_UNORM or DxgiFormat.R8G8B8A8_UNORM_SRGB;
}

public enum TexDimension : uint
{
    Texture1D = 2,
    Texture2D = 3,
    Texture3D = 4,
}

/// <summary>Describes the shape of a texture: mips, array, dimension.</summary>
public sealed record TexMetadata(
    int Width,
    int Height,
    int Depth,
    int ArraySize,
    int MipLevels,
    uint MiscFlags,
    uint MiscFlags2,
    DxgiFormat Format,
    TexDimension Dimension)
{
    /// <summary>TEX_MISC_FLAG (subset).</summary>
    public const uint TexMiscTextureCube = 0x4;
}

/// <summary>
/// A container holding one or more images plus shape metadata. The 2D, single-array
/// case (the common texconv path) is what the pipeline produces.
/// </summary>
public sealed class ScratchImage
{
    public TexMetadata Metadata { get; }
    public IReadOnlyList<Image> Images { get; }

    private ScratchImage(TexMetadata metadata, IReadOnlyList<Image> images)
    {
        Metadata = metadata;
        Images = images;
    }

    /// <summary>Single 2D image, mip 0 only.</summary>
    public static ScratchImage From2D(Image image) => FromMipChain([image]);

    /// <summary>A 2D mip chain (array size 1). <c>images[0]</c> is the base level.</summary>
    public static ScratchImage FromMipChain(IReadOnlyList<Image> images)
    {
        var b = images[0];
        return new ScratchImage(
            new TexMetadata(b.Width, b.Height, 1, 1, images.Count, 0, 0, b.Format, TexDimension.Texture2D),
            images);
    }

    /// <summary>Construct directly from metadata + image list (e.g. when parsing DDS).</summary>
    public static ScratchImage FromImages(TexMetadata metadata, IReadOnlyList<Image> images) => new(metadata, images);

    public Image BaseImage => Images[0];

    /// <summary>Number of mip levels for a full chain down to 1x1.</summary>
    public static int CountMips(int width, int height)
    {
        int mips = 1;
        while (width > 1 || height > 1)
        {
            width = Math.Max(1, width >> 1);
            height = Math.Max(1, height >> 1);
            mips++;
        }
        return mips;
    }
}
