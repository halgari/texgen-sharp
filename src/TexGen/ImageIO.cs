//-------------------------------------------------------------------------------------
// ImageIO.cs
//
// Image file decode/encode. PNG/JPEG/BMP/GIF/WebP/ICO/WBMP decode through SkiaSharp
// (the same Skia codecs Chromium's createImageBitmap uses in texconv-js), with the
// same default of color-managing the decode to sRGB. Radiance .hdr and .dds inputs
// are handled natively. DDS input is decoded (BCn/BC6H/BC7 or uncompressed) to a
// working image so it can be re-converted, like texconv does.
//-------------------------------------------------------------------------------------

using SkiaSharp;
using TexGen.Decoders;

namespace TexGen;

public sealed record LoadOptions
{
    /// <summary>
    /// Color-manage the decode to sRGB (the default, matching browsers) so images carrying
    /// a Display-P3 / Adobe-RGB profile decode to correct sRGB values. Set false to read
    /// the raw encoded bytes — only correct when the source is already plain sRGB and you
    /// want byte-exact texconv/WIC-style passthrough.
    /// </summary>
    public bool ColorManage { get; init; } = true;
}

public static class ImageIO
{
    private static readonly string[] SkiaExtensions = [".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".gif", ".webp", ".ico", ".wbmp"];

    /// <summary>File extensions <see cref="Load(string, LoadOptions?)"/> understands.</summary>
    public static IReadOnlyList<string> SupportedExtensions { get; } = [.. SkiaExtensions, ".hdr", ".dds"];

    public static bool IsSupported(string path) =>
        SupportedExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());

    /// <summary>Load an image file as an R8G8B8A8_UNORM or (HDR) R32G32B32A32_FLOAT Image (mip 0).</summary>
    public static Image Load(string path, LoadOptions? options = null) =>
        Load(File.ReadAllBytes(path), Path.GetFileName(path), options);

    /// <summary>Decode image bytes; <paramref name="name"/> is used for format sniffing fallbacks.</summary>
    public static Image Load(byte[] data, string name, LoadOptions? options = null)
    {
        options ??= new LoadOptions();
        if (data.Length >= 4 && data[0] == 'D' && data[1] == 'D' && data[2] == 'S' && data[3] == ' ')
            return DecodeToWorking(Dds.Read(data).BaseImage);
        if (Hdr.IsHdrFile(name) || (data.Length > 2 && data[0] == '#' && data[1] == '?'))
            return Hdr.Load(data);
        return DecodeWithSkia(data, name, options.ColorManage);
    }

    private static Image DecodeWithSkia(byte[] data, string name, bool colorManage)
    {
        using var skData = SKData.CreateCopy(data);
        using var codec = SKCodec.Create(skData)
            ?? throw new InvalidDataException($"{name}: unrecognized or unsupported image format");
        var info = codec.Info;
        // Color-managed: ask Skia for sRGB output (it converts from any embedded profile).
        // Unmanaged: request the source's own color space so no conversion happens.
        var colorSpace = colorManage ? SKColorSpace.CreateSrgb() : info.ColorSpace;
        var target = new SKImageInfo(info.Width, info.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul, colorSpace);
        var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, info.Width, info.Height);
        unsafe
        {
            fixed (byte* p = img.Pixels)
            {
                var result = codec.GetPixels(target, (IntPtr)p, img.RowPitch, new SKCodecOptions(0));
                if (result is not (SKCodecResult.Success or SKCodecResult.IncompleteInput))
                    throw new InvalidDataException($"{name}: decode failed ({result})");
            }
        }
        return img;
    }

    /// <summary>
    /// Decode any supported DXGI image (block-compressed or uncompressed) to a working
    /// image: R8G8B8A8(_SRGB) for LDR formats, R32G32B32A32_FLOAT for HDR ones.
    /// </summary>
    public static Image DecodeToWorking(Image src)
    {
        var f = src.Format;
        Image decoded;
        if (BcnDecoder.Supports(f)) decoded = BcnDecoder.Decode(src);
        else if (f is DxgiFormat.BC7_UNORM or DxgiFormat.BC7_UNORM_SRGB) decoded = Bc7Decoder.Decode(src);
        else if (f is DxgiFormat.BC6H_UF16 or DxgiFormat.BC6H_SF16) return Bc6hDecoder.Decode(src);
        else return PixelConvert.Decode(src);
        // Block decoders emit UNORM; carry the source's sRGB tag through.
        return Dxgi.IsSrgb(f) ? decoded.Reinterpret(DxgiFormat.R8G8B8A8_UNORM_SRGB) : decoded;
    }

    /// <summary>
    /// Save the base level of <paramref name="image"/> as PNG/JPEG/WebP (texconv -ft).
    /// Compressed and float images are decoded / clamped to 8-bit first.
    /// </summary>
    public static void SaveImage(Image image, string path, int quality = 95)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        var format = ext switch
        {
            ".png" => SKEncodedImageFormat.Png,
            ".jpg" or ".jpeg" => SKEncodedImageFormat.Jpeg,
            ".webp" => SKEncodedImageFormat.Webp,
            _ => throw new NotSupportedException($"Output file type '{ext}' is not supported (use dds, png, jpg, webp)"),
        };
        var working = Dxgi.IsCompressed(image.Format) ? DecodeToWorking(image) : PixelConvert.Decode(image);
        if (!working.IsRgba8) working = PixelConvert.Encode(working, DxgiFormat.R8G8B8A8_UNORM);

        var info = new SKImageInfo(working.Width, working.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul, SKColorSpace.CreateSrgb());
        using var bitmap = new SKBitmap();
        unsafe
        {
            fixed (byte* p = working.Pixels)
            {
                bitmap.InstallPixels(info, (IntPtr)p, working.RowPitch);
                using var pixmap = bitmap.PeekPixels();
                using var encoded = pixmap.Encode(format, quality)
                    ?? throw new InvalidOperationException($"Failed to encode {path}");
                using var fs = File.Create(path);
                encoded.SaveTo(fs);
            }
        }
    }
}
