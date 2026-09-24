//-------------------------------------------------------------------------------------
// PixelConvert.cs
//
// CPU conversions between the pipeline's working formats (R8G8B8A8 and linear
// R32G32B32A32_FLOAT) and the uncompressed DXGI formats texconv can write:
// RGBA8/BGRA8/BGRX8, RGBA16 UNORM/FLOAT, RGBA32 FLOAT, R32 FLOAT, R16/R8/RG8 UNORM.
// Used for uncompressed -f targets and for reading uncompressed DDS input.
//-------------------------------------------------------------------------------------

using System.Buffers.Binary;

namespace TexGen;

public static class PixelConvert
{
    /// <summary>True for uncompressed formats this library can write from a working image.</summary>
    public static bool CanEncode(DxgiFormat fmt) => fmt is
        DxgiFormat.R8G8B8A8_UNORM or DxgiFormat.R8G8B8A8_UNORM_SRGB or DxgiFormat.B8G8R8A8_UNORM
        or DxgiFormat.B8G8R8A8_UNORM_SRGB or DxgiFormat.B8G8R8X8_UNORM or DxgiFormat.R16G16B16A16_UNORM
        or DxgiFormat.R16G16B16A16_FLOAT or DxgiFormat.R32G32B32A32_FLOAT or DxgiFormat.R32_FLOAT
        or DxgiFormat.R16_UNORM or DxgiFormat.R8_UNORM or DxgiFormat.R8G8_UNORM;

    /// <summary>Formats whose natural working representation is linear float.</summary>
    public static bool IsFloatFormat(DxgiFormat fmt) =>
        fmt is DxgiFormat.R16G16B16A16_FLOAT or DxgiFormat.R32G32B32A32_FLOAT or DxgiFormat.R32_FLOAT;

    /// <summary>
    /// Convert a working image (R8G8B8A8 family or R32G32B32A32_FLOAT) to an
    /// uncompressed target format. Byte-exact for 8-bit to 8-bit targets.
    /// </summary>
    public static Image Encode(Image src, DxgiFormat target)
    {
        if (!src.IsRgba8 && src.Format != DxgiFormat.R32G32B32A32_FLOAT)
            throw new ArgumentException($"PixelConvert: working image must be RGBA8 or RGBA32F; got {src.Format}");
        if (!CanEncode(target)) throw new NotSupportedException($"PixelConvert: cannot encode {target}");
        if (target == src.Format) return src;

        // Same bytes, different color-space tag.
        if (src.IsRgba8 && target is DxgiFormat.R8G8B8A8_UNORM or DxgiFormat.R8G8B8A8_UNORM_SRGB)
            return src.Reinterpret(target);

        var dst = Image.Create(target, src.Width, src.Height);
        int bpp = Dxgi.BitsPerPixel(target) / 8;
        bool srcFloat = !src.IsRgba8;
        Parallel.For(0, src.Height, y =>
        {
            Span<float> px = stackalloc float[4];
            for (int x = 0; x < src.Width; x++)
            {
                var o = dst.Pixels.AsSpan(y * dst.RowPitch + x * bpp, bpp);
                if (!srcFloat)
                {
                    var s = src.Pixels.AsSpan(y * src.RowPitch + x * 4, 4);
                    // 8-bit -> 8-bit targets stay byte-exact.
                    switch (target)
                    {
                        case DxgiFormat.B8G8R8A8_UNORM or DxgiFormat.B8G8R8A8_UNORM_SRGB:
                            o[0] = s[2]; o[1] = s[1]; o[2] = s[0]; o[3] = s[3];
                            continue;
                        case DxgiFormat.B8G8R8X8_UNORM:
                            o[0] = s[2]; o[1] = s[1]; o[2] = s[0]; o[3] = 255;
                            continue;
                        case DxgiFormat.R8_UNORM:
                            o[0] = s[0];
                            continue;
                        case DxgiFormat.R8G8_UNORM:
                            o[0] = s[0]; o[1] = s[1];
                            continue;
                    }
                    for (int c = 0; c < 4; c++) px[c] = s[c] / 255f;
                }
                else
                {
                    src.Floats.Slice((y * src.Width + x) * 4, 4).CopyTo(px);
                }
                WriteFloatPixel(target, px, o);
            }
        });
        return dst;
    }

    private static byte Unorm8(float v) => (byte)MathF.Floor(Math.Clamp(v, 0f, 1f) * 255f + 0.5f);

    private static ushort Unorm16(float v) => (ushort)MathF.Floor(Math.Clamp(v, 0f, 1f) * 65535f + 0.5f);

    private static void WriteFloatPixel(DxgiFormat target, ReadOnlySpan<float> px, Span<byte> o)
    {
        switch (target)
        {
            case DxgiFormat.R8G8B8A8_UNORM or DxgiFormat.R8G8B8A8_UNORM_SRGB:
                for (int c = 0; c < 4; c++) o[c] = Unorm8(px[c]);
                break;
            case DxgiFormat.B8G8R8A8_UNORM or DxgiFormat.B8G8R8A8_UNORM_SRGB:
                o[0] = Unorm8(px[2]); o[1] = Unorm8(px[1]); o[2] = Unorm8(px[0]); o[3] = Unorm8(px[3]);
                break;
            case DxgiFormat.B8G8R8X8_UNORM:
                o[0] = Unorm8(px[2]); o[1] = Unorm8(px[1]); o[2] = Unorm8(px[0]); o[3] = 255;
                break;
            case DxgiFormat.R16G16B16A16_UNORM:
                for (int c = 0; c < 4; c++) BinaryPrimitives.WriteUInt16LittleEndian(o[(c * 2)..], Unorm16(px[c]));
                break;
            case DxgiFormat.R16G16B16A16_FLOAT:
                for (int c = 0; c < 4; c++) BinaryPrimitives.WriteHalfLittleEndian(o[(c * 2)..], (Half)px[c]);
                break;
            case DxgiFormat.R32G32B32A32_FLOAT:
                for (int c = 0; c < 4; c++) BinaryPrimitives.WriteSingleLittleEndian(o[(c * 4)..], px[c]);
                break;
            case DxgiFormat.R32_FLOAT:
                BinaryPrimitives.WriteSingleLittleEndian(o, px[0]);
                break;
            case DxgiFormat.R16_UNORM:
                BinaryPrimitives.WriteUInt16LittleEndian(o, Unorm16(px[0]));
                break;
            case DxgiFormat.R8_UNORM:
                o[0] = Unorm8(px[0]);
                break;
            case DxgiFormat.R8G8_UNORM:
                o[0] = Unorm8(px[0]); o[1] = Unorm8(px[1]);
                break;
        }
    }

    /// <summary>
    /// Expand an uncompressed image of any supported format to a working image:
    /// R8G8B8A8_UNORM for 8-bit formats, R32G32B32A32_FLOAT for 16/32-bit formats.
    /// Missing channels decode as in D3D (G/B = 0, A = 1).
    /// </summary>
    public static Image Decode(Image src)
    {
        if (src.IsRgba8 || src.Format == DxgiFormat.R32G32B32A32_FLOAT) return src;
        if (!CanEncode(src.Format)) throw new NotSupportedException($"PixelConvert: cannot decode {src.Format}");

        bool toFloat = src.Format is DxgiFormat.R16G16B16A16_UNORM or DxgiFormat.R16G16B16A16_FLOAT
            or DxgiFormat.R32_FLOAT or DxgiFormat.R16_UNORM;
        var working = toFloat ? DxgiFormat.R32G32B32A32_FLOAT
            : Dxgi.IsSrgb(src.Format) ? DxgiFormat.R8G8B8A8_UNORM_SRGB : DxgiFormat.R8G8B8A8_UNORM;
        var dst = Image.Create(working, src.Width, src.Height);
        int bpp = Dxgi.BitsPerPixel(src.Format) / 8;
        Parallel.For(0, src.Height, y =>
        {
            for (int x = 0; x < src.Width; x++)
            {
                var s = src.Pixels.AsSpan(y * src.RowPitch + x * bpp, bpp);
                if (toFloat)
                {
                    var f = dst.Floats.Slice((y * src.Width + x) * 4, 4);
                    switch (src.Format)
                    {
                        case DxgiFormat.R16G16B16A16_UNORM:
                            for (int c = 0; c < 4; c++) f[c] = BinaryPrimitives.ReadUInt16LittleEndian(s[(c * 2)..]) / 65535f;
                            break;
                        case DxgiFormat.R16G16B16A16_FLOAT:
                            for (int c = 0; c < 4; c++) f[c] = (float)BinaryPrimitives.ReadHalfLittleEndian(s[(c * 2)..]);
                            break;
                        case DxgiFormat.R32_FLOAT:
                            f[0] = BinaryPrimitives.ReadSingleLittleEndian(s); f[1] = 0; f[2] = 0; f[3] = 1;
                            break;
                        default: // R16_UNORM
                            f[0] = BinaryPrimitives.ReadUInt16LittleEndian(s) / 65535f; f[1] = 0; f[2] = 0; f[3] = 1;
                            break;
                    }
                }
                else
                {
                    var o = dst.Pixels.AsSpan(y * dst.RowPitch + x * 4, 4);
                    switch (src.Format)
                    {
                        case DxgiFormat.B8G8R8A8_UNORM or DxgiFormat.B8G8R8A8_UNORM_SRGB:
                            o[0] = s[2]; o[1] = s[1]; o[2] = s[0]; o[3] = s[3];
                            break;
                        case DxgiFormat.B8G8R8X8_UNORM:
                            o[0] = s[2]; o[1] = s[1]; o[2] = s[0]; o[3] = 255;
                            break;
                        case DxgiFormat.R8_UNORM:
                            o[0] = s[0]; o[1] = 0; o[2] = 0; o[3] = 255;
                            break;
                        default: // R8G8_UNORM
                            o[0] = s[0]; o[1] = s[1]; o[2] = 0; o[3] = 255;
                            break;
                    }
                }
            }
        });
        return dst;
    }
}
