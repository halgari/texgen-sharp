//-------------------------------------------------------------------------------------
// Cpu/CpuCompressor.cs
//
// CPU block-compression driver: port of DirectXTex's CompressBC / CompressBC_Parallel
// (DirectXTexCompress.cpp), i.e. the texconv -nogpu path. Blocks are encoded in
// parallel. Partial edge blocks are filled with DirectXTex's replication pattern
// (missing columns/rows copy from index {0,0,0,1}), and UNORM sources feeding the
// SNORM BC4/BC5 encoders are remapped to [-1,1] as ConvertScanline does.
//
// Like the GPU encoders, _SRGB targets encode the stored values unchanged (the
// format tag decides how they are read back); no transfer-function conversion.
//-------------------------------------------------------------------------------------

using System.Numerics;

namespace TexGen.Cpu;

public static class CpuCompressor
{
    /// <summary>Default BC1 alpha cutoff (TEX_THRESHOLD_DEFAULT).</summary>
    public const float DefaultAlphaThreshold = 0.5f;

    public static bool Supports(DxgiFormat format) => format is
        DxgiFormat.BC1_UNORM or DxgiFormat.BC1_UNORM_SRGB or DxgiFormat.BC2_UNORM or DxgiFormat.BC2_UNORM_SRGB
        or DxgiFormat.BC3_UNORM or DxgiFormat.BC3_UNORM_SRGB or DxgiFormat.BC4_UNORM or DxgiFormat.BC4_SNORM
        or DxgiFormat.BC5_UNORM or DxgiFormat.BC5_SNORM or DxgiFormat.BC6H_UF16 or DxgiFormat.BC6H_SF16
        or DxgiFormat.BC7_UNORM or DxgiFormat.BC7_UNORM_SRGB;

    private delegate void EncodeFn(Span<byte> bc, ReadOnlySpan<Vector4> color, float threshold, BcFlags flags);

    private static EncodeFn EncoderFor(DxgiFormat format) => format switch
    {
        DxgiFormat.BC1_UNORM or DxgiFormat.BC1_UNORM_SRGB => BcCodec.EncodeBC1,
        DxgiFormat.BC2_UNORM or DxgiFormat.BC2_UNORM_SRGB => (b, c, _, f) => BcCodec.EncodeBC2(b, c, f),
        DxgiFormat.BC3_UNORM or DxgiFormat.BC3_UNORM_SRGB => (b, c, _, f) => BcCodec.EncodeBC3(b, c, f),
        DxgiFormat.BC4_UNORM => (b, c, _, f) => BcCodec.EncodeBC4U(b, c, f),
        DxgiFormat.BC4_SNORM => (b, c, _, f) => BcCodec.EncodeBC4S(b, c, f),
        DxgiFormat.BC5_UNORM => (b, c, _, f) => BcCodec.EncodeBC5U(b, c, f),
        DxgiFormat.BC5_SNORM => (b, c, _, f) => BcCodec.EncodeBC5S(b, c, f),
        DxgiFormat.BC6H_UF16 => (b, c, _, f) => Bc6hBc7Codec.EncodeBC6HU(b, c, f),
        DxgiFormat.BC6H_SF16 => (b, c, _, f) => Bc6hBc7Codec.EncodeBC6HS(b, c, f),
        DxgiFormat.BC7_UNORM or DxgiFormat.BC7_UNORM_SRGB => (b, c, _, f) => Bc6hBc7Codec.EncodeBC7(b, c, f),
        _ => throw new NotSupportedException($"CpuCompressor: unsupported format {format}"),
    };

    /// <summary>
    /// Compress an R8G8B8A8(_SRGB) or R32G32B32A32_FLOAT image to a BC format on the CPU.
    /// </summary>
    /// <param name="alphaThreshold">BC1 only: alpha below this becomes transparent (texconv -at).</param>
    public static Image Compress(Image src, DxgiFormat format, BcFlags flags = BcFlags.None,
        float alphaThreshold = DefaultAlphaThreshold)
    {
        if (!src.IsRgba8 && src.Format != DxgiFormat.R32G32B32A32_FLOAT)
            throw new ArgumentException($"CpuCompressor: source must be R8G8B8A8(_SRGB) or R32G32B32A32_FLOAT; got {src.Format}");
        var encode = EncoderFor(format);
        bool toSigned = src.IsRgba8 && format is DxgiFormat.BC4_SNORM or DxgiFormat.BC5_SNORM;

        var dst = Image.Create(format, src.Width, src.Height);
        int blockBytes = Dxgi.BlockBytes(format);
        int xblocks = Math.Max(1, (src.Width + 3) >> 2);
        int yblocks = Math.Max(1, (src.Height + 3) >> 2);

        Parallel.For(0, yblocks, by =>
        {
            Span<Vector4> temp = stackalloc Vector4[16];
            for (int bx = 0; bx < xblocks; bx++)
            {
                LoadBlock(src, bx * 4, by * 4, temp);
                if (toSigned)
                {
                    for (int i = 0; i < 16; i++) temp[i] = temp[i] * 2f - Vector4.One;
                }
                encode(dst.Pixels.AsSpan((by * xblocks + bx) * blockBytes, blockBytes), temp, alphaThreshold, flags);
            }
        });
        return dst;
    }

    // Source index for a missing column/row of a partial block (DirectXTex's uSrc).
    private static ReadOnlySpan<byte> ReplicateSrc => [0, 0, 0, 1];

    /// <summary>Load the 4x4 block at (x, y) as floats, replicating pixels for partial blocks.</summary>
    private static void LoadBlock(Image src, int x, int y, Span<Vector4> temp)
    {
        int pw = Math.Min(4, src.Width - x);
        int ph = Math.Min(4, src.Height - y);
        if (src.IsRgba8)
        {
            for (int t = 0; t < ph; t++)
            {
                var row = src.Pixels.AsSpan((y + t) * src.RowPitch + x * 4, pw * 4);
                for (int s = 0; s < pw; s++)
                    temp[t * 4 + s] = new Vector4(row[s * 4], row[s * 4 + 1], row[s * 4 + 2], row[s * 4 + 3]) * (1f / 255f);
            }
        }
        else
        {
            var floats = src.Floats;
            for (int t = 0; t < ph; t++)
            {
                int o = ((y + t) * src.Width + x) * 4;
                for (int s = 0; s < pw; s++, o += 4)
                    temp[t * 4 + s] = new Vector4(floats[o], floats[o + 1], floats[o + 2], floats[o + 3]);
            }
        }

        if (pw < 4)
        {
            for (int t = 0; t < ph; t++)
                for (int s = pw; s < 4; s++)
                    temp[t * 4 + s] = temp[t * 4 + ReplicateSrc[s]];
        }
        if (ph < 4)
        {
            for (int t = ph; t < 4; t++)
                for (int s = 0; s < 4; s++)
                    temp[t * 4 + s] = temp[ReplicateSrc[t] * 4 + s];
        }
    }
}
