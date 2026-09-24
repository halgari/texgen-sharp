//-------------------------------------------------------------------------------------
// Decoders/BcnDecoder.cs
//
// Reference CPU decoders for BC1/BC2/BC3/BC4/BC5, used for verification, preview and
// DDS input. Matches the D3D/DirectXTex decode (and thus the BcnEncoder palettes).
//-------------------------------------------------------------------------------------

namespace TexGen.Decoders;

public static class BcnDecoder
{
    public static bool Supports(DxgiFormat format) => format is
        DxgiFormat.BC1_UNORM or DxgiFormat.BC1_UNORM_SRGB or DxgiFormat.BC2_UNORM or DxgiFormat.BC2_UNORM_SRGB
        or DxgiFormat.BC3_UNORM or DxgiFormat.BC3_UNORM_SRGB or DxgiFormat.BC4_UNORM or DxgiFormat.BC5_UNORM;

    private static (int R, int G, int B) Expand565(int w)
    {
        int r = (w >> 11) & 31, g = (w >> 5) & 63, b = w & 31;
        return ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2));
    }

    // JS Math.round semantics (half away from zero for positives) to match texconv-js exactly.
    private static int Round(double v) => (int)Math.Floor(v + 0.5);

    /// <summary>Decode a BC1 color block (8 bytes) into 16 RGBA pixels.</summary>
    private static void DecodeColorBlock(ReadOnlySpan<byte> b, Span<byte> outPx, bool forceFourColor)
    {
        int c0 = b[0] | (b[1] << 8);
        int c1 = b[2] | (b[3] << 8);
        var e0 = Expand565(c0);
        var e1 = Expand565(c1);
        Span<byte> pal = stackalloc byte[16];
        pal[0] = (byte)e0.R; pal[1] = (byte)e0.G; pal[2] = (byte)e0.B; pal[3] = 255;
        pal[4] = (byte)e1.R; pal[5] = (byte)e1.G; pal[6] = (byte)e1.B; pal[7] = 255;
        if (c0 > c1 || forceFourColor)
        {
            pal[8] = (byte)Round((2 * e0.R + e1.R) / 3.0);
            pal[9] = (byte)Round((2 * e0.G + e1.G) / 3.0);
            pal[10] = (byte)Round((2 * e0.B + e1.B) / 3.0);
            pal[11] = 255;
            pal[12] = (byte)Round((e0.R + 2 * e1.R) / 3.0);
            pal[13] = (byte)Round((e0.G + 2 * e1.G) / 3.0);
            pal[14] = (byte)Round((e0.B + 2 * e1.B) / 3.0);
            pal[15] = 255;
        }
        else
        {
            pal[8] = (byte)Round((e0.R + e1.R) / 2.0);
            pal[9] = (byte)Round((e0.G + e1.G) / 2.0);
            pal[10] = (byte)Round((e0.B + e1.B) / 2.0);
            pal[11] = 255;
            // pal[12..15] stays transparent black
        }
        uint idx = (uint)(b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24));
        for (int p = 0; p < 16; p++)
        {
            int e = (int)((idx >> (p * 2)) & 3);
            pal.Slice(e * 4, 4).CopyTo(outPx.Slice(p * 4, 4));
        }
    }

    /// <summary>Decode a BC4 block (8 bytes) into 16 channel values [0,255].</summary>
    private static void DecodeAlphaBlock(ReadOnlySpan<byte> b, Span<byte> outV)
    {
        int r0 = b[0], r1 = b[1];
        Span<double> pal = stackalloc double[8];
        pal[0] = r0;
        pal[1] = r1;
        if (r0 > r1)
        {
            for (int k = 1; k < 7; k++) pal[k + 1] = (r0 * (7 - k) + r1 * k) / 7.0;
        }
        else
        {
            for (int k = 1; k < 5; k++) pal[k + 1] = (r0 * (5 - k) + r1 * k) / 5.0;
            pal[6] = 0;
            pal[7] = 255;
        }
        // 48 bits of 3-bit indices begin at bit 16 of the 64-bit block.
        ulong bits = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(b);
        for (int p = 0; p < 16; p++) outV[p] = (byte)Round(pal[(int)((bits >> (16 + p * 3)) & 7)]);
    }

    private static int BlockSize(DxgiFormat format) => format switch
    {
        DxgiFormat.BC1_UNORM or DxgiFormat.BC1_UNORM_SRGB or DxgiFormat.BC4_UNORM => 8,
        _ when Supports(format) => 16,
        _ => throw new NotSupportedException($"BcnDecoder: unsupported format {format}"),
    };

    private static void DecodeBlock(DxgiFormat format, ReadOnlySpan<byte> b, Span<byte> outPx)
    {
        Span<byte> a = stackalloc byte[16];
        switch (format)
        {
            case DxgiFormat.BC1_UNORM or DxgiFormat.BC1_UNORM_SRGB:
                DecodeColorBlock(b, outPx, false);
                break;
            case DxgiFormat.BC2_UNORM or DxgiFormat.BC2_UNORM_SRGB:
                DecodeColorBlock(b[8..], outPx, true);
                for (int p = 0; p < 16; p++)
                {
                    int nib = (b[p >> 1] >> ((p & 1) * 4)) & 0xf;
                    outPx[p * 4 + 3] = (byte)((nib << 4) | nib);
                }
                break;
            case DxgiFormat.BC3_UNORM or DxgiFormat.BC3_UNORM_SRGB:
                DecodeColorBlock(b[8..], outPx, true);
                DecodeAlphaBlock(b, a);
                for (int p = 0; p < 16; p++) outPx[p * 4 + 3] = a[p];
                break;
            case DxgiFormat.BC4_UNORM:
                DecodeAlphaBlock(b, a);
                for (int p = 0; p < 16; p++)
                {
                    outPx[p * 4] = a[p];
                    outPx[p * 4 + 1] = 0;
                    outPx[p * 4 + 2] = 0;
                    outPx[p * 4 + 3] = 255;
                }
                break;
            default: // BC5
                DecodeAlphaBlock(b, a);
                Span<byte> g = stackalloc byte[16];
                DecodeAlphaBlock(b[8..], g);
                for (int p = 0; p < 16; p++)
                {
                    outPx[p * 4] = a[p];
                    outPx[p * 4 + 1] = g[p];
                    outPx[p * 4 + 2] = 0;
                    outPx[p * 4 + 3] = 255;
                }
                break;
        }
    }

    /// <summary>Decode a BC1/2/3/4/5 image to R8G8B8A8_UNORM.</summary>
    public static Image Decode(Image src)
    {
        int bytes = BlockSize(src.Format);
        var dst = Image.Create(DxgiFormat.R8G8B8A8_UNORM, src.Width, src.Height);
        BlockDecode.Scatter(src, dst, bytes, 4, (block, px) => DecodeBlock(src.Format, block, px));
        return dst;
    }
}

/// <summary>Shared block-walking for the CPU decoders (rows of blocks run in parallel).</summary>
internal static class BlockDecode
{
    public delegate void BlockFn(ReadOnlySpan<byte> block, Span<byte> pixels);

    /// <summary>
    /// Decode every block of <paramref name="src"/> into <paramref name="dst"/>.
    /// <paramref name="pixelBytes"/> is the destination bytes per pixel (4 for RGBA8, 16 for RGBA32F).
    /// </summary>
    public static void Scatter(Image src, Image dst, int blockBytes, int pixelBytes, BlockFn decode)
    {
        int xblocks = Math.Max(1, (src.Width + 3) >> 2);
        int yblocks = Math.Max(1, (src.Height + 3) >> 2);
        Parallel.For(0, yblocks, by =>
        {
            Span<byte> block = stackalloc byte[16 * 16];
            var px = block[..(16 * pixelBytes)];
            for (int bx = 0; bx < xblocks; bx++)
            {
                decode(src.Pixels.AsSpan((by * xblocks + bx) * blockBytes, blockBytes), px);
                for (int py = 0; py < 4; py++)
                {
                    int y = by * 4 + py;
                    if (y >= src.Height) break;
                    for (int pxi = 0; pxi < 4; pxi++)
                    {
                        int x = bx * 4 + pxi;
                        if (x >= src.Width) break;
                        px.Slice((py * 4 + pxi) * pixelBytes, pixelBytes)
                            .CopyTo(dst.Pixels.AsSpan(y * dst.RowPitch + x * pixelBytes, pixelBytes));
                    }
                }
            }
        });
    }
}
