//-------------------------------------------------------------------------------------
// Tests for the DirectXTex CPU BC1-5 encoders (TexGen.Cpu.BcCodec): encode synthetic
// images block by block, decode with the reference decoders, and check PSNR plus the
// codec-specific behaviors (BC1 alpha threshold, dithering, uniform weighting, SNORM).
//-------------------------------------------------------------------------------------

using System.Buffers.Binary;
using System.Numerics;
using TexGen.Cpu;
using TexGen.Decoders;

namespace TexGen.Tests.Cpu;

public class BcCodecTests
{
    private delegate void BlockEncoder(Span<byte> bc, ReadOnlySpan<Vector4> color);

    /// <summary>
    /// Walk the 4x4 blocks of an RGBA8 image (Parallel.For over block rows), filling
    /// partial blocks with DirectXTex's {0,0,0,1} replication, and encode each block.
    /// <paramref name="signed"/> maps UNORM [0,1] to SNORM [-1,1] (x*2-1) like ConvertScanline.
    /// </summary>
    private static Image Encode(Image src, DxgiFormat fmt, BlockEncoder encode, bool signed = false)
    {
        var dst = Image.Create(fmt, src.Width, src.Height);
        int bpb = Dxgi.BlockBytes(fmt);
        int xblocks = Math.Max(1, (src.Width + 3) >> 2);
        int yblocks = Math.Max(1, (src.Height + 3) >> 2);
        int[] uSrc = [0, 0, 0, 1];
        Parallel.For(0, yblocks, by =>
        {
            Span<Vector4> px = stackalloc Vector4[16];
            for (int bx = 0; bx < xblocks; bx++)
            {
                int pw = Math.Min(4, src.Width - bx * 4), ph = Math.Min(4, src.Height - by * 4);
                for (int t = 0; t < 4; t++)
                {
                    for (int s = 0; s < 4; s++)
                    {
                        int sx = s < pw ? s : uSrc[s];
                        int sy = t < ph ? t : uSrc[t];
                        int o = (by * 4 + sy) * src.RowPitch + (bx * 4 + sx) * 4;
                        var v = new Vector4(src.Pixels[o], src.Pixels[o + 1], src.Pixels[o + 2], src.Pixels[o + 3]) / 255f;
                        px[t * 4 + s] = signed ? v * 2f - Vector4.One : v;
                    }
                }
                encode(dst.Pixels.AsSpan((by * xblocks + bx) * bpb, bpb), px);
            }
        });
        return dst;
    }

    private static Image EncodeBcn(Image src, DxgiFormat fmt, BcFlags flags = BcFlags.None, float threshold = 0.5f) => fmt switch
    {
        DxgiFormat.BC1_UNORM => Encode(src, fmt, (b, c) => BcCodec.EncodeBC1(b, c, threshold, flags)),
        DxgiFormat.BC2_UNORM => Encode(src, fmt, (b, c) => BcCodec.EncodeBC2(b, c, flags)),
        DxgiFormat.BC3_UNORM => Encode(src, fmt, (b, c) => BcCodec.EncodeBC3(b, c, flags)),
        DxgiFormat.BC4_UNORM => Encode(src, fmt, (b, c) => BcCodec.EncodeBC4U(b, c, flags)),
        DxgiFormat.BC5_UNORM => Encode(src, fmt, (b, c) => BcCodec.EncodeBC5U(b, c, flags)),
        _ => throw new ArgumentException(fmt.ToString()),
    };

    [Theory]
    [InlineData(DxgiFormat.BC1_UNORM, new[] { 0, 1, 2 }, 30)]
    [InlineData(DxgiFormat.BC2_UNORM, new[] { 0, 1, 2, 3 }, 30)]
    [InlineData(DxgiFormat.BC3_UNORM, new[] { 0, 1, 2, 3 }, 30)]
    [InlineData(DxgiFormat.BC4_UNORM, new[] { 0 }, 40)]
    [InlineData(DxgiFormat.BC5_UNORM, new[] { 0, 1 }, 40)]
    public void Gradient64PsnrAboveThreshold(DxgiFormat fmt, int[] channels, double min)
    {
        // Opaque source for BC1 (its 1-bit alpha would otherwise key the low-alpha half).
        var src = TestImages.Gradient(64, 64, alpha: fmt != DxgiFormat.BC1_UNORM);
        double q = TestImages.Psnr(src, BcnDecoder.Decode(EncodeBcn(src, fmt)), channels);
        Assert.True(q > min, $"{fmt}: {q:F2} dB");
    }

    [Theory]
    [InlineData(DxgiFormat.BC1_UNORM)]
    [InlineData(DxgiFormat.BC3_UNORM)]
    [InlineData(DxgiFormat.BC5_UNORM)]
    public void NonMultipleOf4Sizes(DxgiFormat fmt)
    {
        var src = TestImages.Gradient(30, 18, alpha: false);
        var enc = EncodeBcn(src, fmt);
        Assert.Equal(Dxgi.ComputePitch(fmt, 30, 18).SlicePitch, enc.SlicePitch);
        int[] ch = fmt == DxgiFormat.BC5_UNORM ? [0, 1] : [0, 1, 2];
        double q = TestImages.Psnr(src, BcnDecoder.Decode(enc), ch);
        Assert.True(q > 28, $"{fmt}: {q:F2} dB");
    }

    [Fact]
    public void SolidColorBlocksAreNearExact()
    {
        var src = TestImages.Flat(8, 8, 200, 100, 50, 77);
        var bc3 = BcnDecoder.Decode(EncodeBcn(src, DxgiFormat.BC3_UNORM));
        // 565 quantization of (200,100,50) plus exact alpha.
        Assert.InRange(bc3.Pixels[0], 197, 206);
        Assert.InRange(bc3.Pixels[1], 97, 103);
        Assert.InRange(bc3.Pixels[2], 47, 55);
        Assert.Equal(77, bc3.Pixels[3]);

        var bc4 = BcnDecoder.Decode(EncodeBcn(src, DxgiFormat.BC4_UNORM));
        Assert.InRange(bc4.Pixels[0], 199, 201);
    }

    [Fact]
    public void Bc1AlphaThresholdSelectsThreeColorModeWithTransparentIndex()
    {
        Span<Vector4> px = stackalloc Vector4[16];
        for (int i = 0; i < 16; i++) px[i] = new Vector4(i / 15f, 0.5f, 1 - i / 15f, i < 4 ? 0f : 1f);
        Span<byte> bc = stackalloc byte[8];
        BcCodec.EncodeBC1(bc, px, 0.5f, BcFlags.None);

        ushort c0 = BinaryPrimitives.ReadUInt16LittleEndian(bc);
        ushort c1 = BinaryPrimitives.ReadUInt16LittleEndian(bc[2..]);
        uint bits = BinaryPrimitives.ReadUInt32LittleEndian(bc[4..]);
        Assert.True(c0 <= c1, "3-color mode requires c0 <= c1");
        for (int i = 0; i < 16; i++)
        {
            uint idx = (bits >> (i * 2)) & 3;
            if (i < 4) Assert.Equal(3u, idx);
            else Assert.NotEqual(3u, idx);
        }

        // Fully transparent block -> the canonical all-transparent encoding.
        for (int i = 0; i < 16; i++) px[i] = new Vector4(1, 0, 0, 0);
        BcCodec.EncodeBC1(bc, px, 0.5f, BcFlags.None);
        Assert.Equal(new byte[] { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff }, bc.ToArray());

        // Opaque block never uses 3-color mode.
        for (int i = 0; i < 16; i++) px[i] = new Vector4(i / 15f, 0.2f, 0.7f, 1);
        BcCodec.EncodeBC1(bc, px, 0.5f, BcFlags.None);
        Assert.True(BinaryPrimitives.ReadUInt16LittleEndian(bc) > BinaryPrimitives.ReadUInt16LittleEndian(bc[2..]));
    }

    [Fact]
    public void Bc1WithAlphaGradientDecodesToPunchThroughAlpha()
    {
        var src = TestImages.Gradient(64, 64); // alpha ramps 255 -> 0 left to right
        var dec = BcnDecoder.Decode(EncodeBcn(src, DxgiFormat.BC1_UNORM));
        for (int y = 0; y < 64; y += 7)
        {
            for (int x = 0; x < 64; x++)
            {
                int o = y * src.RowPitch + x * 4;
                byte expected = src.Pixels[o + 3] / 255f < 0.5f ? (byte)0 : (byte)255;
                Assert.Equal(expected, dec.Pixels[o + 3]);
            }
        }
    }

    [Theory]
    [InlineData(DxgiFormat.BC1_UNORM, BcFlags.DitherRgb | BcFlags.DitherA)]
    [InlineData(DxgiFormat.BC2_UNORM, BcFlags.DitherRgb | BcFlags.DitherA)]
    [InlineData(DxgiFormat.BC3_UNORM, BcFlags.DitherRgb | BcFlags.DitherA)]
    [InlineData(DxgiFormat.BC1_UNORM, BcFlags.Uniform)]
    [InlineData(DxgiFormat.BC3_UNORM, BcFlags.Uniform)]
    [InlineData(DxgiFormat.BC3_UNORM, BcFlags.Uniform | BcFlags.DitherRgb | BcFlags.DitherA)]
    public void DitherAndUniformFlagsProduceGoodOutput(DxgiFormat fmt, BcFlags flags)
    {
        var src = TestImages.Gradient(64, 64, alpha: fmt != DxgiFormat.BC1_UNORM);
        var plain = EncodeBcn(src, fmt);
        var flagged = EncodeBcn(src, fmt, flags);
        int[] ch = fmt == DxgiFormat.BC1_UNORM ? [0, 1, 2] : [0, 1, 2, 3];
        double q = TestImages.Psnr(src, BcnDecoder.Decode(flagged), ch);
        Assert.True(q > 30, $"{fmt} {flags}: {q:F2} dB");
        Assert.NotEqual(plain.Pixels, flagged.Pixels); // the flag actually changed the encoding
    }

    [Fact]
    public void EncodingIsDeterministic()
    {
        var src = TestImages.Gradient(64, 64);
        Assert.Equal(EncodeBcn(src, DxgiFormat.BC3_UNORM).Pixels, EncodeBcn(src, DxgiFormat.BC3_UNORM).Pixels);
    }

    //---------------------------------------------------------------------------------
    // SNORM (BC4S / BC5S): small private decoder mirroring BC4_SNORM::DecodeFromIndex.
    //---------------------------------------------------------------------------------

    private static float DecodeSnormTexel(ReadOnlySpan<byte> b, int texel)
    {
        sbyte r0 = (sbyte)b[0], r1 = (sbyte)b[1];
        int index = (int)((BinaryPrimitives.ReadUInt64LittleEndian(b) >> (3 * texel + 16)) & 7);
        float f0 = Math.Max((int)r0, -127) / 127f, f1 = Math.Max((int)r1, -127) / 127f;
        if (index == 0) return f0;
        if (index == 1) return f1;
        if (r0 > r1) return (f0 * (8 - index) + f1 * (index - 1)) / 7f;
        if (index == 6) return -1f;
        if (index == 7) return 1f;
        return (f0 * (6 - index) + f1 * (index - 1)) / 5f;
    }

    [Fact]
    public void Bc4sAndBc5sSignedRoundTrip()
    {
        var src = TestImages.Gradient(64, 64);
        var bc4 = Encode(src, DxgiFormat.BC4_SNORM, (b, c) => BcCodec.EncodeBC4S(b, c, BcFlags.None), signed: true);
        var bc5 = Encode(src, DxgiFormat.BC5_SNORM, (b, c) => BcCodec.EncodeBC5S(b, c, BcFlags.None), signed: true);

        double sse4 = 0, sse5 = 0, max4 = 0;
        int xblocks = 16;
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                int o = y * src.RowPitch + x * 4;
                float r = src.Pixels[o] / 255f * 2 - 1, g = src.Pixels[o + 1] / 255f * 2 - 1;
                int blk = (y / 4) * xblocks + x / 4, texel = (y % 4) * 4 + x % 4;
                float d4 = DecodeSnormTexel(bc4.Pixels.AsSpan(blk * 8, 8), texel) - r;
                float d5r = DecodeSnormTexel(bc5.Pixels.AsSpan(blk * 16, 8), texel) - r;
                float d5g = DecodeSnormTexel(bc5.Pixels.AsSpan(blk * 16 + 8, 8), texel) - g;
                sse4 += d4 * d4;
                sse5 += d5r * d5r + d5g * d5g;
                max4 = Math.Max(max4, Math.Abs(d4));
            }
        }
        // PSNR over the [-1,1] range (peak-to-peak 2).
        double q4 = 10 * Math.Log10(4 / (sse4 / 4096)), q5 = 10 * Math.Log10(4 / (sse5 / 8192));
        Assert.True(q4 > 40, $"BC4S {q4:F2} dB");
        Assert.True(q5 > 40, $"BC5S {q5:F2} dB");
        Assert.True(max4 < 0.05, $"BC4S max error {max4}");
    }

    [Fact]
    public void Bc4sExtremesAreExact()
    {
        Span<Vector4> px = stackalloc Vector4[16];
        for (int i = 0; i < 16; i++) px[i] = new Vector4(i % 2 == 0 ? -1f : 1f, 0, 0, 1);
        Span<byte> bc = stackalloc byte[8];
        BcCodec.EncodeBC4S(bc, px, BcFlags.None);
        for (int i = 0; i < 16; i++) Assert.Equal(i % 2 == 0 ? -1f : 1f, DecodeSnormTexel(bc, i), 5);
    }
}
