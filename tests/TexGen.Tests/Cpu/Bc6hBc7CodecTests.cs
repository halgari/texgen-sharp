//-------------------------------------------------------------------------------------
// Tests for the DirectXTex CPU BC6H/BC7 codec port: encode synthetic images block by
// block, decode with the reference decoders, and check quality and block validity.
//-------------------------------------------------------------------------------------

using System.Numerics;
using TexGen.Cpu;
using TexGen.Decoders;

namespace TexGen.Tests.Cpu;

public class Bc6hBc7CodecTests
{
    private delegate void BlockEncoder(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags);

    /// <summary>
    /// Encode an image with a per-block codec, replicating DirectXTex's partial-block
    /// handling (missing pixels copy from {0,0,0,1}[s]).
    /// </summary>
    private static Image EncodeImage(Image src, DxgiFormat format, BlockEncoder encode, BcFlags flags = BcFlags.None)
    {
        var dst = Image.Create(format, src.Width, src.Height);
        int xblocks = Math.Max(1, (src.Width + 3) >> 2);
        int yblocks = Math.Max(1, (src.Height + 3) >> 2);
        bool isFloat = src.Format == DxgiFormat.R32G32B32A32_FLOAT;

        Parallel.For(0, yblocks, by =>
        {
            Span<Vector4> px = stackalloc Vector4[16];
            ReadOnlySpan<int> rep = [0, 0, 0, 1];
            for (int bx = 0; bx < xblocks; bx++)
            {
                int pw = Math.Min(4, src.Width - bx * 4);
                int ph = Math.Min(4, src.Height - by * 4);
                for (int t = 0; t < ph; t++)
                {
                    for (int s = 0; s < pw; s++)
                    {
                        int x = bx * 4 + s, y = by * 4 + t;
                        if (isFloat)
                        {
                            var f = src.Floats.Slice((y * src.Width + x) * 4, 4);
                            px[t * 4 + s] = new Vector4(f[0], f[1], f[2], f[3]);
                        }
                        else
                        {
                            int o = y * src.RowPitch + x * 4;
                            px[t * 4 + s] = new Vector4(src.Pixels[o], src.Pixels[o + 1], src.Pixels[o + 2], src.Pixels[o + 3]) / 255f;
                        }
                    }
                }
                // Replicate pixels for partial block
                for (int t = 0; t < ph; t++)
                    for (int s = pw; s < 4; s++)
                        px[t * 4 + s] = px[t * 4 + rep[s]];
                for (int t = ph; t < 4; t++)
                    for (int s = 0; s < 4; s++)
                        px[t * 4 + s] = px[rep[t] * 4 + s];

                encode(dst.Pixels.AsSpan((by * xblocks + bx) * 16, 16), px, flags);
            }
        });
        return dst;
    }

    /// <summary>Noisy image with hard edges and translucent cells (exercises all modes).</summary>
    private static Image Noisy(int w, int h, bool opaque = false)
    {
        var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, w, h);
        var rng = new Random(42);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int o = y * img.RowPitch + x * 4;
                bool cell = ((x / 5) + (y / 7)) % 3 == 0;
                img.Pixels[o] = (byte)rng.Next(256);
                img.Pixels[o + 1] = (byte)(cell ? 230 : rng.Next(64));
                img.Pixels[o + 2] = (byte)((x * 37) ^ (y * 11));
                img.Pixels[o + 3] = opaque ? (byte)255 : (byte)(cell ? rng.Next(256) : 255);
            }
        }
        return img;
    }

    private static int Bc7Mode(ReadOnlySpan<byte> block)
    {
        for (int bit = 0; bit < 8; bit++)
            if ((block[0] & (1 << bit)) != 0) return bit;
        return 8; // reserved
    }

    private static int[] ModeHistogram(Image bc7)
    {
        var hist = new int[9];
        for (int o = 0; o < bc7.SlicePitch; o += 16) hist[Bc7Mode(bc7.Pixels.AsSpan(o, 16))]++;
        return hist;
    }

    private static double Bc7Psnr(Image src, BcFlags flags = BcFlags.None) =>
        TestImages.Psnr(src, Bc7Decoder.Decode(EncodeImage(src, DxgiFormat.BC7_UNORM, Bc6hBc7Codec.EncodeBC7, flags)));

    [Fact]
    public void Bc7Gradient64Above40dB()
    {
        double q = Bc7Psnr(TestImages.Gradient(64, 64));
        Assert.True(q > 40, $"{q:F2} dB");
    }

    [Fact]
    public void Bc7NonMultipleOf4Above35dB()
    {
        double q = Bc7Psnr(TestImages.Gradient(37, 21));
        Assert.True(q > 35, $"{q:F2} dB");
    }

    [Fact]
    public void Bc7Mode6OnlyEmitsOnlyMode6()
    {
        var src = TestImages.Gradient(64, 64);
        var enc = EncodeImage(src, DxgiFormat.BC7_UNORM, Bc6hBc7Codec.EncodeBC7, BcFlags.ForceBc7Mode6);
        var hist = ModeHistogram(enc);
        Assert.Equal(16 * 16, hist[6]);
        double q = TestImages.Psnr(src, Bc7Decoder.Decode(enc));
        Assert.True(q > 35, $"{q:F2} dB");
    }

    [Fact]
    public void Bc7ThreeSubsetsIsNoWorseThanDefault()
    {
        var src = Noisy(32, 32);
        double def = Bc7Psnr(src);
        double three = Bc7Psnr(src, BcFlags.Use3Subsets);
        Assert.True(three >= def - 0.05, $"3-subsets {three:F2} dB vs default {def:F2} dB");
    }

    [Theory]
    [InlineData(BcFlags.None)]
    [InlineData(BcFlags.Use3Subsets)]
    [InlineData(BcFlags.ForceBc7Mode6)]
    public void Bc7BlocksUseLegalModes(BcFlags flags)
    {
        var enc = EncodeImage(Noisy(64, 64), DxgiFormat.BC7_UNORM, Bc6hBc7Codec.EncodeBC7, flags);
        var hist = ModeHistogram(enc);
        Assert.Equal(0, hist[8]);
        if ((flags & BcFlags.Use3Subsets) == 0)
        {
            Assert.Equal(0, hist[0]);
            Assert.Equal(0, hist[2]);
        }
        if (flags == BcFlags.None)
            Assert.True(hist.Count(c => c > 0) >= 3, string.Join(",", hist));
        if (flags == BcFlags.Use3Subsets)
            Assert.True(hist[0] + hist[2] > 0, string.Join(",", hist));
    }

    [Fact]
    public void Bc7OpaqueBlocksNeverUseMode7()
    {
        var hist = ModeHistogram(EncodeImage(Noisy(64, 64, opaque: true), DxgiFormat.BC7_UNORM, Bc6hBc7Codec.EncodeBC7));
        Assert.Equal(0, hist[7]);
    }

    [Fact]
    public void Bc7SolidColorIsNearExact()
    {
        // DirectXTex does not guarantee exact solid colors (p-bit parity); native output is 54.15 dB.
        double q = Bc7Psnr(TestImages.Flat(8, 8, 12, 200, 99, 255));
        Assert.True(q > 50, $"{q:F2} dB");
    }

    private static Image EncodeBc6h(Image src, bool signed) =>
        EncodeImage(src, signed ? DxgiFormat.BC6H_SF16 : DxgiFormat.BC6H_UF16,
            signed ? Bc6hBc7Codec.EncodeBC6HS : Bc6hBc7Codec.EncodeBC6HU);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Bc6hGradient64Above40dB(bool signed)
    {
        var src = TestImages.HdrGradient(64, 64);
        double q = TestImages.PsnrHdr(src, Bc6hDecoder.Decode(EncodeBc6h(src, signed)));
        Assert.True(q > 40, $"{(signed ? "SF16" : "UF16")}: {q:F2} dB");
    }

    [Fact]
    public void Bc6hNonMultipleOf4Above35dB()
    {
        var src = TestImages.HdrGradient(37, 21);
        double q = TestImages.PsnrHdr(src, Bc6hDecoder.Decode(EncodeBc6h(src, false)));
        Assert.True(q > 35, $"{q:F2} dB");
    }

    [Fact]
    public void Bc6hSf16PreservesNegativeValues()
    {
        // Unlike the GPU shader (max(pixel, 0)), DirectXTex's CPU SF16 encoder keeps negatives.
        var src = TestImages.HdrGradient(16, 16, signed: true); // blue spans -1..1
        var dec = Bc6hDecoder.Decode(EncodeBc6h(src, true));
        var f = dec.Floats;
        float minBlue = float.MaxValue;
        for (int i = 2; i < f.Length; i += 4) minBlue = Math.Min(minBlue, f[i]);
        Assert.True(minBlue < -0.9f, $"min blue {minBlue}");
        // Sign-crossing data is hard for SF16 (one fewer magnitude bit); native DirectXTex gives 33.33 dB.
        double q = TestImages.PsnrHdr(src, dec);
        Assert.True(q > 30, $"{q:F2} dB");
    }

    [Fact]
    public void Bc6hUf16ClampsNegativesToZero()
    {
        var dec = Bc6hDecoder.Decode(EncodeBc6h(TestImages.HdrGradient(16, 16, signed: true), false));
        var f = dec.Floats;
        for (int i = 2; i < f.Length; i += 4) Assert.True(f[i] >= 0);
    }

    [Fact]
    public void EncodingIsDeterministic()
    {
        var src = Noisy(32, 32);
        var a = EncodeImage(src, DxgiFormat.BC7_UNORM, Bc6hBc7Codec.EncodeBC7);
        var b = EncodeImage(src, DxgiFormat.BC7_UNORM, Bc6hBc7Codec.EncodeBC7);
        Assert.Equal(a.Pixels, b.Pixels);
    }

    //---------------------------------------------------------------------------------
    // Bit-exact parity with native DirectXTex. The expected bytes were produced by
    // DirectXTex's own BC6HBC7.cpp (built natively) from the same deterministic blocks:
    // solid, gradient, noise, two-color, opaque and alpha-ramp patterns (LDR), and
    // 1x/8x/300x-scaled versions with sign-crossing values (HDR).
    //---------------------------------------------------------------------------------

    private static Vector4[] GoldenBlocks(bool hdr, bool signed)
    {
        const int n = 12;
        var rng = new Random(7);
        var px = new Vector4[n * 16];
        for (int b = 0; b < n; b++)
        {
            int kind = b % 6;
            var c0 = new Vector4(rng.NextSingle(), rng.NextSingle(), rng.NextSingle(), rng.NextSingle());
            var c1 = new Vector4(rng.NextSingle(), rng.NextSingle(), rng.NextSingle(), rng.NextSingle());
            for (int i = 0; i < 16; i++)
            {
                Vector4 v = kind switch
                {
                    0 => c0,
                    1 => Vector4.Lerp(c0, c1, i / 15f),
                    2 => new Vector4(rng.NextSingle(), rng.NextSingle(), rng.NextSingle(), rng.NextSingle()),
                    3 => (i & 5) != 0 ? c0 : c1,
                    4 => new Vector4(Vector3.Lerp(new(c0.X, c0.Y, c0.Z), new(c1.X, c1.Y, c1.Z), (i % 4) / 3f), 1f),
                    _ => new Vector4(c0.X, c0.Y, c0.Z, (i % 4) / 3f),
                };
                if (hdr)
                {
                    float scale = (b % 3) switch { 0 => 1f, 1 => 8f, _ => 300f };
                    v = new Vector4(v.X * scale, v.Y * scale, v.Z * scale, 1f);
                    if (signed && (b % 2 == 1)) v = new Vector4(v.X - scale * 0.5f, v.Y - scale * 0.5f, -v.Z, 1f);
                }
                else
                {
                    v = new Vector4(MathF.Round(v.X * 255) / 255, MathF.Round(v.Y * 255) / 255,
                        MathF.Round(v.Z * 255) / 255, MathF.Round(v.W * 255) / 255);
                }
                px[b * 16 + i] = v;
            }
        }
        return px;
    }

    private const string GoldenBc7Default =
        "C058ECFDA6520D0600000000000000004074CF7CF1A9E73801215376A8CBEDFFD04A8F3F97A1139C81A709ACB9122429401F5223D2BCDC52F1F0FFFFF0F0FFFF08602C11CAFF8C7BF6D545AC78787878C06412895428007F50FA50FA50FA50FA4093C9E3190C56AB0100000000000000C0632E4AD75AC04B1032547698BADCFE50577D708792D3CBAA5CB06FEF686FE340584AC6AB96AA02F1F0FFFFF0F0FFFF08D896F6B044369FA63C88C978787878200A0506E3F100FC03000000E4E4E4E4";

    private const string GoldenBc7Mode6 =
        "C058ECFDA6520D0600000000000000004074CF7CF1A9E73801215376A8CBEDFFC04DFFA01F6324D86083AAC945702B86401F5223D2BCDC52F1F0FFFFF0F0FFFF404CE47FEB17FF7F51FA50FA50FA50FAC06412895428007F50FA50FA50FA50FA4093C9E3190C56AB0100000000000000C0632E4AD75AC04B1032547698BADCFEC0F06B370733F20FB0F9565A7265EE4640584AC6AB96AA02F1F0FFFFF0F0FFFF409B5D324F21FE7F50FA50FA50FA50FA40850283F178007F50FA50FA50FA50FA";

    private const string GoldenBc6hU =
        "C2EFE665EF601817C03E9424499224493692C816F373D701F3AE018DF5EF9613796BE8BE1EFEF9BB5D09D0F1E504E1D92BE64213D5504736F1F0FFFFF0F0FFFFE7FF524231AF75E351F950F950F950F987B8E110DA5FC000131111111111111107EDAA0D25A03F041211111111111111BA31C71EFF4A18FEBBBF018DF56F9713FE1DF5DC74FD8DB6DB727374E800AEE0676FC0679FCF82D9F0F0FFFFF0F0FFFFA1785F860FD648DF8801F0033FF0033FA720AEE4CADF40021311111111111111";

    private const string GoldenBc6hS =
        "CBEFE665D7E7007E12111111111111111E3AE8DF7BA091D7E7A90144EB6DDB01726BA8BE1CFEF9BB5D09D0F1E504E1D9C3ED935FBE856591F1F0FFFFF0F0FFFFEB7F53443187556251F950F950F950F90BA07DF0C5B7408012111111111111118236D586FA200800802F94244992244903A57EC42519227E25223233EBFEFFFFFA8B29B6EE9061A13ACC64FFB6A2C88C874E3F4F94DF4813F1F0FFFFF0F0FFFFB2481F860ED64CDF8801F0033FF0033F8BCA3B1D050820800000000000000000";

    private static string EncodeGolden(bool hdr, bool signed, BlockEncoder encode, BcFlags flags)
    {
        var px = GoldenBlocks(hdr, signed);
        var output = new byte[px.Length];
        for (int b = 0; b < px.Length / 16; b++) encode(output.AsSpan(b * 16, 16), px.AsSpan(b * 16, 16), flags);
        return Convert.ToHexString(output);
    }

    [Fact]
    public void Bc7MatchesNativeDirectXTex() =>
        Assert.Equal(GoldenBc7Default, EncodeGolden(false, false, Bc6hBc7Codec.EncodeBC7, BcFlags.None));

    [Fact]
    public void Bc7ThreeSubsetsMatchesNativeDirectXTex() =>
        // For these blocks DirectXTex picks no 3-subset mode, so the output equals the default.
        Assert.Equal(GoldenBc7Default, EncodeGolden(false, false, Bc6hBc7Codec.EncodeBC7, BcFlags.Use3Subsets));

    [Fact]
    public void Bc7Mode6MatchesNativeDirectXTex() =>
        Assert.Equal(GoldenBc7Mode6, EncodeGolden(false, false, Bc6hBc7Codec.EncodeBC7, BcFlags.ForceBc7Mode6));

    [Fact]
    public void Bc6hUf16MatchesNativeDirectXTex() =>
        Assert.Equal(GoldenBc6hU, EncodeGolden(true, false, Bc6hBc7Codec.EncodeBC6HU, BcFlags.None));

    [Fact]
    public void Bc6hSf16MatchesNativeDirectXTex() =>
        Assert.Equal(GoldenBc6hS, EncodeGolden(true, true, Bc6hBc7Codec.EncodeBC6HS, BcFlags.None));
}
