using System.Text;

namespace TexGen.Tests;

public class HdrTests
{
    private static byte[] Concat(string header, params byte[] body) => [.. Encoding.ASCII.GetBytes(header), .. body];

    private static string Header(int w, int h) => $"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n";

    [Fact]
    public void FlatScanlinesDecodeToLinearFloats()
    {
        // (128, 64, 255, 129): scale = 2^(129-136) = 1/128 -> (1.0, 0.5, ~1.99)
        var img = Hdr.Load(Concat(Header(2, 1), 128, 64, 255, 129, 0, 0, 0, 0));
        Assert.Equal(DxgiFormat.R32G32B32A32_FLOAT, img.Format);
        Assert.Equal(2, img.Width);
        var f = img.Floats;
        Assert.Equal(1.0f, f[0], 5);
        Assert.Equal(0.5f, f[1], 5);
        Assert.Equal(255f / 128, f[2], 5);
        Assert.Equal(1f, f[3]);
        Assert.Equal(0f, f[4]); // zero exponent = black
    }

    [Fact]
    public void NewStyleRleRunsDecode()
    {
        const int width = 8;
        var planes = new List<byte> { 2, 2, 0, width };
        foreach (byte v in new byte[] { 200, 100, 50, 130 }) planes.AddRange([128 + width, v]);
        var f = Hdr.Load(Concat(Header(width, 1), [.. planes])).Floats;
        float scale = MathF.Pow(2, 130 - 136);
        for (int x = 0; x < width; x++)
        {
            Assert.Equal(200 * scale, f[x * 4], 5);
            Assert.Equal(100 * scale, f[x * 4 + 1], 5);
            Assert.Equal(50 * scale, f[x * 4 + 2], 5);
        }
    }

    [Fact]
    public void NewStyleRleLiteralSpansDecode()
    {
        const int width = 8;
        byte[] r = [10, 20, 30, 40, 50, 60, 70, 80];
        var planes = new List<byte> { 2, 2, 0, width };
        foreach (int add in new[] { 0, 1, 2 })
        {
            planes.Add(width);
            planes.AddRange(r.Select(v => (byte)(v + add)));
        }
        planes.AddRange([128 + width, 136]); // exponent: run of constant 136
        var f = Hdr.Load(Concat(Header(width, 1), [.. planes])).Floats;
        Assert.Equal(10f, f[0], 5);
        Assert.Equal(80f, f[4 * 7], 5);
    }

    [Fact]
    public void OldStyleRleRunsRepeatThePreviousPixel()
    {
        var f = Hdr.Load(Concat(Header(4, 1), 128, 64, 32, 136, 1, 1, 1, 2, 200, 100, 50, 136)).Floats;
        Assert.Equal(128f, f[0], 5);
        Assert.Equal(128f, f[4], 5);
        Assert.Equal(128f, f[8], 5);
        Assert.Equal(200f, f[12], 5);
    }

    [Fact]
    public void TruncatedScanlineThrows()
    {
        var e = Assert.Throws<InvalidDataException>(() => Hdr.Load(Concat(Header(4, 1), 10, 20)));
        Assert.Contains("truncated", e.Message);
    }

    [Fact]
    public void Rgba8ToLinearFloatAppliesSrgbCurveAndTonemapRoundTrips()
    {
        var src = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 2, 1);
        byte[] px = [255, 188, 0, 255, 0, 0, 0, 128];
        px.CopyTo(src.Pixels, 0);
        var lin = Hdr.Rgba8ToLinearFloat(src);
        var f = lin.Floats;
        Assert.Equal(1.0f, f[0], 5);
        Assert.Equal(MathF.Pow((188f / 255 + 0.055f) / 1.055f, 2.4f), f[1], 4);
        Assert.Equal(1f, f[3]);
        Assert.Equal(128f / 255, f[7], 3);

        // Reinhard maps 1.0 -> 0.5 linear; verify the sRGB-encoded result.
        var tone = Hdr.TonemapToRgba8(lin);
        int expected = (int)Math.Round((1.055 * Math.Pow(0.5, 1 / 2.4) - 0.055) * 255);
        Assert.Equal(expected, tone.Pixels[0]);
    }
}
