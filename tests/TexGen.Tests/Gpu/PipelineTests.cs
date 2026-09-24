//-------------------------------------------------------------------------------------
// GPU pipeline tests: resize, mip generation, premultiply, and DDS round-trip.
//-------------------------------------------------------------------------------------

using TexGen.Gpu;

namespace TexGen.Tests.Gpu;

[Collection("gpu")]
public class PipelineTests(GpuFixture gpu)
{
    private Image Resize(Image src, int w, int h, bool srgb)
    {
        using var tex = GpuTexture.Upload(gpu.Device, src);
        using var outTex = gpu.Device.ImageOps.Resize(tex, w, h, srgb);
        return outTex.Download();
    }

    [Fact]
    public void Resize64To16PreservesFlatColor()
    {
        var o = Resize(TestImages.Flat(64, 64, 10, 200, 50, 255), 16, 16, false);
        Assert.Equal(16, o.Width);
        Assert.Equal(16, o.Height);
        Assert.Equal([10, 200, 50, 255], o.Pixels[..4]);
    }

    [Fact]
    public void Resize64To32DimsAndPlausibleValue()
    {
        var o = Resize(TestImages.Gradient(64, 64, alpha: false), 32, 32, false);
        int mid = o.Pixels[16 * o.RowPitch + 16 * 4];
        Assert.InRange(mid, 101, 159);
    }

    [Fact]
    public void SrgbAwareDownsampleOfBlackWhiteCheckerIsBrighterThanLinearAverage()
    {
        var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 2, 2);
        byte[] px = [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255];
        px.CopyTo(img.Pixels, 0);
        Assert.Equal(128, Resize(img, 1, 1, false).Pixels[0]);
        Assert.Equal(188, Resize(img, 1, 1, true).Pixels[0]); // linear 0.5 -> sRGB 188
    }

    [Fact]
    public void FullMipChainToBc3ThenDdsRoundTrip()
    {
        var scratch = Converter.Convert(TestImages.Gradient(64, 48, alpha: false),
            new ConvertOptions { Format = DxgiFormat.BC3_UNORM, MipLevels = 0 }, gpu.Device);
        int expected = ScratchImage.CountMips(64, 48); // 7 levels
        Assert.Equal(expected, scratch.Metadata.MipLevels);
        Assert.Equal(64, scratch.Images[0].Width);
        Assert.Equal(48, scratch.Images[0].Height);
        Assert.Equal(1, scratch.Images[^1].Width);

        var back = Dds.Read(Dds.Write(scratch));
        Assert.Equal(DxgiFormat.BC3_UNORM, back.Metadata.Format);
        Assert.Equal(expected, back.Metadata.MipLevels);
        Assert.Equal(scratch.Images[0].Pixels, back.Images[0].Pixels);
    }

    [Fact]
    public void ResizeAndCompressCombined()
    {
        var s = Converter.Convert(TestImages.Gradient(100, 100, alpha: false),
            new ConvertOptions { Format = DxgiFormat.BC1_UNORM, Width = 64, Height = 64 }, gpu.Device);
        Assert.Equal(64, s.Images[0].Width);
        Assert.Equal(64, s.Images[0].Height);
        Assert.Equal(DxgiFormat.BC1_UNORM, s.Metadata.Format);
    }

    [Fact]
    public void UncompressedMipChainWithGpuPremultiply()
    {
        var src = TestImages.Flat(8, 8, 200, 100, 50, 128);
        var s = Converter.Convert(src, new ConvertOptions { MipLevels = 0, PremultiplyAlpha = true }, gpu.Device);
        Assert.Equal(4, s.Metadata.MipLevels);
        var expected = Transforms.PremultiplyAlpha(src).Pixels[..4];
        foreach (var level in s.Images) Assert.Equal(expected, level.Pixels[..4]);
    }

    [Fact]
    public void LdrSourceResizeAndMipsToBc6h()
    {
        // 8-bit source goes through sRGB->linear float expansion, float resize, float mips.
        var ldr = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 64, 64);
        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++)
            {
                int o = y * ldr.RowPitch + x * 4;
                ldr.Pixels[o] = (byte)(x * 4);
                ldr.Pixels[o + 1] = (byte)(y * 4);
                ldr.Pixels[o + 2] = 128;
                ldr.Pixels[o + 3] = 255;
            }
        var s = Converter.Convert(ldr,
            new ConvertOptions { Format = DxgiFormat.BC6H_UF16, Width = 32, Height = 32, MipLevels = 0 }, gpu.Device);
        Assert.Equal(DxgiFormat.BC6H_UF16, s.Metadata.Format);
        Assert.Equal(6, s.Metadata.MipLevels); // 32x32 -> 6 levels
        Assert.Equal(32, s.Images[0].Width);
        Assert.Equal(1, s.Images[^1].Width);

        // Compare against the same resize done in float on the GPU, decoded.
        var dec = TexGen.Decoders.Bc6hDecoder.Decode(s.BaseImage);
        using var tex = GpuTexture.Upload(gpu.Device, Hdr.Rgba8ToLinearFloat(ldr));
        using var small = gpu.Device.ImageOps.Resize(tex, 32, 32, false);
        double q = TestImages.PsnrHdr(small.Download(), dec);
        Assert.True(q > 35, $"{q:F2} dB"); // small-image bound from the texconv-js BC6H suite
    }

    [Fact]
    public void HdrFloatSourceToFp16AndBc6hMips()
    {
        var hdr = TestImages.HdrGradient(64, 32);
        var fp16 = Converter.Convert(hdr, new ConvertOptions { Format = DxgiFormat.R16G16B16A16_FLOAT, MipLevels = 0 }, gpu.Device);
        Assert.Equal(7, fp16.Metadata.MipLevels);
        Assert.Equal(4f, (float)BitConverter.ToHalf(fp16.BaseImage.Pixels, (63 * 4) * 2), 2);

        var bc6 = Converter.Convert(hdr, new ConvertOptions { Format = DxgiFormat.BC6H_SF16, MipLevels = 3 }, gpu.Device);
        Assert.Equal(3, bc6.Metadata.MipLevels);
        Assert.True(TestImages.PsnrHdr(hdr, TexGen.Decoders.Bc6hDecoder.Decode(bc6.BaseImage)) > 40);
    }

    [Fact]
    public void ConcurrentConversionsOnOneDeviceMatchSequentialResults()
    {
        var formats = new[] { DxgiFormat.BC7_UNORM, DxgiFormat.BC1_UNORM, DxgiFormat.BC6H_UF16, DxgiFormat.BC3_UNORM };
        var src = TestImages.Gradient(16, 16); // small: also runs on the (slow) CPU accelerator
        byte[] Run(DxgiFormat f) => Dds.Write(Converter.Convert(src, new ConvertOptions { Format = f, MipLevels = 0 }, gpu.Device));
        var expected = formats.Select(Run).ToArray();
        var actual = new byte[16][];
        Parallel.For(0, actual.Length, i => actual[i] = Run(formats[i % formats.Length]));
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i % formats.Length], actual[i]);
    }
}
