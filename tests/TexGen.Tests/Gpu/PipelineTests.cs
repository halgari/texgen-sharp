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
}
