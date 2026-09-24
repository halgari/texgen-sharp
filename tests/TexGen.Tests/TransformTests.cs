namespace TexGen.Tests;

public class TransformTests
{
    private static Image Img2x2(params byte[] pixels)
    {
        var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 2, 2);
        pixels.CopyTo(img.Pixels, 0);
        return img;
    }

    // Distinct pixel per corner: [TL, TR, BL, BR] with r encoding position.
    private static Image Corners() => Img2x2(10, 0, 0, 255, 20, 0, 0, 255, 30, 0, 0, 255, 40, 0, 0, 255);

    private static byte[] Reds(Image i) => [i.Pixels[0], i.Pixels[4], i.Pixels[8], i.Pixels[12]];

    [Fact]
    public void HFlipMirrorsColumns() => Assert.Equal([20, 10, 40, 30], Reds(Transforms.Flip(Corners(), true, false)));

    [Fact]
    public void VFlipMirrorsRows() => Assert.Equal([30, 40, 10, 20], Reds(Transforms.Flip(Corners(), false, true)));

    [Fact]
    public void FlipWorksOnFloatImages()
    {
        var hdr = Image.Create(DxgiFormat.R32G32B32A32_FLOAT, 2, 1);
        float[] v = [1, 0, 0, 1, 2, 0, 0, 1];
        v.CopyTo(hdr.Floats);
        var f = Transforms.Flip(hdr, true, false).Floats;
        Assert.Equal(2f, f[0]);
        Assert.Equal(1f, f[4]);
    }

    [Fact]
    public void ParsesSwizzleMasksWithFillForward()
    {
        Assert.Equal([3, 2, 1, 0], Transforms.ParseSwizzle("abgr"));
        Assert.Equal([0, 1, 2, 5], Transforms.ParseSwizzle("rgb1"));
        Assert.Equal([0, 0, 0, 0], Transforms.ParseSwizzle("r"));
        Assert.Equal([0, 1, 2, 3], Transforms.ParseSwizzle("xyzw"));
        Assert.Contains("invalid", Assert.Throws<FormatException>(() => Transforms.ParseSwizzle("rq")).Message);
    }

    [Fact]
    public void SwizzleRearrangesChannelsWithConstants()
    {
        var img = Img2x2(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        Assert.Equal([3, 2, 0, 255], Transforms.Swizzle(img, "bg01").Pixels[..4]);
    }

    [Fact]
    public void ColorKeyKeysMatchingPixelsAndForcesOthersOpaque()
    {
        var img = Img2x2(0, 255, 0, 200, 10, 250, 5, 200, 200, 0, 0, 200, 0, 0, 255, 200);
        var k = Transforms.ColorKey(img, 0x00ff00);
        Assert.Equal([0, 0, 0, 0], k.Pixels[..4]);
        Assert.Equal([0, 0, 0, 0], k.Pixels[4..8]);
        Assert.Equal([200, 0, 0, 255], k.Pixels[8..12]);
    }

    [Fact]
    public void PremultipliesAndUndoes()
    {
        var img = Img2x2(200, 100, 50, 128, 0, 0, 0, 0, 255, 255, 255, 255, 10, 20, 30, 64);
        var pm = Transforms.PremultiplyAlpha(img);
        Assert.Equal((int)Math.Round(200 * 128 / 255.0), pm.Pixels[0]);
        Assert.Equal(128, pm.Pixels[3]);
        var undone = Transforms.PremultiplyAlpha(pm, undo: true);
        Assert.InRange(undone.Pixels[0], 199, 201);
    }
}

public class ConverterCpuTests
{
    [Fact]
    public void BgraProducesSwappedChannels()
    {
        var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 2, 2);
        byte[] px = [10, 20, 30, 40];
        px.CopyTo(img.Pixels, 0);
        var s = Converter.Convert(img, new ConvertOptions { Format = DxgiFormat.B8G8R8A8_UNORM });
        Assert.Equal(DxgiFormat.B8G8R8A8_UNORM, s.Metadata.Format);
        Assert.Equal([30, 20, 10, 40], s.BaseImage.Pixels[..4]);
    }

    [Fact]
    public void AppliesHFlipAndSwizzle()
    {
        var corners = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 2, 2);
        byte[] px = [10, 0, 0, 255, 20, 0, 0, 255, 30, 0, 0, 255, 40, 0, 0, 255];
        px.CopyTo(corners.Pixels, 0);
        var s = Converter.Convert(corners, new ConvertOptions { HFlip = true, Swizzle = "gr" });
        // hflip puts 20 at x=0; swizzle "gr" -> [g, r, r, r]
        Assert.Equal([0, 20, 20, 20], s.BaseImage.Pixels[..4]);
    }

    [Fact]
    public void RejectsRgba8OnlyTransformsOnFloatSources()
    {
        var hdr = Image.Create(DxgiFormat.R32G32B32A32_FLOAT, 4, 4);
        var e = Assert.Throws<ArgumentException>(() =>
            Converter.Convert(hdr, new ConvertOptions { Format = DxgiFormat.BC6H_UF16, Swizzle = "abgr" }));
        Assert.Contains("not supported for HDR float sources", e.Message);
    }

    [Fact]
    public void RejectsPmalphaOnBc6hTargets()
    {
        var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 4, 4);
        var e = Assert.Throws<ArgumentException>(() =>
            Converter.Convert(img, new ConvertOptions { Format = DxgiFormat.BC6H_UF16, PremultiplyAlpha = true }));
        Assert.Contains("not supported for BC6H", e.Message);
    }

    [Fact]
    public void RejectsFloatSourceToLdrTarget()
    {
        var e = Assert.Throws<ArgumentException>(() =>
            Converter.Convert(TestImages.HdrGradient(8, 8), new ConvertOptions { Format = DxgiFormat.BC7_UNORM }));
        Assert.Contains("BC6H", e.Message);
    }

    [Fact]
    public void UncompressedFloatAndSingleChannelTargets()
    {
        var img = TestImages.Flat(2, 2, 255, 128, 0, 255);
        var fp16 = Converter.Convert(img, new ConvertOptions { Format = DxgiFormat.R16G16B16A16_FLOAT, SrgbFilter = false });
        Assert.Equal(8 * 4, fp16.BaseImage.SlicePitch);
        Assert.Equal(1f, (float)BitConverter.ToHalf(fp16.BaseImage.Pixels, 0));

        var r8 = Converter.Convert(img, new ConvertOptions { Format = DxgiFormat.R8_UNORM });
        Assert.Equal([255, 255, 255, 255], r8.BaseImage.Pixels);

        // Round-trip back through the uncompressed decoder.
        var back = PixelConvert.Decode(fp16.BaseImage);
        Assert.Equal(128f / 255, back.Floats[1], 3);
    }
}
