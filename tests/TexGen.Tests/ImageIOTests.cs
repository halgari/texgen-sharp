namespace TexGen.Tests;

public class ImageIOTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("texgen-io-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void PngRoundTripsExactly()
    {
        var src = TestImages.Gradient(33, 17);
        var path = Path.Combine(_dir, "g.png");
        ImageIO.SaveImage(src, path);
        var back = ImageIO.Load(path);
        Assert.Equal(DxgiFormat.R8G8B8A8_UNORM, back.Format);
        Assert.Equal((33, 17), (back.Width, back.Height));
        Assert.Equal(src.Pixels, back.Pixels);
        // Color management is a no-op for an untagged sRGB PNG.
        Assert.Equal(src.Pixels, ImageIO.Load(path, new LoadOptions { ColorManage = false }).Pixels);
    }

    [Fact]
    public void DdsInputIsDecodedToAWorkingImage()
    {
        var bgra = PixelConvert.Encode(TestImages.Gradient(8, 8), DxgiFormat.B8G8R8A8_UNORM);
        var path = Path.Combine(_dir, "t.dds");
        File.WriteAllBytes(path, Dds.Write(ScratchImage.From2D(bgra)));
        var back = ImageIO.Load(path);
        Assert.Equal(DxgiFormat.R8G8B8A8_UNORM, back.Format);
        Assert.Equal(TestImages.Gradient(8, 8).Pixels, back.Pixels);
    }

    [Fact]
    public void UnrecognizedDataThrows() =>
        Assert.Throws<InvalidDataException>(() => ImageIO.Load([1, 2, 3, 4, 5, 6, 7, 8], "junk.png"));
}
