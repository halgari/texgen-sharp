//-------------------------------------------------------------------------------------
// GPU integration tests for the BC1/2/3/4/5 encoders: encode, decode with the CPU
// reference decoder, and assert PSNR (thresholds from the texconv-js suite).
//-------------------------------------------------------------------------------------

using TexGen.Decoders;
using TexGen.Gpu;

namespace TexGen.Tests.Gpu;

[Collection("gpu")]
public class BcnTests(GpuFixture gpu)
{
    private Image Encode(Image src, DxgiFormat fmt)
    {
        using var tex = GpuTexture.Upload(gpu.Device, src);
        using var blocks = gpu.Device.Bcn.Encode(tex, fmt);
        return blocks.Download();
    }

    [Theory]
    [InlineData(DxgiFormat.BC1_UNORM, new[] { 0, 1, 2 }, 30)]
    [InlineData(DxgiFormat.BC2_UNORM, new[] { 0, 1, 2, 3 }, 30)]
    [InlineData(DxgiFormat.BC3_UNORM, new[] { 0, 1, 2, 3 }, 30)]
    [InlineData(DxgiFormat.BC4_UNORM, new[] { 0 }, 40)]
    [InlineData(DxgiFormat.BC5_UNORM, new[] { 0, 1 }, 40)]
    public void Gradient64PsnrAboveThreshold(DxgiFormat fmt, int[] channels, double min)
    {
        var src = TestImages.Gradient(64, 64);
        var enc = Encode(src, fmt);
        Assert.Equal(fmt, enc.Format);
        Assert.Equal(16 * 16 * Dxgi.BlockBytes(fmt), enc.SlicePitch);
        double q = TestImages.Psnr(src, BcnDecoder.Decode(enc), channels);
        Assert.True(q > min, $"{fmt}: {q:F2} dB");
    }

    [Fact]
    public void Bc3NonMultipleOf4DoesNotCorrupt()
    {
        var src = TestImages.Gradient(30, 18);
        double q = TestImages.Psnr(src, BcnDecoder.Decode(Encode(src, DxgiFormat.BC3_UNORM)));
        Assert.True(q > 28, $"{q:F2} dB");
    }

    [Fact]
    public void LargeBc1Over65535Blocks()
    {
        // 1024x1024 = 65536 blocks — the size that broke WebGPU's per-dimension dispatch cap.
        var src = TestImages.Gradient(1024, 1024, alpha: false);
        double q = TestImages.Psnr(src, BcnDecoder.Decode(Encode(src, DxgiFormat.BC1_UNORM)), 0, 1, 2);
        Assert.True(q > 30, $"{q:F2} dB");
    }
}
