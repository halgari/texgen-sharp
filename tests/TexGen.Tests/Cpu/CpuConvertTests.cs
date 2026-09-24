//-------------------------------------------------------------------------------------
// End-to-end tests of the DirectXTex CPU codec through Converter (routing, driver).
//-------------------------------------------------------------------------------------

using TexGen.Decoders;

namespace TexGen.Tests.Cpu;

public class CpuConvertTests
{
    [Theory]
    [InlineData(DxgiFormat.BC1_UNORM, 30)]
    [InlineData(DxgiFormat.BC3_UNORM, 30)]
    [InlineData(DxgiFormat.BC5_UNORM, 40)]
    public void CpuCodecNonMultipleOf4NeedsNoDevice(DxgiFormat fmt, double min)
    {
        // Compression-only with Codec=Cpu never touches a GPU device.
        var src = TestImages.Gradient(37, 21, alpha: false);
        var s = Converter.Convert(src, new ConvertOptions { Format = fmt, Codec = CodecPreference.Cpu });
        Assert.Equal(fmt, s.Metadata.Format);
        int[] ch = fmt == DxgiFormat.BC5_UNORM ? [0, 1] : [0, 1, 2];
        double q = TestImages.Psnr(src, BcnDecoder.Decode(s.BaseImage), ch);
        Assert.True(q > min, $"{fmt}: {q:F2} dB");
    }

    [Fact]
    public void SnormFormatsRouteToTheCpuCodec()
    {
        // GPU codec requested, but only the CPU codec implements SNORM.
        var src = TestImages.Flat(8, 8, 255, 0, 128, 255);
        var s = Converter.Convert(src, new ConvertOptions { Format = DxgiFormat.BC4_SNORM, Codec = CodecPreference.Gpu });
        Assert.Equal(DxgiFormat.BC4_SNORM, s.Metadata.Format);
        // Red 255 -> +1.0 -> endpoint 127 (signed).
        Assert.Equal(127, (sbyte)s.BaseImage.Pixels[0]);
    }

    [Fact]
    public void CpuCodecMipChainMatchesSingleLevelEncode()
    {
        var src = TestImages.Gradient(16, 16);
        var chain = Converter.Convert(src, new ConvertOptions { Format = DxgiFormat.BC3_UNORM, MipLevels = 0, Codec = CodecPreference.Cpu });
        Assert.Equal(5, chain.Metadata.MipLevels);
        var single = Converter.Convert(src, new ConvertOptions { Format = DxgiFormat.BC3_UNORM, Codec = CodecPreference.Cpu });
        Assert.Equal(single.BaseImage.Pixels, chain.Images[0].Pixels);
    }
}
