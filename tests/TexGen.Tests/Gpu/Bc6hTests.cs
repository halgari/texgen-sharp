//-------------------------------------------------------------------------------------
// GPU integration tests for the BC6H encoder: encode HDR float gradients, decode with
// the reference CPU decoder, and assert quality in linear float space (thresholds
// from the texconv-js suite).
//-------------------------------------------------------------------------------------

using TexGen.Decoders;
using TexGen.Gpu;

namespace TexGen.Tests.Gpu;

[Collection("gpu")]
public class Bc6hTests(GpuFixture gpu)
{
    private Image Encode(Image src, DxgiFormat fmt)
    {
        using var tex = GpuTexture.Upload(gpu.Device, src);
        using var blocks = gpu.Device.Bc6h.Encode(tex, fmt);
        return blocks.Download();
    }

    [Theory]
    [InlineData(DxgiFormat.BC6H_UF16)]
    [InlineData(DxgiFormat.BC6H_SF16)] // positive data through the signed format's quantization
    public void Gradient64PsnrAbove40(DxgiFormat fmt)
    {
        var src = TestImages.HdrGradient(64, 64);
        var enc = Encode(src, fmt);
        Assert.Equal(fmt, enc.Format);
        Assert.Equal(256 * 16, enc.SlicePitch);
        double q = TestImages.PsnrHdr(src, Bc6hDecoder.Decode(enc));
        Assert.True(q > 40, $"{fmt}: {q:F2} dB");
    }

    [Fact]
    public void Sf16ClampsNegativeInputsToZero()
    {
        // DirectXTex's DirectCompute encoder does max(pixel, 0) for BOTH formats
        // (BC6HEncode.hlsl), so even SF16 never encodes negative values.
        var src = TestImages.HdrGradient(8, 8, signed: true); // blue spans -1..1
        var f = Bc6hDecoder.Decode(Encode(src, DxgiFormat.BC6H_SF16)).Floats;
        float minBlue = float.PositiveInfinity;
        for (int i = 2; i < f.Length; i += 4) minBlue = Math.Min(minBlue, f[i]);
        Assert.True(minBlue >= 0, $"min blue {minBlue}");
    }

    [Fact]
    public void Uf16NonMultipleOf4NoOutOfBounds()
    {
        var src = TestImages.HdrGradient(37, 21);
        var enc = Encode(src, DxgiFormat.BC6H_UF16);
        Assert.Equal(10 * 6 * 16, enc.SlicePitch);
        double q = TestImages.PsnrHdr(src, Bc6hDecoder.Decode(enc));
        Assert.True(q > 35, $"{q:F2} dB");
    }

    [Fact]
    public void LargeUf16()
    {
        // 1024x1024 = 65536 blocks (past WebGPU's per-dimension dispatch cap). Slow on
        // the CPU accelerator, so only run on real hardware.
        if (!gpu.Device.IsHardware) return;
        var src = TestImages.HdrGradient(1024, 1024);
        double q = TestImages.PsnrHdr(src, Bc6hDecoder.Decode(Encode(src, DxgiFormat.BC6H_UF16)));
        Assert.True(q > 40, $"{q:F2} dB");
    }

    [Fact]
    public void EncodingIsDeterministicAcrossCalls()
    {
        // Re-encoding reuses the cached ping-pong buffers; results must not depend on
        // what a previous (larger) encode left in them.
        var big = TestImages.HdrGradient(64, 64);
        var small = TestImages.HdrGradient(16, 12, signed: true);
        var first = Encode(small, DxgiFormat.BC6H_SF16);
        Encode(big, DxgiFormat.BC6H_UF16);
        var second = Encode(small, DxgiFormat.BC6H_SF16);
        Assert.Equal(first.Pixels, second.Pixels);
    }
}

public class Bc6hDecoderTests
{
    [Fact]
    public void DecodesMode11SolidBlock()
    {
        // Mode 11 (0x03, 10-bit endpoints, not transformed), UF16. Both endpoints
        // r=g=b=0x3FF (the max 10-bit value unquantizes to 0xFFFF), all indices 0:
        // every pixel = FinishUnquantize(0xFFFF) = 0xFFFF*31>>6 = 0x7BFF = 65504 (max half).
        var block = new byte[16];
        var bits = new System.Collections.BitArray(128);
        int pos = 0;
        void Put(int v, int n)
        {
            for (int i = 0; i < n; i++) bits[pos++] = ((v >> i) & 1) != 0;
        }
        Put(0x03, 5);
        for (int e = 0; e < 2; e++) // RW,GW,BW then RX,GX,BX (10 bits each)
            for (int c = 0; c < 3; c++)
                Put(0x3FF, 10);
        bits.CopyTo(block, 0);

        var img = Image.FromPixels(DxgiFormat.BC6H_UF16, 4, 4, block);
        var f = Bc6hDecoder.Decode(img).Floats;
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(65504f, f[i * 4]);
            Assert.Equal(65504f, f[i * 4 + 1]);
            Assert.Equal(65504f, f[i * 4 + 2]);
            Assert.Equal(1f, f[i * 4 + 3]);
        }
    }

    [Fact]
    public void ReservedModeDecodesToOpaqueBlack()
    {
        var block = new byte[16];
        block[0] = 0x13; // mode 0x13 is reserved
        var f = Bc6hDecoder.Decode(Image.FromPixels(DxgiFormat.BC6H_SF16, 4, 4, block)).Floats;
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(0f, f[i * 4]);
            Assert.Equal(1f, f[i * 4 + 3]);
        }
    }

    [Fact]
    public void HalfConversionHelpersRoundTrip()
    {
        // f32tof16 is round-toward-zero with finite overflow saturating to max half.
        Assert.Equal(0x3C00u, Bc6hEncoder.F32ToF16(1.0f));
        Assert.Equal(0x7BFFu, Bc6hEncoder.F32ToF16(1e6f));
        Assert.Equal(0x7C00u, Bc6hEncoder.F32ToF16(float.PositiveInfinity));
        Assert.Equal(0x0001u, Bc6hEncoder.F32ToF16(5.9604645e-8f)); // smallest denormal
        Assert.Equal(0x3C00u, Bc6hEncoder.F32ToF16(1.0009f)); // truncates (RNE would give 0x3C01)
        for (uint h = 0; h < 0x7C00; h++)
        {
            float f = Bc6hEncoder.F16ToF32(h);
            Assert.Equal((float)BitConverter.UInt16BitsToHalf((ushort)h), f);
            Assert.Equal(h, Bc6hEncoder.F32ToF16(f));
        }
    }
}
