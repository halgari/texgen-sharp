//-------------------------------------------------------------------------------------
// GPU integration tests for the BC7 encoder (ported from texconv-js bc7.test.ts and
// large.test.ts): encode, decode with the CPU reference decoder, assert PSNR.
//-------------------------------------------------------------------------------------

using TexGen.Decoders;
using TexGen.Gpu;

namespace TexGen.Tests.Gpu;

[Collection("gpu")]
public class Bc7Tests(GpuFixture gpu)
{
    private Image Encode(Image src, bool quick = false, bool use3Subsets = false, float alphaWeight = 1f)
    {
        using var tex = GpuTexture.Upload(gpu.Device, src);
        using var blocks = gpu.Device.Bc7.Encode(tex, DxgiFormat.BC7_UNORM, alphaWeight, quick, use3Subsets);
        return blocks.Download();
    }

    [Fact]
    public void Gradient64PsnrAbove40()
    {
        var src = TestImages.Gradient(64, 64);
        var bc7 = Encode(src);
        Assert.Equal(DxgiFormat.BC7_UNORM, bc7.Format);
        Assert.Equal(256 * 16, bc7.SlicePitch);
        double q = TestImages.Psnr(src, Bc7Decoder.Decode(bc7));
        Assert.True(q > 40, $"{q:F2} dB");
    }

    [Fact]
    public void NonMultipleOf4NoOobPsnrAbove35()
    {
        var src = TestImages.Gradient(37, 21);
        double q = TestImages.Psnr(src, Bc7Decoder.Decode(Encode(src)));
        Assert.True(q > 35, $"{q:F2} dB");
    }

    [Fact]
    public void QuickModePsnrAbove35()
    {
        var src = TestImages.Gradient(64, 64);
        double q = TestImages.Psnr(src, Bc7Decoder.Decode(Encode(src, quick: true)));
        Assert.True(q > 35, $"{q:F2} dB");
    }

    [Fact]
    public void Use3SubsetsPsnrAbove40AndNoWorseThanDefault()
    {
        var src = TestImages.Gradient(64, 64);
        double def = TestImages.Psnr(src, Bc7Decoder.Decode(Encode(src)));
        double q = TestImages.Psnr(src, Bc7Decoder.Decode(Encode(src, use3Subsets: true)));
        Assert.True(q > 40, $"{q:F2} dB");
        // Extra candidate modes can only lower the per-block error metric.
        Assert.True(q >= def - 0.05, $"3-subset {q:F2} dB vs default {def:F2} dB");
    }

    [Fact]
    public void NoisyImageUsesMultipleModesAndDecodesClose()
    {
        // Deterministic noise + edges exercises the partitioned modes and packing paths.
        var src = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 64, 64);
        var rng = new Random(1234);
        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++)
            {
                int o = y * src.RowPitch + x * 4;
                bool edge = ((x / 3) + (y / 5)) % 2 == 0;
                src.Pixels[o] = (byte)(edge ? 220 : 30 + rng.Next(20));
                src.Pixels[o + 1] = (byte)(edge ? 40 : 200 + rng.Next(20));
                src.Pixels[o + 2] = (byte)rng.Next(256);
                src.Pixels[o + 3] = (byte)(x < 32 ? 255 : 128 + rng.Next(64));
            }
        var bc7 = Encode(src, use3Subsets: true);
        var modes = new HashSet<int>();
        for (int b = 0; b < bc7.SlicePitch; b += 16)
        {
            int mode = 0;
            while (mode < 8 && ((bc7.Pixels[b + (mode >> 3)] >> (mode & 7)) & 1) == 0) mode++;
            Assert.True(mode < 8, "reserved BC7 mode emitted");
            modes.Add(mode);
        }
        Assert.True(modes.Count >= 3, $"modes used: {string.Join(",", modes)}");
        // Full-range random blue is not representable by any BC7 mode, so this only
        // guards against gross corruption; exact-partition fidelity is tested below.
        double q = TestImages.Psnr(src, Bc7Decoder.Decode(bc7));
        Assert.True(q > 15, $"{q:F2} dB");
    }

    /// <summary>
    /// Each 4x4 block is 2 (or 3) flat colors laid out in an actual BC7 partition
    /// shape, so a correct encoder finds a partitioned mode that reproduces it up to
    /// endpoint quantization. Exercises the 2/3-subset search, fix-up anchors and packing.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void ExactPartitionBlocksReproduceClosely(int subsets)
    {
        const int size = 64; // 256 blocks: every partition shape appears several times
        var src = Image.Create(DxgiFormat.R8G8B8A8_UNORM, size, size);
        var rng = new Random(42 + subsets);
        int xblocks = size / 4;
        for (int by = 0; by < size / 4; by++)
            for (int bx = 0; bx < xblocks; bx++)
            {
                int blk = by * xblocks + bx;
                int part = blk % 64;
                var colors = new byte[3][];
                for (int s = 0; s < 3; s++)
                    colors[s] = [(byte)rng.Next(256), (byte)rng.Next(256), (byte)rng.Next(256), 255];
                for (int pix = 0; pix < 16; pix++)
                {
                    int subset = subsets == 2
                        ? (Bc7Decoder.Partition2[part] >> pix) & 1
                        : (int)((Bc7Decoder.Partition3[part] >> (pix * 2)) & 3);
                    int o = (by * 4 + pix / 4) * src.RowPitch + (bx * 4 + pix % 4) * 4;
                    colors[subset].CopyTo(src.Pixels, o);
                }
            }

        var bc7 = Encode(src, use3Subsets: subsets == 3);
        double q = TestImages.Psnr(src, Bc7Decoder.Decode(bc7));
        Assert.True(q > 38, $"{subsets}-subset shapes: {q:F2} dB");
    }

    [Fact]
    public void LargeOver65535BlocksPsnrAbove40()
    {
        // Only practical on real hardware; the CPU accelerator is too slow for 64K blocks.
        if (!gpu.Device.IsHardware) return;
        var src = TestImages.Gradient(1024, 1024, alpha: false);
        double q = TestImages.Psnr(src, Bc7Decoder.Decode(Encode(src)), 0, 1, 2);
        Assert.True(q > 40, $"{q:F2} dB");
    }

    [Fact]
    public void DecoderHandBuiltMode6SolidBlock()
    {
        // Mode 6: 7 mode bits (0b1000000), 7-bit RGBA endpoints, 2 p-bits, 4-bit indices.
        var block = new byte[16];
        int bit = 0;
        void Put(int value, int n)
        {
            for (int i = 0; i < n; i++, bit++)
                if (((value >> i) & 1) != 0) block[bit >> 3] |= (byte)(1 << (bit & 7));
        }
        Put(0b1000000, 7);
        Put(0x40, 7); Put(0x10, 7); // R0, R1
        Put(0x20, 7); Put(0x10, 7); // G0, G1
        Put(0x7F, 7); Put(0x10, 7); // B0, B1
        Put(0x55, 7); Put(0x10, 7); // A0, A1
        Put(1, 1); Put(0, 1);       // p0 = 1, p1 = 0
        // All indices zero -> every pixel is endpoint 0.
        var px = new byte[64];
        Bc7Decoder.DecodeBlock(block, px);
        for (int p = 0; p < 16; p++)
        {
            Assert.Equal((0x40 << 1) | 1, px[p * 4]);
            Assert.Equal((0x20 << 1) | 1, px[p * 4 + 1]);
            Assert.Equal((0x7F << 1) | 1, px[p * 4 + 2]);
            Assert.Equal((0x55 << 1) | 1, px[p * 4 + 3]);
        }
    }

    [Fact]
    public void DecoderReservedModeDecodesToZero()
    {
        var px = new byte[64];
        Array.Fill(px, (byte)0xAB);
        Bc7Decoder.DecodeBlock(new byte[16], px);
        Assert.All(px, b => Assert.Equal(0, b));
    }
}
