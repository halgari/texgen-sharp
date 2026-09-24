namespace TexGen.Tests;

public class DdsTests
{
    [Fact]
    public void ComputePitchUncompressedRgba8() =>
        Assert.Equal(new Pitch(64, 512), Dxgi.ComputePitch(DxgiFormat.R8G8B8A8_UNORM, 16, 8));

    [Fact]
    public void ComputePitchBc1() =>
        // 16x16 -> 4x4 blocks -> rowPitch 4*8=32, slice 32*4=128
        Assert.Equal(new Pitch(32, 128), Dxgi.ComputePitch(DxgiFormat.BC1_UNORM, 16, 16));

    [Fact]
    public void ComputePitchBc7NonMultipleOf4() =>
        // 5x5 -> 2x2 blocks -> rowPitch 2*16=32, slice 32*2=64
        Assert.Equal(new Pitch(32, 64), Dxgi.ComputePitch(DxgiFormat.BC7_UNORM, 5, 5));

    [Fact]
    public void RoundTripsUncompressedRgba8ViaDx10Header()
    {
        var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 4, 4);
        for (int i = 0; i < img.Pixels.Length; i++) img.Pixels[i] = (byte)(i * 7);
        var dds = Dds.Write(ScratchImage.From2D(img));
        Assert.Equal(148 + 64, dds.Length); // magic + header + dx10 + pixels

        var back = Dds.Read(dds);
        Assert.Equal(DxgiFormat.R8G8B8A8_UNORM, back.Metadata.Format);
        Assert.Equal(4, back.Metadata.Width);
        Assert.Equal(4, back.Metadata.Height);
        Assert.Single(back.Images);
        Assert.Equal(img.Pixels, back.Images[0].Pixels);
    }

    [Fact]
    public void RoundTripsABc7MipChainHeader()
    {
        var b = Image.Create(DxgiFormat.BC7_UNORM, 8, 8);
        var mip1 = Image.Create(DxgiFormat.BC7_UNORM, 4, 4);
        Array.Fill(b.Pixels, (byte)0xab);
        Array.Fill(mip1.Pixels, (byte)0xcd);
        var back = Dds.Read(Dds.Write(ScratchImage.FromMipChain([b, mip1]), DdsAlphaMode.Premultiplied));
        Assert.Equal(2, back.Metadata.MipLevels);
        Assert.Equal(DxgiFormat.BC7_UNORM, back.Metadata.Format);
        Assert.Equal((uint)DdsAlphaMode.Premultiplied, back.Metadata.MiscFlags2);
        Assert.Equal(0xab, back.Images[0].Pixels[0]);
        Assert.Equal(0xcd, back.Images[1].Pixels[0]);
    }

    /// <summary>Hand-build a legacy (non-DX10) FourCC DDS for the given format + one block.</summary>
    private static byte[] LegacyFourCCDds(string fourCC, int width, int height, byte fill)
    {
        int slice = ((width + 3) >> 2) * ((height + 3) >> 2) * 16; // BC2/BC3 = 16 B/block
        var buf = new byte[4 + 124 + slice];
        void U32(int off, uint v) => BitConverter.TryWriteBytes(buf.AsSpan(off), v);
        U32(0, 0x20534444); // "DDS "
        U32(4, 124);
        U32(8, 0x1007 | 0x80000);
        U32(12, (uint)height);
        U32(16, (uint)width);
        U32(76, 32); // pf size
        U32(80, 0x4); // DDPF_FOURCC
        U32(84, (uint)(fourCC[0] | (fourCC[1] << 8) | (fourCC[2] << 16) | (fourCC[3] << 24)));
        buf.AsSpan(4 + 124).Fill(fill);
        return buf;
    }

    [Fact]
    public void LegacyDxt5MapsToBc3()
    {
        var back = Dds.Read(LegacyFourCCDds("DXT5", 4, 4, 0x5a));
        Assert.Equal(DxgiFormat.BC3_UNORM, back.Metadata.Format);
        Assert.Equal(4, back.Metadata.Width);
        Assert.Single(back.Images);
        Assert.Equal(0x5a, back.Images[0].Pixels[0]);
    }

    [Fact]
    public void LegacyDxt3MapsToBc2() =>
        Assert.Equal(DxgiFormat.BC2_UNORM, Dds.Read(LegacyFourCCDds("DXT3", 8, 8, 0x11)).Metadata.Format);

    [Fact]
    public void ThrowsOnUnsupportedFourCC()
    {
        var e = Assert.Throws<InvalidDataException>(() => Dds.Read(LegacyFourCCDds("ZZZZ", 4, 4, 0)));
        Assert.Contains("unsupported FourCC", e.Message);
    }
}
