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

    private static uint U32At(byte[] dds, int offset) => BitConverter.ToUInt32(dds, offset);

    private static Image Filled(DxgiFormat format, int width, int height, byte seed)
    {
        var img = Image.Create(format, width, height);
        for (int i = 0; i < img.Pixels.Length; i++) img.Pixels[i] = (byte)(seed + i * 13);
        return img;
    }

    /// <summary>Six faces of a two-level mip chain, in DirectXTex subresource order.</summary>
    private static ScratchImage Cube(DxgiFormat format)
    {
        var images = new List<Image>();
        for (int face = 0; face < 6; face++)
        {
            images.Add(Filled(format, 8, 8, (byte)(face * 40)));
            images.Add(Filled(format, 4, 4, (byte)(face * 40 + 1)));
        }
        var meta = new TexMetadata(8, 8, 1, 6, 2, TexMetadata.TexMiscTextureCube, 0, format, TexDimension.Texture2D);
        return ScratchImage.FromImages(meta, images);
    }

    [Theory]
    [InlineData(DxgiFormat.BC1_UNORM, "DXT1")]
    [InlineData(DxgiFormat.BC2_UNORM, "DXT3")]
    [InlineData(DxgiFormat.BC3_UNORM, "DXT5")]
    [InlineData(DxgiFormat.BC4_UNORM, "BC4U")]
    [InlineData(DxgiFormat.BC5_UNORM, "BC5U")]
    public void LegacyHeaderUsesTheFourCCTexconvWrites(DxgiFormat format, string fourCC)
    {
        var img = Filled(format, 8, 8, 3);
        var dds = Dds.Write(ScratchImage.From2D(img), header: DdsHeader.LegacyWhenPossible);

        Assert.Equal(128 + img.SlicePitch, dds.Length); // no DX10 extension
        Assert.Equal(fourCC, System.Text.Encoding.ASCII.GetString(dds, 84, 4));
        var back = Dds.Read(dds);
        Assert.Equal(format, back.Metadata.Format);
        Assert.Equal(img.Pixels, back.Images[0].Pixels);
    }

    [Theory]
    [InlineData(DxgiFormat.BC7_UNORM)]
    [InlineData(DxgiFormat.BC1_UNORM_SRGB)]
    [InlineData(DxgiFormat.R8G8B8A8_UNORM_SRGB)]
    public void FormatsWithoutALegacyHeaderStayDx10(DxgiFormat format)
    {
        var dds = Dds.Write(ScratchImage.From2D(Filled(format, 8, 8, 1)), header: DdsHeader.LegacyWhenPossible);

        Assert.Equal("DX10", System.Text.Encoding.ASCII.GetString(dds, 84, 4));
        Assert.Equal(format, Dds.Read(dds).Metadata.Format);
    }

    [Fact]
    public void PremultipliedAlphaNeedsTheDx10Header()
    {
        var dds = Dds.Write(ScratchImage.From2D(Filled(DxgiFormat.BC3_UNORM, 4, 4, 1)), DdsAlphaMode.Premultiplied,
            DdsHeader.LegacyWhenPossible);

        Assert.Equal("DX10", System.Text.Encoding.ASCII.GetString(dds, 84, 4));
    }

    [Theory]
    [InlineData(DxgiFormat.B8G8R8A8_UNORM)]
    [InlineData(DxgiFormat.B8G8R8X8_UNORM)]
    [InlineData(DxgiFormat.R8G8B8A8_UNORM)]
    [InlineData(DxgiFormat.R8_UNORM)]
    [InlineData(DxgiFormat.R8G8_UNORM)]
    [InlineData(DxgiFormat.R16_UNORM)]
    public void UncompressedFormatsRoundTripThroughABitmaskHeader(DxgiFormat format)
    {
        var img = Filled(format, 5, 3, 7);
        var dds = Dds.Write(ScratchImage.From2D(img), header: DdsHeader.LegacyWhenPossible);

        Assert.Equal(128 + img.SlicePitch, dds.Length);
        Assert.Equal(0u, U32At(dds, 80) & 0x4); // not FourCC
        var back = Dds.Read(dds);
        Assert.Equal(format, back.Metadata.Format);
        Assert.Equal(img.Pixels, back.Images[0].Pixels);
    }

    [Theory]
    [InlineData(DdsHeader.Dx10)]
    [InlineData(DdsHeader.LegacyWhenPossible)]
    public void CubemapsRoundTripEveryFaceAndMip(DdsHeader header)
    {
        var cube = Cube(DxgiFormat.BC1_UNORM);
        var dds = Dds.Write(cube, header: header);

        var back = Dds.Read(dds);
        Assert.Equal(6, back.Metadata.ArraySize);
        Assert.Equal(2, back.Metadata.MipLevels);
        Assert.NotEqual(0u, back.Metadata.MiscFlags & TexMetadata.TexMiscTextureCube);
        Assert.Equal(12, back.Images.Count);
        for (int i = 0; i < 12; i++) Assert.Equal(cube.Images[i].Pixels, back.Images[i].Pixels);
        if (header == DdsHeader.Dx10) Assert.Equal(1u, U32At(dds, 128 + 12)); // one cube, not six slices
        else Assert.Equal(0xFE00u, U32At(dds, 112)); // DDSCAPS2_CUBEMAP | all faces
    }

    [Fact]
    public void ArraysThatAreNotCubesNeedDx10()
    {
        var images = new List<Image> { Filled(DxgiFormat.BC1_UNORM, 4, 4, 1), Filled(DxgiFormat.BC1_UNORM, 4, 4, 2) };
        var meta = new TexMetadata(4, 4, 1, 2, 1, 0, 0, DxgiFormat.BC1_UNORM, TexDimension.Texture2D);
        var dds = Dds.Write(ScratchImage.FromImages(meta, images), header: DdsHeader.LegacyWhenPossible);

        Assert.Equal("DX10", System.Text.Encoding.ASCII.GetString(dds, 84, 4));
        var back = Dds.Read(dds);
        Assert.Equal(2, back.Metadata.ArraySize);
        Assert.Equal(images[1].Pixels, back.Images[1].Pixels);
    }

    /// <summary>Hand-build a legacy bitmask DDS with the given pixel bytes.</summary>
    private static byte[] LegacyBitmaskDds(uint pfFlags, uint bits, uint r, uint g, uint b, uint a, int width,
        int height, byte[] pixels)
    {
        var buf = new byte[4 + 124 + pixels.Length];
        void U32(int off, uint v) => BitConverter.TryWriteBytes(buf.AsSpan(off), v);
        U32(0, 0x20534444);
        U32(4, 124);
        U32(8, 0x100F);
        U32(12, (uint)height);
        U32(16, (uint)width);
        U32(76, 32);
        U32(80, pfFlags);
        U32(88, bits);
        U32(92, r);
        U32(96, g);
        U32(100, b);
        U32(104, a);
        pixels.CopyTo(buf, 128);
        return buf;
    }

    [Fact]
    public void TwentyFourBitRgbIsExpandedToRgba8()
    {
        // Two pixels, stored B, G, R.
        var dds = LegacyBitmaskDds(0x40, 24, 0xff0000, 0xff00, 0xff, 0, 2, 1, [10, 20, 30, 40, 50, 60]);

        var back = Dds.Read(dds);
        Assert.Equal(DxgiFormat.R8G8B8A8_UNORM, back.Metadata.Format);
        Assert.Equal(new byte[] { 30, 20, 10, 255, 60, 50, 40, 255 }, back.Images[0].Pixels);
    }

    [Fact]
    public void R5G6B5IsRescaledTo8BitChannels()
    {
        // 0xF800 = full red, 0x07E0 = full green.
        var dds = LegacyBitmaskDds(0x40, 16, 0xF800, 0x07E0, 0x001F, 0, 2, 1, [0x00, 0xF8, 0xE0, 0x07]);

        var back = Dds.Read(dds);
        Assert.Equal(new byte[] { 255, 0, 0, 255, 0, 255, 0, 255 }, back.Images[0].Pixels);
    }

    [Fact]
    public void A4R4G4B4KeepsItsAlpha()
    {
        // 0x8F00: alpha 8/15, red 15/15.
        var dds = LegacyBitmaskDds(0x41, 16, 0x0F00, 0x00F0, 0x000F, 0xF000, 1, 1, [0x00, 0x8F]);

        var back = Dds.Read(dds);
        Assert.Equal(new byte[] { 255, 0, 0, 136 }, back.Images[0].Pixels);
    }

    [Fact]
    public void X8B8G8R8IgnoresWhateverIsInItsPaddingByte()
    {
        var dds = LegacyBitmaskDds(0x40, 32, 0xff, 0xff00, 0xff0000, 0xff000000, 1, 1, [1, 2, 3, 99]);

        var back = Dds.Read(dds);
        Assert.Equal(new byte[] { 1, 2, 3, 255 }, back.Images[0].Pixels);
    }
}
