//-------------------------------------------------------------------------------------
// Dds.cs
//
// DDS container read/write. The writer always emits the modern DX10 extended
// header (matching texconv's default), which carries the DXGI_FORMAT directly and
// avoids the legacy FourCC/bitmask ambiguity. The reader covers the DX10 and common
// legacy-FourCC cases for round-tripping.
//-------------------------------------------------------------------------------------

using System.Buffers.Binary;

namespace TexGen;

public enum DdsAlphaMode : uint
{
    Unknown = 0,
    Straight = 1,
    Premultiplied = 2,
    Opaque = 3,
    Custom = 4,
}

public static class Dds
{
    private const uint Magic = 0x20534444; // "DDS "

    // DDS_HEADER flags
    private const uint DDSD_CAPS = 0x1, DDSD_HEIGHT = 0x2, DDSD_WIDTH = 0x4, DDSD_PITCH = 0x8;
    private const uint DDSD_PIXELFORMAT = 0x1000, DDSD_MIPMAPCOUNT = 0x20000, DDSD_LINEARSIZE = 0x80000;
    private const uint DDS_HEADER_FLAGS_TEXTURE = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT;

    private const uint DDPF_FOURCC = 0x4;

    private const uint DDSCAPS_COMPLEX = 0x8, DDSCAPS_TEXTURE = 0x1000, DDSCAPS_MIPMAP = 0x400000;

    private const int HeaderSize = 124; // sizeof(DDS_HEADER)
    private const int PixelFormatSize = 32;
    private const int Dx10ExtSize = 20; // sizeof(DDS_HEADER_DXT10)

    private static uint FourCC(string s) =>
        (uint)(s[0] & 0xff) | ((uint)(s[1] & 0xff) << 8) | ((uint)(s[2] & 0xff) << 16) | ((uint)(s[3] & 0xff) << 24);

    // Legacy FourCC -> DXGI_FORMAT for the common reader path.
    private static readonly Dictionary<uint, DxgiFormat> FourCCToDxgi = new()
    {
        [FourCC("DXT1")] = DxgiFormat.BC1_UNORM,
        [FourCC("DXT2")] = DxgiFormat.BC2_UNORM,
        [FourCC("DXT3")] = DxgiFormat.BC2_UNORM,
        [FourCC("DXT4")] = DxgiFormat.BC3_UNORM,
        [FourCC("DXT5")] = DxgiFormat.BC3_UNORM,
        [FourCC("BC4U")] = DxgiFormat.BC4_UNORM,
        [FourCC("BC4S")] = DxgiFormat.BC4_SNORM,
        [FourCC("ATI1")] = DxgiFormat.BC4_UNORM,
        [FourCC("BC5U")] = DxgiFormat.BC5_UNORM,
        [FourCC("BC5S")] = DxgiFormat.BC5_SNORM,
        [FourCC("ATI2")] = DxgiFormat.BC5_UNORM,
    };

    /// <summary>
    /// Serialize a ScratchImage to DDS bytes using the DX10 extended header. Images are
    /// written in DirectXTex subresource order (mip 0..N within array slice 0).
    /// </summary>
    public static byte[] Write(ScratchImage scratch, DdsAlphaMode alphaMode = DdsAlphaMode.Unknown)
    {
        var meta = scratch.Metadata;
        const int headerBytes = 4 + HeaderSize + Dx10ExtSize;
        long dataBytes = 0;
        foreach (var img in scratch.Images) dataBytes += img.SlicePitch;

        var bytes = new byte[headerBytes + dataBytes];
        int o = 0;
        void U32(uint v)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(o), v);
            o += 4;
        }

        U32(Magic);

        bool compressed = Dxgi.IsCompressed(meta.Format);
        var b = scratch.Images[0];
        uint flags = DDS_HEADER_FLAGS_TEXTURE;
        if (meta.MipLevels > 1) flags |= DDSD_MIPMAPCOUNT;
        flags |= compressed ? DDSD_LINEARSIZE : DDSD_PITCH;

        // --- DDS_HEADER (124 bytes) ---
        U32(HeaderSize);
        U32(flags);
        U32((uint)meta.Height);
        U32((uint)meta.Width);
        U32((uint)(compressed ? b.SlicePitch : b.RowPitch)); // pitchOrLinearSize
        U32(meta.Dimension == TexDimension.Texture3D ? (uint)meta.Depth : 0);
        U32((uint)meta.MipLevels);
        for (int i = 0; i < 11; i++) U32(0); // reserved1[11]

        // DDS_PIXELFORMAT (32 bytes) — always the DX10 marker
        U32(PixelFormatSize);
        U32(DDPF_FOURCC);
        U32(FourCC("DX10"));
        for (int i = 0; i < 5; i++) U32(0); // RGBBitCount + 4 masks

        uint caps = DDSCAPS_TEXTURE;
        if (meta.MipLevels > 1) caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP;
        U32(caps);
        U32(0); // caps2
        U32(0); // caps3
        U32(0); // caps4
        U32(0); // reserved2

        // --- DDS_HEADER_DXT10 (20 bytes) ---
        U32((uint)meta.Format);
        U32((uint)meta.Dimension);
        U32(meta.MiscFlags);
        U32((uint)meta.ArraySize);
        U32((uint)alphaMode); // miscFlags2

        // --- pixel data ---
        foreach (var img in scratch.Images)
        {
            img.Pixels.AsSpan(0, img.SlicePitch).CopyTo(bytes.AsSpan(o));
            o += img.SlicePitch;
        }
        return bytes;
    }

    private static uint ReadU32(ReadOnlySpan<byte> data, int offset) =>
        BinaryPrimitives.ReadUInt32LittleEndian(data[offset..]);

    /// <summary>
    /// Parse DDS bytes into a ScratchImage. Supports DX10-extended headers and the common
    /// legacy DXTn/BCn FourCC tags. Only the slice-0 2D mip chain is materialized.
    /// </summary>
    public static ScratchImage Read(ReadOnlySpan<byte> data)
    {
        if (data.Length < 4 + HeaderSize) throw new InvalidDataException("DDS: too small");
        if (ReadU32(data, 0) != Magic) throw new InvalidDataException("DDS: bad magic");
        if (ReadU32(data, 4) != HeaderSize) throw new InvalidDataException("DDS: bad header size");

        int height = (int)ReadU32(data, 12);
        int width = (int)ReadU32(data, 16);
        int mipLevels = Math.Max(1, (int)ReadU32(data, 28));

        // DDS_PIXELFORMAT begins at offset 4 + 72 = 76
        uint pfFlags = ReadU32(data, 80);
        uint pfFourCC = ReadU32(data, 84);

        int offset = 4 + HeaderSize;
        DxgiFormat format;
        int arraySize = 1;
        uint miscFlags = 0, miscFlags2 = 0;
        var dimension = TexDimension.Texture2D;

        if ((pfFlags & DDPF_FOURCC) != 0 && pfFourCC == FourCC("DX10"))
        {
            if (data.Length < offset + Dx10ExtSize) throw new InvalidDataException("DDS: truncated DX10 header");
            format = (DxgiFormat)ReadU32(data, offset);
            dimension = (TexDimension)ReadU32(data, offset + 4);
            miscFlags = ReadU32(data, offset + 8);
            arraySize = Math.Max(1, (int)ReadU32(data, offset + 12));
            miscFlags2 = ReadU32(data, offset + 16);
            offset += Dx10ExtSize;
        }
        else if ((pfFlags & DDPF_FOURCC) != 0)
        {
            if (!FourCCToDxgi.TryGetValue(pfFourCC, out format))
                throw new InvalidDataException($"DDS: unsupported FourCC 0x{pfFourCC:x}");
        }
        else
        {
            throw new InvalidDataException("DDS: legacy uncompressed bitmask layouts not yet supported");
        }

        if (format == DxgiFormat.UNKNOWN) throw new InvalidDataException("DDS: unknown format");
        if (!Dxgi.IsCompressed(format) && Dxgi.BitsPerPixel(format) == 0)
            throw new InvalidDataException($"DDS: unsupported format {format}");

        // Reject array, cubemap, and volume textures rather than returning metadata
        // that disagrees with the images actually read.
        if (arraySize > 1) throw new InvalidDataException($"DDS: array textures not yet supported (arraySize={arraySize})");
        if (dimension == TexDimension.Texture3D) throw new InvalidDataException("DDS: 3D/volume textures not yet supported");

        var images = new List<Image>(mipLevels);
        int w = width, h = height;
        for (int m = 0; m < mipLevels; m++)
        {
            var (rowPitch, slicePitch) = Dxgi.ComputePitch(format, w, h);
            if (offset + (long)slicePitch > data.Length) throw new InvalidDataException("DDS: truncated pixel data");
            images.Add(new Image(w, h, format, rowPitch, slicePitch, data.Slice(offset, slicePitch).ToArray()));
            offset += slicePitch;
            w = Math.Max(1, w >> 1);
            h = Math.Max(1, h >> 1);
        }

        var metadata = new TexMetadata(width, height, 1, arraySize, mipLevels, miscFlags, miscFlags2, format, dimension);
        return ScratchImage.FromImages(metadata, images);
    }
}
