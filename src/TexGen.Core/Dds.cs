//-------------------------------------------------------------------------------------
// Dds.cs
//
// DDS container read/write. The writer emits the DX10 extended header by default, or,
// with DdsHeader.LegacyWhenPossible, the pre-DX10 FourCC/bitmask header whenever the
// format and layout fit one, as texconv does by default. DX9-era games (Oblivion,
// Fallout 3/New Vegas, Skyrim LE) only read the legacy header. The reader covers DX10
// headers, legacy FourCC tags, legacy bitmask layouts, cubemaps and arrays.
//-------------------------------------------------------------------------------------

using System.Buffers.Binary;
using System.Numerics;

namespace TexGen;

public enum DdsAlphaMode : uint
{
    Unknown = 0,
    Straight = 1,
    Premultiplied = 2,
    Opaque = 3,
    Custom = 4,
}

/// <summary>Which header <see cref="Dds.Write"/> emits.</summary>
public enum DdsHeader
{
    /// <summary>Always the DX10 extended header.</summary>
    Dx10,

    /// <summary>
    /// The legacy header when the format and layout fit one (texconv's default), otherwise DX10.
    /// sRGB formats, BC6H/BC7, premultiplied alpha and non-cube arrays need DX10.
    /// </summary>
    LegacyWhenPossible,
}

public static class Dds
{
    private const uint Magic = 0x20534444; // "DDS "

    // DDS_HEADER flags
    private const uint DDSD_CAPS = 0x1, DDSD_HEIGHT = 0x2, DDSD_WIDTH = 0x4, DDSD_PITCH = 0x8;
    private const uint DDSD_PIXELFORMAT = 0x1000, DDSD_MIPMAPCOUNT = 0x20000, DDSD_LINEARSIZE = 0x80000;
    private const uint DDS_HEADER_FLAGS_TEXTURE = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT;

    // DDS_PIXELFORMAT flags
    private const uint DDPF_ALPHAPIXELS = 0x1, DDPF_ALPHA = 0x2, DDPF_FOURCC = 0x4, DDPF_RGB = 0x40;
    private const uint DDPF_LUMINANCE = 0x20000;

    private const uint DDSCAPS_COMPLEX = 0x8, DDSCAPS_TEXTURE = 0x1000, DDSCAPS_MIPMAP = 0x400000;
    private const uint DDSCAPS2_CUBEMAP = 0x200, DDSCAPS2_CUBEMAP_ALLFACES = 0xFC00, DDSCAPS2_VOLUME = 0x200000;

    private const int HeaderSize = 124; // sizeof(DDS_HEADER)
    private const int PixelFormatSize = 32;
    private const int Dx10ExtSize = 20; // sizeof(DDS_HEADER_DXT10)

    private static uint FourCC(string s) =>
        (uint)(s[0] & 0xff) | ((uint)(s[1] & 0xff) << 8) | ((uint)(s[2] & 0xff) << 16) | ((uint)(s[3] & 0xff) << 24);

    // Legacy FourCC -> DXGI_FORMAT. The small numbers are D3DFORMAT codes stored in the FourCC field.
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
        [36] = DxgiFormat.R16G16B16A16_UNORM,
        [113] = DxgiFormat.R16G16B16A16_FLOAT,
        [114] = DxgiFormat.R32_FLOAT,
        [116] = DxgiFormat.R32G32B32A32_FLOAT,
    };

    // DXGI_FORMAT -> legacy FourCC, as DirectXTex writes it when the DX10 header is not forced.
    private static readonly Dictionary<DxgiFormat, uint> DxgiToFourCC = new()
    {
        [DxgiFormat.BC1_UNORM] = FourCC("DXT1"),
        [DxgiFormat.BC2_UNORM] = FourCC("DXT3"),
        [DxgiFormat.BC3_UNORM] = FourCC("DXT5"),
        [DxgiFormat.BC4_UNORM] = FourCC("BC4U"),
        [DxgiFormat.BC4_SNORM] = FourCC("BC4S"),
        [DxgiFormat.BC5_UNORM] = FourCC("BC5U"),
        [DxgiFormat.BC5_SNORM] = FourCC("BC5S"),
        [DxgiFormat.R16G16B16A16_UNORM] = 36,
        [DxgiFormat.R16G16B16A16_FLOAT] = 113,
        [DxgiFormat.R32_FLOAT] = 114,
        [DxgiFormat.R32G32B32A32_FLOAT] = 116,
    };

    private readonly record struct Bitmask(uint Flags, uint BitCount, uint R, uint G, uint B, uint A);

    // DXGI_FORMAT -> legacy bitmask pixel format (DirectXTex's DDSPF_* constants).
    private static readonly Dictionary<DxgiFormat, Bitmask> DxgiToBitmask = new()
    {
        [DxgiFormat.R8G8B8A8_UNORM] = new(DDPF_RGB | DDPF_ALPHAPIXELS, 32, 0xff, 0xff00, 0xff0000, 0xff000000),
        [DxgiFormat.B8G8R8A8_UNORM] = new(DDPF_RGB | DDPF_ALPHAPIXELS, 32, 0xff0000, 0xff00, 0xff, 0xff000000),
        [DxgiFormat.B8G8R8X8_UNORM] = new(DDPF_RGB, 32, 0xff0000, 0xff00, 0xff, 0),
        [DxgiFormat.R8_UNORM] = new(DDPF_LUMINANCE, 8, 0xff, 0, 0, 0),
        [DxgiFormat.R16_UNORM] = new(DDPF_LUMINANCE, 16, 0xffff, 0, 0, 0),
        [DxgiFormat.R8G8_UNORM] = new(DDPF_LUMINANCE | DDPF_ALPHAPIXELS, 16, 0xff, 0, 0, 0xff00),
    };

    /// <summary>
    /// Serialize a ScratchImage to DDS bytes. Images are written in DirectXTex subresource
    /// order: every mip of array slice (or cube face) 0, then every mip of slice 1, and so on.
    /// A cubemap has an <c>ArraySize</c> that is a multiple of 6 and
    /// <see cref="TexMetadata.TexMiscTextureCube"/> set.
    /// </summary>
    public static byte[] Write(ScratchImage scratch, DdsAlphaMode alphaMode = DdsAlphaMode.Unknown,
        DdsHeader header = DdsHeader.Dx10)
    {
        var meta = scratch.Metadata;
        bool cube = (meta.MiscFlags & TexMetadata.TexMiscTextureCube) != 0;
        if (cube && meta.ArraySize % 6 != 0)
            throw new ArgumentException($"DDS: a cubemap needs a multiple of 6 faces; got {meta.ArraySize}");

        uint legacyFourCC = 0;
        Bitmask? legacyMask = null;
        bool legacy = header == DdsHeader.LegacyWhenPossible
                      && meta.Dimension == TexDimension.Texture2D
                      && meta.ArraySize == (cube ? 6 : 1)
                      && alphaMode != DdsAlphaMode.Premultiplied;
        if (legacy)
        {
            if (DxgiToBitmask.TryGetValue(meta.Format, out var mask)) legacyMask = mask;
            else if (!DxgiToFourCC.TryGetValue(meta.Format, out legacyFourCC)) legacy = false;
        }

        int headerBytes = 4 + HeaderSize + (legacy ? 0 : Dx10ExtSize);
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

        // DDS_PIXELFORMAT (32 bytes)
        U32(PixelFormatSize);
        if (!legacy)
        {
            U32(DDPF_FOURCC);
            U32(FourCC("DX10"));
            for (int i = 0; i < 5; i++) U32(0); // RGBBitCount + 4 masks
        }
        else if (legacyMask is { } mask)
        {
            U32(mask.Flags);
            U32(0);
            U32(mask.BitCount);
            U32(mask.R);
            U32(mask.G);
            U32(mask.B);
            U32(mask.A);
        }
        else
        {
            U32(DDPF_FOURCC);
            U32(legacyFourCC);
            for (int i = 0; i < 5; i++) U32(0);
        }

        uint caps = DDSCAPS_TEXTURE;
        if (meta.MipLevels > 1) caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP;
        if (cube) caps |= DDSCAPS_COMPLEX;
        U32(caps);
        U32(cube ? DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_ALLFACES : 0); // caps2
        U32(0); // caps3
        U32(0); // caps4
        U32(0); // reserved2

        if (!legacy)
        {
            // --- DDS_HEADER_DXT10 (20 bytes) ---
            U32((uint)meta.Format);
            U32((uint)meta.Dimension);
            U32(meta.MiscFlags);
            U32((uint)(cube ? meta.ArraySize / 6 : meta.ArraySize)); // a cube counts once, not per face
            U32((uint)alphaMode); // miscFlags2
        }

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
    /// Parse DDS bytes into a ScratchImage holding every mip of every array slice or cube face,
    /// in the order <see cref="Write"/> takes them. Supports DX10 headers, the legacy DXTn/BCn
    /// and D3DFORMAT FourCC tags, and legacy bitmask layouts. Bitmask layouts with a DXGI
    /// equivalent (A8R8G8B8, X8R8G8B8, A8B8G8R8, L8, L16, A8L8) keep it; any other 8/16/24/32-bit
    /// layout (R8G8B8, R5G6B5, A1R5G5B5, A4R4G4B4, A8, X8B8G8R8, ...) is expanded to R8G8B8A8_UNORM.
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
        var pfMask = new Bitmask(pfFlags, ReadU32(data, 88), ReadU32(data, 92), ReadU32(data, 96),
            ReadU32(data, 100), ReadU32(data, 104));
        uint caps2 = ReadU32(data, 112);

        int offset = 4 + HeaderSize;
        DxgiFormat format;
        int arraySize = 1;
        uint miscFlags = 0, miscFlags2 = 0;
        var dimension = TexDimension.Texture2D;
        Bitmask? expand = null;

        if ((pfFlags & DDPF_FOURCC) != 0 && pfFourCC == FourCC("DX10"))
        {
            if (data.Length < offset + Dx10ExtSize) throw new InvalidDataException("DDS: truncated DX10 header");
            format = (DxgiFormat)ReadU32(data, offset);
            dimension = (TexDimension)ReadU32(data, offset + 4);
            miscFlags = ReadU32(data, offset + 8);
            arraySize = Math.Max(1, (int)ReadU32(data, offset + 12));
            miscFlags2 = ReadU32(data, offset + 16);
            offset += Dx10ExtSize;
            if ((miscFlags & TexMetadata.TexMiscTextureCube) != 0) arraySize *= 6;
        }
        else
        {
            if ((caps2 & DDSCAPS2_VOLUME) != 0)
                throw new InvalidDataException("DDS: 3D/volume textures not yet supported");
            if ((caps2 & DDSCAPS2_CUBEMAP) != 0)
            {
                // DirectXTex refuses a legacy cubemap that leaves faces out, and so does this.
                if ((caps2 & DDSCAPS2_CUBEMAP_ALLFACES) != DDSCAPS2_CUBEMAP_ALLFACES)
                    throw new InvalidDataException("DDS: partial cubemaps are not supported");
                arraySize = 6;
                miscFlags |= TexMetadata.TexMiscTextureCube;
            }

            if ((pfFlags & DDPF_FOURCC) != 0)
            {
                if (!FourCCToDxgi.TryGetValue(pfFourCC, out format))
                    throw new InvalidDataException($"DDS: unsupported FourCC 0x{pfFourCC:x}");
            }
            else if ((pfFlags & (DDPF_RGB | DDPF_LUMINANCE | DDPF_ALPHA)) != 0)
            {
                format = BitmaskFormat(pfMask);
                if (format == DxgiFormat.UNKNOWN)
                {
                    if (pfMask.BitCount is not (8 or 16 or 24 or 32))
                        throw new InvalidDataException($"DDS: unsupported {pfMask.BitCount}-bit bitmask layout");
                    expand = pfMask;
                    format = DxgiFormat.R8G8B8A8_UNORM;
                }
            }
            else
            {
                throw new InvalidDataException($"DDS: unsupported pixel format flags 0x{pfFlags:x}");
            }
        }

        if (format == DxgiFormat.UNKNOWN) throw new InvalidDataException("DDS: unknown format");
        if (!Dxgi.IsCompressed(format) && Dxgi.BitsPerPixel(format) == 0)
            throw new InvalidDataException($"DDS: unsupported format {format}");
        if (dimension == TexDimension.Texture3D) throw new InvalidDataException("DDS: 3D/volume textures not yet supported");

        var images = new List<Image>(mipLevels * arraySize);
        for (int slice = 0; slice < arraySize; slice++)
        {
            int w = width, h = height;
            for (int m = 0; m < mipLevels; m++)
            {
                if (expand is { } mask)
                {
                    int srcPitch = (w * (int)mask.BitCount + 7) / 8;
                    long size = (long)srcPitch * h;
                    if (offset + size > data.Length) throw new InvalidDataException("DDS: truncated pixel data");
                    images.Add(Expand(data.Slice(offset, (int)size), w, h, srcPitch, mask));
                    offset += (int)size;
                }
                else
                {
                    var (rowPitch, slicePitch) = Dxgi.ComputePitch(format, w, h);
                    if (offset + (long)slicePitch > data.Length) throw new InvalidDataException("DDS: truncated pixel data");
                    images.Add(new Image(w, h, format, rowPitch, slicePitch, data.Slice(offset, slicePitch).ToArray()));
                    offset += slicePitch;
                }
                w = Math.Max(1, w >> 1);
                h = Math.Max(1, h >> 1);
            }
        }

        var metadata = new TexMetadata(width, height, 1, arraySize, mipLevels, miscFlags, miscFlags2, format, dimension);
        return ScratchImage.FromImages(metadata, images);
    }

    /// <summary>The DXGI format a legacy bitmask layout is stored as unchanged, or UNKNOWN.</summary>
    private static DxgiFormat BitmaskFormat(Bitmask m)
    {
        foreach (var (format, known) in DxgiToBitmask)
        {
            bool sameKind = ((m.Flags ^ known.Flags) & (DDPF_RGB | DDPF_LUMINANCE)) == 0;
            // Without DDPF_ALPHAPIXELS the alpha mask means nothing, whatever a writer left in it.
            uint alpha = (m.Flags & DDPF_ALPHAPIXELS) != 0 ? m.A : 0;
            if (sameKind && m.BitCount == known.BitCount && m.R == known.R && m.G == known.G && m.B == known.B
                && alpha == known.A)
                return format;
        }
        return DxgiFormat.UNKNOWN;
    }

    /// <summary>Expand one surface of an 8/16/24/32-bit bitmask layout to R8G8B8A8_UNORM.</summary>
    private static Image Expand(ReadOnlySpan<byte> src, int width, int height, int srcPitch, Bitmask m)
    {
        var dst = Image.Create(DxgiFormat.R8G8B8A8_UNORM, width, height);
        int bytesPerPixel = (int)m.BitCount / 8;
        bool luminance = (m.Flags & DDPF_LUMINANCE) != 0;
        bool alphaOnly = (m.Flags & DDPF_ALPHA) != 0 && (m.Flags & (DDPF_RGB | DDPF_LUMINANCE)) == 0;
        // DDPF_ALPHA keeps the alpha mask in the A slot; some writers leave it in R.
        uint alphaMask = alphaOnly && m.A == 0 ? m.R : m.A;
        bool hasAlpha = alphaMask != 0 && (alphaOnly || (m.Flags & DDPF_ALPHAPIXELS) != 0);

        for (int y = 0; y < height; y++)
        {
            var row = src.Slice(y * srcPitch);
            var o = dst.Pixels.AsSpan(y * dst.RowPitch, width * 4);
            for (int x = 0; x < width; x++)
            {
                uint p = 0;
                for (int i = 0; i < bytesPerPixel; i++) p |= (uint)row[x * bytesPerPixel + i] << (8 * i);
                byte r = 0, g = 0, b = 0;
                if (luminance)
                {
                    r = g = b = Channel(p, m.R);
                }
                else if (!alphaOnly)
                {
                    r = Channel(p, m.R);
                    g = Channel(p, m.G);
                    b = Channel(p, m.B);
                }
                o[x * 4] = r;
                o[x * 4 + 1] = g;
                o[x * 4 + 2] = b;
                o[x * 4 + 3] = hasAlpha ? Channel(p, alphaMask) : (byte)255;
            }
        }
        return dst;
    }

    /// <summary>One channel of a packed pixel, rescaled from its mask's width to 8 bits.</summary>
    private static byte Channel(uint pixel, uint mask)
    {
        if (mask == 0) return 0;
        int shift = BitOperations.TrailingZeroCount(mask);
        int bits = BitOperations.PopCount(mask);
        ulong max = (1UL << bits) - 1;
        ulong v = (pixel & mask) >> shift;
        return bits == 8 ? (byte)v : (byte)((v * 255 + max / 2) / max);
    }
}
