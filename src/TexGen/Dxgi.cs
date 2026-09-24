//-------------------------------------------------------------------------------------
// Dxgi.cs
//
// DXGI_FORMAT enumeration and format metadata, ported from DirectXTex
// (DirectXTexUtil.cpp) by way of texconv-js. Member names deliberately keep the
// DXGI spelling so texconv -f names resolve directly via Enum.TryParse.
//-------------------------------------------------------------------------------------

namespace TexGen;

/// <summary>Standard DXGI_FORMAT numeric values (a focused subset).</summary>
public enum DxgiFormat : uint
{
    UNKNOWN = 0,
    R32G32B32A32_TYPELESS = 1,
    R32G32B32A32_FLOAT = 2,
    R16G16B16A16_FLOAT = 10,
    R16G16B16A16_UNORM = 11,
    R8G8B8A8_TYPELESS = 27,
    R8G8B8A8_UNORM = 28,
    R8G8B8A8_UNORM_SRGB = 29,
    R32_FLOAT = 41,
    R8G8_UNORM = 49,
    R16_UNORM = 56,
    R8_UNORM = 61,
    BC1_TYPELESS = 70,
    BC1_UNORM = 71,
    BC1_UNORM_SRGB = 72,
    BC2_TYPELESS = 73,
    BC2_UNORM = 74,
    BC2_UNORM_SRGB = 75,
    BC3_TYPELESS = 76,
    BC3_UNORM = 77,
    BC3_UNORM_SRGB = 78,
    BC4_TYPELESS = 79,
    BC4_UNORM = 80,
    BC4_SNORM = 81,
    BC5_TYPELESS = 82,
    BC5_UNORM = 83,
    BC5_SNORM = 84,
    B8G8R8A8_UNORM = 87,
    B8G8R8X8_UNORM = 88,
    B8G8R8A8_UNORM_SRGB = 91,
    BC6H_TYPELESS = 94,
    BC6H_UF16 = 95,
    BC6H_SF16 = 96,
    BC7_TYPELESS = 97,
    BC7_UNORM = 98,
    BC7_UNORM_SRGB = 99,
}

/// <summary>Flags mirroring DirectXTex CP_FLAGS (only the ones we use).</summary>
[Flags]
public enum CpFlags
{
    None = 0,
    /// <summary>Legacy DXTn nonsquare tail handling (DDS_FLAGS_BAD_DXTN_TAILS).</summary>
    BadDxtnTails = 0x10,
}

public readonly record struct Pitch(int RowPitch, int SlicePitch);

public static class Dxgi
{
    /// <summary>True for the BCn block-compressed formats.</summary>
    public static bool IsCompressed(DxgiFormat fmt) =>
        fmt is >= DxgiFormat.BC1_TYPELESS and <= DxgiFormat.BC5_SNORM
            or >= DxgiFormat.BC6H_TYPELESS and <= DxgiFormat.BC7_UNORM_SRGB;

    /// <summary>Bytes per 4x4 block for a compressed format (8 for BC1/BC4, else 16).</summary>
    public static int BlockBytes(DxgiFormat fmt) =>
        fmt is >= DxgiFormat.BC1_TYPELESS and <= DxgiFormat.BC1_UNORM_SRGB
            or >= DxgiFormat.BC4_TYPELESS and <= DxgiFormat.BC4_SNORM
            ? 8
            : 16;

    /// <summary>True for the *_UNORM_SRGB formats.</summary>
    public static bool IsSrgb(DxgiFormat fmt) => fmt switch
    {
        DxgiFormat.R8G8B8A8_UNORM_SRGB or DxgiFormat.BC1_UNORM_SRGB or DxgiFormat.BC2_UNORM_SRGB
            or DxgiFormat.BC3_UNORM_SRGB or DxgiFormat.BC7_UNORM_SRGB or DxgiFormat.B8G8R8A8_UNORM_SRGB => true,
        _ => false,
    };

    /// <summary>Map a UNORM format to its _SRGB variant where one exists (texconv -srgbo).</summary>
    public static DxgiFormat ToSrgb(DxgiFormat fmt) => fmt switch
    {
        DxgiFormat.R8G8B8A8_UNORM => DxgiFormat.R8G8B8A8_UNORM_SRGB,
        DxgiFormat.B8G8R8A8_UNORM => DxgiFormat.B8G8R8A8_UNORM_SRGB,
        DxgiFormat.BC1_UNORM => DxgiFormat.BC1_UNORM_SRGB,
        DxgiFormat.BC2_UNORM => DxgiFormat.BC2_UNORM_SRGB,
        DxgiFormat.BC3_UNORM => DxgiFormat.BC3_UNORM_SRGB,
        DxgiFormat.BC7_UNORM => DxgiFormat.BC7_UNORM_SRGB,
        _ => fmt,
    };

    /// <summary>Bits per pixel for non-block formats (used for uncompressed pitch).</summary>
    public static int BitsPerPixel(DxgiFormat fmt) => fmt switch
    {
        DxgiFormat.R32G32B32A32_TYPELESS or DxgiFormat.R32G32B32A32_FLOAT => 128,
        DxgiFormat.R16G16B16A16_FLOAT or DxgiFormat.R16G16B16A16_UNORM => 64,
        DxgiFormat.R8G8B8A8_TYPELESS or DxgiFormat.R8G8B8A8_UNORM or DxgiFormat.R8G8B8A8_UNORM_SRGB
            or DxgiFormat.B8G8R8A8_UNORM or DxgiFormat.B8G8R8X8_UNORM or DxgiFormat.B8G8R8A8_UNORM_SRGB
            or DxgiFormat.R32_FLOAT => 32,
        DxgiFormat.R8G8_UNORM or DxgiFormat.R16_UNORM => 16,
        DxgiFormat.R8_UNORM => 8,
        _ => 0,
    };

    /// <summary>
    /// Compute the row and slice pitch for a surface, mirroring DirectX::ComputePitch
    /// (the CP_FLAGS_PAGED4K/NV12 cases are omitted).
    /// </summary>
    public static Pitch ComputePitch(DxgiFormat fmt, int width, int height, CpFlags flags = CpFlags.None)
    {
        if (fmt == DxgiFormat.UNKNOWN) throw new ArgumentException("ComputePitch: UNKNOWN format");

        if (IsCompressed(fmt))
        {
            int bpb = BlockBytes(fmt);
            if ((flags & CpFlags.BadDxtnTails) != 0)
            {
                int pitch = Math.Max(1, (width >> 2) * bpb);
                return new Pitch(pitch, Math.Max(1, pitch * (height >> 2)));
            }
            int nbw = Math.Max(1, (width + 3) >> 2);
            int nbh = Math.Max(1, (height + 3) >> 2);
            return new Pitch(nbw * bpb, nbw * bpb * nbh);
        }

        int bpp = BitsPerPixel(fmt);
        if (bpp == 0) throw new NotSupportedException($"ComputePitch: unsupported format {fmt}");
        int rowPitch = (width * bpp + 7) / 8;
        return new Pitch(rowPitch, rowPitch * height);
    }

    /// <summary>Resolve a texconv -f format name (case-insensitive, with aliases) to a DXGI format.</summary>
    public static DxgiFormat? FormatFromName(string name)
    {
        var key = name.ToUpperInvariant();
        if (key.StartsWith("DXGI_FORMAT_", StringComparison.Ordinal)) key = key["DXGI_FORMAT_".Length..];
        switch (key)
        {
            case "DXT1": return DxgiFormat.BC1_UNORM;
            case "DXT2": case "DXT3": return DxgiFormat.BC2_UNORM;
            case "DXT4": case "DXT5": return DxgiFormat.BC3_UNORM;
            case "RGBA": return DxgiFormat.R8G8B8A8_UNORM;
            case "BGRA": return DxgiFormat.B8G8R8A8_UNORM;
            // texconv's g_pFormatAliases
            case "BGR": return DxgiFormat.B8G8R8X8_UNORM;
            case "BPTC": return DxgiFormat.BC7_UNORM;
            case "BPTC_FLOAT": return DxgiFormat.BC6H_UF16;
            case "FP16": return DxgiFormat.R16G16B16A16_FLOAT;
            case "FP32": return DxgiFormat.R32G32B32A32_FLOAT;
        }
        // Enum.TryParse would also accept numeric strings; texconv names only.
        if (key.Length == 0 || char.IsDigit(key[0])) return null;
        return Enum.TryParse<DxgiFormat>(key, ignoreCase: false, out var f) ? f : null;
    }
}
