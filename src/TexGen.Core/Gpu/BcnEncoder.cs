//-------------------------------------------------------------------------------------
// Gpu/BcnEncoder.cs
//
// BC1/BC2/BC3/BC4/BC5 GPU encoders (ILGPU port of texconv-js bcn.wgsl). DirectXTex
// has no GPU reference for these (it compresses them on the CPU), so they follow
// the texconv-js design:
//
// One thread per 4x4 block. Endpoints come from the per-block bounding box with a
// small inset (the van Waveren real-time DXT approach); indices are nearest-palette.
// Palettes match the D3D/DirectXTex decode so output is valid for any DDS consumer:
//   BC1 4-color: p0=c0, p1=c1, p2=(2c0+c1)/3, p3=(c0+2c1)/3
//   BC4 8-value: p0=r0, p1=r1, p2=(6r0+r1)/7 ... p7=(r0+6r1)/7  (r0>r1)
//-------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Runtime;

namespace TexGen.Gpu;

public struct BcnParams
{
    public int Width, Height, NumBlockX, NumTotalBlocks;
    /// <summary>Which encoder: see the Mode* constants.</summary>
    public int Mode;
}

internal sealed class BcnEncoder
{
    private const int ModeBC1 = 1, ModeBC2 = 2, ModeBC3 = 3, ModeBC4 = 4, ModeBC5 = 5;

    private readonly Accelerator _acc;
    private readonly Action<Index1D, BcnParams, ArrayView<uint>, ArrayView<uint>> _kernel;

    public BcnEncoder(Accelerator acc)
    {
        _acc = acc;
        _kernel = acc.LoadAutoGroupedStreamKernel<Index1D, BcnParams, ArrayView<uint>, ArrayView<uint>>(EncodeKernel);
    }

    public static bool Supports(DxgiFormat format) => ModeFor(format) != 0;

    private static int ModeFor(DxgiFormat format) => format switch
    {
        DxgiFormat.BC1_UNORM or DxgiFormat.BC1_UNORM_SRGB => ModeBC1,
        DxgiFormat.BC2_UNORM or DxgiFormat.BC2_UNORM_SRGB => ModeBC2,
        DxgiFormat.BC3_UNORM or DxgiFormat.BC3_UNORM_SRGB => ModeBC3,
        DxgiFormat.BC4_UNORM => ModeBC4,
        DxgiFormat.BC5_UNORM => ModeBC5,
        _ => 0,
    };

    /// <summary>Encode an RGBA8 texture; the result stays on the device until downloaded.</summary>
    public GpuBlocks Encode(GpuTexture src, DxgiFormat format)
    {
        int mode = ModeFor(format);
        if (mode == 0) throw new NotSupportedException($"BcnEncoder: unsupported format {format}");
        if (src.IsFloat) throw new ArgumentException("BcnEncoder: source must be R8G8B8A8_UNORM(_SRGB)");

        int xblocks = Math.Max(1, (src.Width + 3) >> 2);
        int yblocks = Math.Max(1, (src.Height + 3) >> 2);
        int numBlocks = xblocks * yblocks;
        int wordsPerBlock = Dxgi.BlockBytes(format) / 4;

        var output = _acc.Allocate1D<uint>((long)numBlocks * wordsPerBlock);
        var p = new BcnParams
        {
            Width = src.Width,
            Height = src.Height,
            NumBlockX = xblocks,
            NumTotalBlocks = numBlocks,
            Mode = mode,
        };
        _kernel(numBlocks, p, src.Rgba8View, output.View);
        return new GpuBlocks(src.Width, src.Height, format, output);
    }

    //---------------------------------------------------------------------------------
    // Kernel
    //---------------------------------------------------------------------------------

    /// <summary>Texel (px, py) of block (bx, by) as 0..255 floats, edge-clamped for partial blocks.</summary>
    private static Float4 LoadTexel(ArrayView<uint> input, BcnParams p, int bx, int by, int px, int py)
    {
        int x = Math.Min(bx * 4 + px, p.Width - 1);
        int y = Math.Min(by * 4 + py, p.Height - 1);
        uint t = input[y * p.Width + x];
        return new Float4(t & 0xFF, (t >> 8) & 0xFF, (t >> 16) & 0xFF, t >> 24);
    }

    private static uint To565(Float3 c)
    {
        uint r = (uint)MathF.Round(GMath.Clamp(c.X, 0f, 255f) / 255f * 31f);
        uint g = (uint)MathF.Round(GMath.Clamp(c.Y, 0f, 255f) / 255f * 63f);
        uint b = (uint)MathF.Round(GMath.Clamp(c.Z, 0f, 255f) / 255f * 31f);
        return (r << 11) | (g << 5) | b;
    }

    private static Float3 Expand565(uint w)
    {
        uint r = (w >> 11) & 31, g = (w >> 5) & 63, b = w & 31;
        return new Float3((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2));
    }

    /// <summary>BC1 color block: low word = c0 | c1 &lt;&lt; 16, high word = 2-bit indices.</summary>
    private static ulong EncodeBC1Color(ArrayView<uint> input, BcnParams p, int bx, int by)
    {
        var mn = new Float3(255f);
        var mx = new Float3(0f);
        for (int i = 0; i < 16; i++)
        {
            var c = LoadTexel(input, p, bx, by, i & 3, i >> 2).Xyz;
            mn = Float3.Min(mn, c);
            mx = Float3.Max(mx, c);
        }
        // Inset the bounding box slightly (reduces error for interior samples).
        var inset = (mx - mn) / 16f;
        mn = Float3.Min(mn + inset, new Float3(255f));
        mx = Float3.Max(mx - inset, new Float3(0f));

        uint c0 = To565(mx), c1 = To565(mn);
        var e0 = Expand565(c0);
        var e1 = Expand565(c1);
        // 4-color mode requires c0 > c1 (c0 >= c1 always holds since mx >= mn).
        if (c0 < c1)
        {
            (c0, c1) = (c1, c0);
            (e0, e1) = (e1, e0);
        }
        var p2 = (2f * e0 + e1) / 3f;
        var p3 = (e0 + 2f * e1) / 3f;

        uint indices = 0;
        for (int i = 0; i < 16; i++)
        {
            var c = LoadTexel(input, p, bx, by, i & 3, i >> 2).Xyz;
            uint best = 0;
            var d = c - e0;
            float bestd = Float3.Dot(d, d);
            d = c - e1;
            float dist = Float3.Dot(d, d);
            if (dist < bestd) { bestd = dist; best = 1; }
            d = c - p2;
            dist = Float3.Dot(d, d);
            if (dist < bestd) { bestd = dist; best = 2; }
            d = c - p3;
            dist = Float3.Dot(d, d);
            if (dist < bestd) best = 3;
            indices |= best << (i * 2);
        }
        return (c0 | (c1 << 16)) | ((ulong)indices << 32);
    }

    /// <summary>BC4 block for one channel (0=R .. 3=A) as the 64-bit block value.</summary>
    private static ulong EncodeBC4(ArrayView<uint> input, BcnParams p, int bx, int by, int channel)
    {
        float mn = 255f, mx = 0f;
        for (int i = 0; i < 16; i++)
        {
            float v = LoadTexel(input, p, bx, by, i & 3, i >> 2)[channel];
            mn = MathF.Min(mn, v);
            mx = MathF.Max(mx, v);
        }
        uint r0 = (uint)MathF.Round(mx); // r0 >= r1
        uint r1 = (uint)MathF.Round(mn);

        ulong block = r0 | (r1 << 8);
        for (int i = 0; i < 16; i++)
        {
            float v = LoadTexel(input, p, bx, by, i & 3, i >> 2)[channel];
            uint best = 0;
            float bestd = 1e30f;
            for (uint k = 0; k < 8; k++)
            {
                float pal = k == 0 ? r0 : k == 1 ? r1 : (r0 * (float)(8 - k) + r1 * (float)(k - 1)) / 7f;
                float d = MathF.Abs(pal - v);
                if (d < bestd) { bestd = d; best = k; }
            }
            block |= (ulong)best << (16 + i * 3); // indices begin at bit 16
        }
        return block;
    }

    /// <summary>BC2 explicit 4-bit alpha block.</summary>
    private static ulong EncodeBC2Alpha(ArrayView<uint> input, BcnParams p, int bx, int by)
    {
        ulong block = 0;
        for (int i = 0; i < 16; i++)
        {
            float a = LoadTexel(input, p, bx, by, i & 3, i >> 2).W;
            ulong a4 = (ulong)MathF.Round(GMath.Clamp(a, 0f, 255f) / 255f * 15f);
            block |= a4 << (i * 4);
        }
        return block;
    }

    private static void Store(ArrayView<uint> output, long word, ulong v)
    {
        output[word] = (uint)v;
        output[word + 1] = (uint)(v >> 32);
    }

    private static void EncodeKernel(Index1D blockID, BcnParams p, ArrayView<uint> input, ArrayView<uint> output)
    {
        int bx = blockID % p.NumBlockX;
        int by = blockID / p.NumBlockX;
        long b = blockID;
        switch (p.Mode)
        {
            case ModeBC1:
                Store(output, b * 2, EncodeBC1Color(input, p, bx, by));
                break;
            case ModeBC2:
                Store(output, b * 4, EncodeBC2Alpha(input, p, bx, by));
                Store(output, b * 4 + 2, EncodeBC1Color(input, p, bx, by));
                break;
            case ModeBC3:
                Store(output, b * 4, EncodeBC4(input, p, bx, by, 3));
                Store(output, b * 4 + 2, EncodeBC1Color(input, p, bx, by));
                break;
            case ModeBC4:
                Store(output, b * 2, EncodeBC4(input, p, bx, by, 0));
                break;
            default:
                Store(output, b * 4, EncodeBC4(input, p, bx, by, 0));
                Store(output, b * 4 + 2, EncodeBC4(input, p, bx, by, 1));
                break;
        }
    }
}
