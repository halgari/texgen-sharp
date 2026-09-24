//-------------------------------------------------------------------------------------
// Gpu/Bc7Encoder.cs
//
// BC7 GPU encoder: ILGPU port of texconv-js bc7.wgsl, which is itself a faithful
// port of DirectXTex/Shaders/BC7Encode.hlsl (REF_DEVICE path: barriers on, no
// early-out). Four kernels run as a pipeline, exactly like BCDirectCompute.cpp:
//
//   TryMode456 -> TryMode137 (mode ids 1,3,7) -> TryMode02 (0,2) -> EncodeBlock
//
// using ping-pong error buffers. Each kernel runs 64-thread groups with a shared
// Bc7Shared[64] scratch array (the HLSL groupshared shared_temp).
//
// Differences from the WGSL original:
//   * Texel loads read the RGBA8 bytes directly and clamp coordinates to the image
//     edge (WGSL did u32(textureLoad * 255.0), which can truncate c/255*255 to c-1,
//     and relied on the implementation's out-of-bounds textureLoad behavior).
//   * Partition-table lookups that WGSL left to index clamping (only reachable for
//     modes that ignore the value) are guarded explicitly.
//   * EncodeBlock has one extra barrier per endpoint-reduction round, so the thread
//     reading the reduced endpoints can't race the next round's writes (the HLSL
//     relied on lockstep execution within a warp).
//   * The cbuffer is a per-launch kernel argument, and CUDA has no 65535-group cap,
//     so each pass is a single launch over all blocks.
//-------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Runtime;

namespace TexGen.Gpu;

/// <summary>The BC7Encode.hlsl cbuffer, passed by value to every launch.</summary>
public struct Bc7Params
{
    public int TexWidth, TexHeight;
    public uint NumBlockX;
    public uint Format;
    public uint ModeId;
    public uint StartBlockId;
    public uint NumTotalBlocks;
    public float AlphaWeight;
}

/// <summary>An element of the HLSL shared_temp array.</summary>
public struct Bc7Shared
{
    public UInt4 Pixel;
    public uint Error;
    public uint Mode;
    public uint Part;
    public uint IndexSelector;
    public uint Rotation;
    public UInt4 EndPointLow;
    public UInt4 EndPointHigh;
    public UInt4 EndPointLowQ;
    public UInt4 EndPointHighQ;
}

internal sealed class Bc7Encoder
{
    private const int ThreadGroupSize = 64;

    private readonly Accelerator _acc;
    private readonly Action<KernelConfig, Bc7Params, ArrayView<uint>, ArrayView<uint>, ArrayView<UInt4>, ArrayView<UInt4>> _tryMode456;
    private readonly Action<KernelConfig, Bc7Params, ArrayView<uint>, ArrayView<uint>, ArrayView<UInt4>, ArrayView<UInt4>> _tryMode137;
    private readonly Action<KernelConfig, Bc7Params, ArrayView<uint>, ArrayView<uint>, ArrayView<UInt4>, ArrayView<UInt4>> _tryMode02;
    private readonly Action<KernelConfig, Bc7Params, ArrayView<uint>, ArrayView<uint>, ArrayView<UInt4>, ArrayView<UInt4>> _encodeBlock;

    // Ping-pong error buffers, reused across calls and grown on demand.
    private MemoryBuffer1D<UInt4, Stride1D.Dense>? _err1, _err2;

    // All constant tables concatenated into one device buffer. Kernels index it via
    // the Off* offsets; this is much faster than ILGPU's inlined static arrays, which
    // get materialized per thread.
    private const int OffSectionBit = 0;
    private const int OffSectionBit2 = OffSectionBit + 64;
    private const int OffFixUp = OffSectionBit2 + 64;
    private const int OffFixUpOrdered = OffFixUp + 256;
    private const int OffAWeight = OffFixUpOrdered + 256;
    private const int OffAStep = OffAWeight + 48;
    private readonly MemoryBuffer1D<uint, Stride1D.Dense> _tables;

    public Bc7Encoder(Accelerator acc)
    {
        _acc = acc;
        _tables = acc.Allocate1D<uint>([.. SectionBit, .. SectionBit2, .. FixUp, .. FixUpOrdered, .. AWeight, .. AStep]);
        _tryMode456 = acc.LoadStreamKernel<Bc7Params, ArrayView<uint>, ArrayView<uint>, ArrayView<UInt4>, ArrayView<UInt4>>(TryMode456CS);
        _tryMode137 = acc.LoadStreamKernel<Bc7Params, ArrayView<uint>, ArrayView<uint>, ArrayView<UInt4>, ArrayView<UInt4>>(TryMode137CS);
        _tryMode02 = acc.LoadStreamKernel<Bc7Params, ArrayView<uint>, ArrayView<uint>, ArrayView<UInt4>, ArrayView<UInt4>>(TryMode02CS);
        _encodeBlock = acc.LoadStreamKernel<Bc7Params, ArrayView<uint>, ArrayView<uint>, ArrayView<UInt4>, ArrayView<UInt4>>(EncodeBlockCS);
    }

    /// <summary>Encode an RGBA8 texture to BC7; the result stays on the device until downloaded.</summary>
    public GpuBlocks Encode(GpuTexture src, DxgiFormat format, float alphaWeight, bool quick, bool use3Subsets)
    {
        if (format is not (DxgiFormat.BC7_UNORM or DxgiFormat.BC7_UNORM_SRGB))
            throw new NotSupportedException($"Bc7Encoder: unsupported format {format}");
        if (src.IsFloat) throw new ArgumentException("Bc7Encoder: source must be R8G8B8A8_UNORM(_SRGB)");

        bool mode137 = !quick;
        bool mode02 = !quick && use3Subsets;

        int xblocks = Math.Max(1, (src.Width + 3) >> 2);
        int yblocks = Math.Max(1, (src.Height + 3) >> 2);
        int numBlocks = xblocks * yblocks;
        int paddedBlocks = (numBlocks + 3) / 4 * 4; // 4 blocks per group in 456/encode

        EnsureErrorBuffers(paddedBlocks);
        ArrayView<UInt4> err1 = _err1!.View.SubView(0, paddedBlocks);
        ArrayView<UInt4> err2 = _err2!.View.SubView(0, paddedBlocks);

        var output = _acc.Allocate1D<uint>((long)paddedBlocks * 4);
        ArrayView<UInt4> outView = ((ArrayView<uint>)output.View).Cast<UInt4>();

        var p = new Bc7Params
        {
            TexWidth = src.Width,
            TexHeight = src.Height,
            NumBlockX = (uint)xblocks,
            Format = (uint)format,
            StartBlockId = 0,
            NumTotalBlocks = (uint)numBlocks,
            AlphaWeight = alphaWeight,
        };
        var input = src.Rgba8View;
        ArrayView<uint> tbl = _tables.View;
        var cfg4 = new KernelConfig(paddedBlocks / 4, ThreadGroupSize); // 456 / encode: 4 blocks per group
        var cfg1 = new KernelConfig(paddedBlocks, ThreadGroupSize); // 137 / 02: 1 block per group

        // Pass 1: modes 4/5/6 -> err1
        _tryMode456(cfg4, p with { ModeId = 0 }, input, tbl, err2, err1);

        // Pass 2: modes 1, 3, 7 (ping-pong err1 <-> err2)
        if (mode137)
        {
            _tryMode137(cfg1, p with { ModeId = 1 }, input, tbl, err1, err2);
            _tryMode137(cfg1, p with { ModeId = 3 }, input, tbl, err2, err1);
            _tryMode137(cfg1, p with { ModeId = 7 }, input, tbl, err1, err2);
        }

        // Pass 3: modes 0, 2 (3-subset)
        if (mode02)
        {
            _tryMode02(cfg1, p with { ModeId = 0 }, input, tbl, err2, err1);
            _tryMode02(cfg1, p with { ModeId = 2 }, input, tbl, err1, err2);
        }

        // Pass 4: emit the final blocks.
        _encodeBlock(cfg4, p with { ModeId = 0 }, input, tbl, mode137 || mode02 ? err2 : err1, outView);

        return new GpuBlocks(src.Width, src.Height, format, output);
    }

    private void EnsureErrorBuffers(int blocks)
    {
        if (_err1 is not null && _err1.Length >= blocks) return;
        // Launches queued on the old buffers must finish before they are freed.
        _acc.Synchronize();
        _err1?.Dispose();
        _err2?.Dispose();
        _err1 = _acc.Allocate1D<UInt4>(blocks);
        _err2 = _acc.Allocate1D<UInt4>(blocks);
    }

    //---------------------------------------------------------------------------------
    // Constant tables (flattened; generated from bc7.wgsl)
    //---------------------------------------------------------------------------------

    // candidateSectionBit (64 entries)
    private static readonly uint[] SectionBit =
    [
        0xccccu, 0x8888u, 0xeeeeu, 0xecc8u, 0xc880u, 0xfeecu, 0xfec8u, 0xec80u, 0xc800u, 0xffecu, 0xfe80u, 0xe800u, 0xffe8u, 0xff00u, 0xfff0u, 0xf000u,
        0xf710u, 0x8eu, 0x7100u, 0x8ceu, 0x8cu, 0x7310u, 0x3100u, 0x8cceu, 0x88cu, 0x3110u, 0x6666u, 0x366cu, 0x17e8u, 0xff0u, 0x718eu, 0x399cu,
        0xaaaau, 0xf0f0u, 0x5a5au, 0x33ccu, 0x3c3cu, 0x55aau, 0x9696u, 0xa55au, 0x73ceu, 0x13c8u, 0x324cu, 0x3bdcu, 0x6996u, 0xc33cu, 0x9966u, 0x660u,
        0x272u, 0x4e4u, 0x4e40u, 0x2720u, 0xc936u, 0x936cu, 0x39c6u, 0x639cu, 0x9336u, 0x9cc6u, 0x817eu, 0xe718u, 0xccf0u, 0xfccu, 0x7744u, 0xee22u,
    ];

    // candidateSectionBit2 (64 entries)
    private static readonly uint[] SectionBit2 =
    [
        0xaa685050u, 0x6a5a5040u, 0x5a5a4200u, 0x5450a0a8u, 0xa5a50000u, 0xa0a05050u, 0x5555a0a0u, 0x5a5a5050u, 0xaa550000u, 0xaa555500u, 0xaaaa5500u, 0x90909090u, 0x94949494u, 0xa4a4a4a4u, 0xa9a59450u, 0x2a0a4250u,
        0xa5945040u, 0xa425054u, 0xa5a5a500u, 0x55a0a0a0u, 0xa8a85454u, 0x6a6a4040u, 0xa4a45000u, 0x1a1a0500u, 0x50a4a4u, 0xaaa59090u, 0x14696914u, 0x69691400u, 0xa08585a0u, 0xaa821414u, 0x50a4a450u, 0x6a5a0200u,
        0xa9a58000u, 0x5090a0a8u, 0xa8a09050u, 0x24242424u, 0xaa5500u, 0x24924924u, 0x24499224u, 0x50a50a50u, 0x500aa550u, 0xaaaa4444u, 0x66660000u, 0xa5a0a5a0u, 0x50a050a0u, 0x69286928u, 0x44aaaa44u, 0x66666600u,
        0xaa444444u, 0x54a854a8u, 0x95809580u, 0x96969600u, 0xa85454a8u, 0x80959580u, 0xaa141414u, 0x96960000u, 0xaaaa1414u, 0xa05050a0u, 0xa0a5a5a0u, 0x96000000u, 0x40804080u, 0xa9a8a9a8u, 0xaaaaaa44u, 0x2a4a5254u,
    ];

    // candidateFixUpIndex1D (256 entries)
    private static readonly uint[] FixUp =
    [
        15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u,
        15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u,
        15u, 0u, 2u, 0u, 8u, 0u, 2u, 0u, 2u, 0u, 8u, 0u, 8u, 0u, 15u, 0u,
        2u, 0u, 8u, 0u, 2u, 0u, 2u, 0u, 8u, 0u, 8u, 0u, 2u, 0u, 2u, 0u,
        15u, 0u, 15u, 0u, 6u, 0u, 8u, 0u, 2u, 0u, 8u, 0u, 15u, 0u, 15u, 0u,
        2u, 0u, 8u, 0u, 2u, 0u, 2u, 0u, 2u, 0u, 15u, 0u, 15u, 0u, 6u, 0u,
        6u, 0u, 2u, 0u, 6u, 0u, 8u, 0u, 15u, 0u, 15u, 0u, 2u, 0u, 2u, 0u,
        15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 2u, 0u, 2u, 0u, 15u, 0u,
        3u, 15u, 3u, 8u, 15u, 8u, 15u, 3u, 8u, 15u, 3u, 15u, 15u, 3u, 15u, 8u,
        8u, 15u, 8u, 15u, 6u, 15u, 6u, 15u, 6u, 15u, 5u, 15u, 3u, 15u, 3u, 8u,
        3u, 15u, 3u, 8u, 8u, 15u, 15u, 3u, 3u, 15u, 3u, 8u, 6u, 15u, 10u, 8u,
        5u, 3u, 8u, 15u, 8u, 6u, 6u, 10u, 8u, 15u, 5u, 15u, 15u, 10u, 15u, 8u,
        8u, 15u, 15u, 3u, 3u, 15u, 5u, 10u, 6u, 10u, 10u, 8u, 8u, 9u, 15u, 10u,
        15u, 6u, 3u, 15u, 15u, 8u, 5u, 15u, 15u, 3u, 15u, 6u, 15u, 6u, 15u, 8u,
        3u, 15u, 15u, 3u, 5u, 15u, 5u, 15u, 5u, 15u, 8u, 15u, 5u, 15u, 10u, 15u,
        5u, 15u, 10u, 15u, 8u, 15u, 13u, 15u, 15u, 3u, 12u, 15u, 3u, 15u, 3u, 8u,
    ];

    // candidateFixUpIndex1DOrdered (256 entries)
    private static readonly uint[] FixUpOrdered =
    [
        15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u,
        15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u,
        15u, 0u, 2u, 0u, 8u, 0u, 2u, 0u, 2u, 0u, 8u, 0u, 8u, 0u, 15u, 0u,
        2u, 0u, 8u, 0u, 2u, 0u, 2u, 0u, 8u, 0u, 8u, 0u, 2u, 0u, 2u, 0u,
        15u, 0u, 15u, 0u, 6u, 0u, 8u, 0u, 2u, 0u, 8u, 0u, 15u, 0u, 15u, 0u,
        2u, 0u, 8u, 0u, 2u, 0u, 2u, 0u, 2u, 0u, 15u, 0u, 15u, 0u, 6u, 0u,
        6u, 0u, 2u, 0u, 6u, 0u, 8u, 0u, 15u, 0u, 15u, 0u, 2u, 0u, 2u, 0u,
        15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 15u, 0u, 2u, 0u, 2u, 0u, 15u, 0u,
        3u, 15u, 3u, 8u, 8u, 15u, 3u, 15u, 8u, 15u, 3u, 15u, 3u, 15u, 8u, 15u,
        8u, 15u, 8u, 15u, 6u, 15u, 6u, 15u, 6u, 15u, 5u, 15u, 3u, 15u, 3u, 8u,
        3u, 15u, 3u, 8u, 8u, 15u, 3u, 15u, 3u, 15u, 3u, 8u, 6u, 15u, 8u, 10u,
        3u, 5u, 8u, 15u, 6u, 8u, 6u, 10u, 8u, 15u, 5u, 15u, 10u, 15u, 8u, 15u,
        8u, 15u, 3u, 15u, 3u, 15u, 5u, 10u, 6u, 10u, 8u, 10u, 8u, 9u, 10u, 15u,
        6u, 15u, 3u, 15u, 8u, 15u, 5u, 15u, 3u, 15u, 6u, 15u, 6u, 15u, 8u, 15u,
        3u, 15u, 3u, 15u, 5u, 15u, 5u, 15u, 5u, 15u, 8u, 15u, 5u, 15u, 10u, 15u,
        5u, 15u, 10u, 15u, 8u, 15u, 13u, 15u, 3u, 15u, 12u, 15u, 3u, 15u, 3u, 8u,
    ];

    // aWeight (48 entries)
    private static readonly uint[] AWeight =
    [
        0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u, 51u, 55u, 60u, 64u,
        0u, 9u, 18u, 27u, 37u, 46u, 55u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 21u, 43u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    ];

    // aStep (192 entries)
    private static readonly uint[] AStep =
    [
        0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u,
        4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u,
        7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u,
        11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u,
        0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u,
        2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u,
        3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u,
        5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u,
        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
        1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u,
        2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u,
    ];
    //---------------------------------------------------------------------------------
    // Helpers
    //---------------------------------------------------------------------------------

    /// <summary>HLSL int4 stand-in (span vectors).</summary>
    private struct I4
    {
        public int X, Y, Z, W;

        public I4(int x, int y, int z, int w) { X = x; Y = y; Z = z; W = w; }

        public static I4 From(UInt4 v) => new((int)v.X, (int)v.Y, (int)v.Z, (int)v.W);
        public static I4 operator -(I4 a, I4 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z, a.W - b.W);
        public static I4 operator -(I4 a) => new(-a.X, -a.Y, -a.Z, -a.W);
    }

    private static int Dot3(I4 a, I4 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    private static int Dot4(I4 a, I4 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;

    /// <summary>|a - b| per component (HLSL Ensure_A_Is_Larger + subtract).</summary>
    private static UInt4 AbsDiff(UInt4 a, UInt4 b) => UInt4.Max(a, b) - UInt4.Min(a, b);

    private static uint ComputeError(UInt4 d, float alphaWeight) =>
        (uint)((float)(d.X * d.X + d.Y * d.Y + d.Z * d.Z) + alphaWeight * (float)(d.W * d.W));

    private static uint Quantize(uint c, int prec) => (((c << 8) + c) * ((1u << prec) - 1) + 32768u) >> 16;

    private static uint Unquantize(uint c, int prec)
    {
        uint x = c << (8 - prec);
        return x | (x >> prec);
    }

    /// <summary>Swap alpha with the channel selected by a mode 4/5 rotation.</summary>
    private static UInt4 Rotate(UInt4 v, uint rotation)
    {
        if (rotation == 1) return new UInt4(v.W, v.Y, v.Z, v.X);
        if (rotation == 2) return new UInt4(v.X, v.W, v.Z, v.Y);
        if (rotation == 3) return new UInt4(v.X, v.Y, v.W, v.Z);
        return v;
    }

    /// <summary>Reconstruct a pixel from interpolated endpoints, uint domain (all 4 channels).</summary>
    private static UInt4 Interp(UInt4 lo, UInt4 hi, uint w) =>
        new(((64u - w) * lo.X + w * hi.X + 32u) >> 6,
            ((64u - w) * lo.Y + w * hi.Y + 32u) >> 6,
            ((64u - w) * lo.Z + w * hi.Z + 32u) >> 6,
            ((64u - w) * lo.W + w * hi.W + 32u) >> 6);

    /// <summary>Quantized index for a projection <paramref name="d"/> onto a span of squared length <paramref name="sns"/>.</summary>
    private static uint StepIndex(ArrayView<uint> tbl, int sns, int d, uint prec)
    {
        if (sns <= 0 || d <= 0) return 0;
        if (d < sns) return tbl[(int)(OffAStep + (prec * 64 + (uint)((float)d * 63.49999f / (float)sns)))];
        return tbl[(int)(OffAStep + (prec * 64 + 63))];
    }

    /// <summary>The HLSL swap heuristic: true when the anchor pixel sits past the span midpoint.</summary>
    private static bool ShouldSwap(int sns, int dp) =>
        sns > 0 && dp > 0 && (uint)((float)dp * 63.49999f) > (uint)(32 * sns);

    private static UInt4 LoadPixel(ArrayView<uint> input, Bc7Params p, uint baseX, uint baseY, int t)
    {
        int x = Math.Min((int)baseX + (t & 3), p.TexWidth - 1);
        int y = Math.Min((int)baseY + (t >> 2), p.TexHeight - 1);
        return Texel.Unpack(input[y * p.TexWidth + x]);
    }

    /// <summary>
    /// compress_endpoints{mode} for a single endpoint: quantizes <paramref name="ep"/> in
    /// place (to its reconstructed value) and returns the quantized bit pattern.
    /// </summary>
    private static UInt4 CompressEndpoint(uint mode, ref UInt4 ep, uint pbit)
    {
        UInt4 q;
        switch (mode)
        {
            case 0:
                q = new UInt4((Quantize(ep.X, 5) & 0xFFFFFFFEu) | pbit, (Quantize(ep.Y, 5) & 0xFFFFFFFEu) | pbit,
                    (Quantize(ep.Z, 5) & 0xFFFFFFFEu) | pbit, 0xFFu);
                ep = new UInt4(Unquantize(q.X, 5), Unquantize(q.Y, 5), Unquantize(q.Z, 5), 0xFFu);
                return q << 3;
            case 1:
                q = new UInt4((Quantize(ep.X, 7) & 0xFFFFFFFEu) | pbit, (Quantize(ep.Y, 7) & 0xFFFFFFFEu) | pbit,
                    (Quantize(ep.Z, 7) & 0xFFFFFFFEu) | pbit, 0xFFu);
                ep = new UInt4(Unquantize(q.X, 7), Unquantize(q.Y, 7), Unquantize(q.Z, 7), 0xFFu);
                return q << 1;
            case 2:
                q = new UInt4(Quantize(ep.X, 5), Quantize(ep.Y, 5), Quantize(ep.Z, 5), 0xFFu);
                ep = new UInt4(Unquantize(q.X, 5), Unquantize(q.Y, 5), Unquantize(q.Z, 5), 0xFFu);
                return q << 3;
            case 3:
                q = new UInt4((ep.X & 0xFFFFFFFEu) | pbit, (ep.Y & 0xFFFFFFFEu) | pbit, (ep.Z & 0xFFFFFFFEu) | pbit, 0xFFu);
                ep = q;
                return q;
            case 4:
            {
                uint qa = Quantize(ep.W, 6);
                q = new UInt4(Quantize(ep.X, 5), Quantize(ep.Y, 5), Quantize(ep.Z, 5), qa);
                ep = new UInt4(Unquantize(q.X, 5), Unquantize(q.Y, 5), Unquantize(q.Z, 5), Unquantize(qa, 6));
                return new UInt4(q.X << 3, q.Y << 3, q.Z << 3, qa << 2);
            }
            case 5:
            {
                uint a = ep.W;
                q = new UInt4(Quantize(ep.X, 7), Quantize(ep.Y, 7), Quantize(ep.Z, 7), a);
                ep = new UInt4(Unquantize(q.X, 7), Unquantize(q.Y, 7), Unquantize(q.Z, 7), a);
                return new UInt4(q.X << 1, q.Y << 1, q.Z << 1, a);
            }
            case 6:
                q = new UInt4((ep.X & 0xFFFFFFFEu) | pbit, (ep.Y & 0xFFFFFFFEu) | pbit,
                    (ep.Z & 0xFFFFFFFEu) | pbit, (ep.W & 0xFFFFFFFEu) | pbit);
                ep = q;
                return q;
            default: // 7
                q = new UInt4((Quantize(ep.X, 6) & 0xFFFFFFFEu) | pbit, (Quantize(ep.Y, 6) & 0xFFFFFFFEu) | pbit,
                    (Quantize(ep.Z, 6) & 0xFFFFFFFEu) | pbit, (Quantize(ep.W, 6) & 0xFFFFFFFEu) | pbit);
                ep = new UInt4(Unquantize(q.X, 6), Unquantize(q.Y, 6), Unquantize(q.Z, 6), Unquantize(q.W, 6));
                return q << 2;
        }
    }

    /// <summary>Tree-reduce endpoint min/max over a block's 16 threads into slot threadBase.</summary>
    private static void ReduceEndpoints(ArrayView<Bc7Shared> sh, int gi, int tib)
    {
        for (int step = 8; step >= 1; step >>= 1)
        {
            if (tib < step)
            {
                sh[gi].EndPointLow = UInt4.Min(sh[gi].EndPointLow, sh[gi + step].EndPointLow);
                sh[gi].EndPointHigh = UInt4.Max(sh[gi].EndPointHigh, sh[gi + step].EndPointHigh);
            }
            Group.Barrier();
        }
    }

    //---------------------------------------------------------------------------------
    // Pass 1: modes 4, 5, 6 (1 subset) — 16 threads per block, 4 blocks per group
    //---------------------------------------------------------------------------------
    private static void TryMode456CS(Bc7Params p, ArrayView<uint> input, ArrayView<uint> tbl, ArrayView<UInt4> inBuf, ArrayView<UInt4> outBuf)
    {
        var sh = SharedMemory.Allocate<Bc7Shared>(ThreadGroupSize);
        const int MaxUsedThread = 16;
        int gi = Group.IdxX;
        int blockInGroup = gi / MaxUsedThread;
        uint blockID = p.StartBlockId + (uint)(Grid.IdxX * (ThreadGroupSize / MaxUsedThread) + blockInGroup);
        int threadBase = blockInGroup * MaxUsedThread;
        int tib = gi - threadBase;

        uint blockY = blockID / p.NumBlockX;
        uint blockX = blockID - blockY * p.NumBlockX;
        uint baseX = blockX * 4, baseY = blockY * 4;

        if (tib < 16)
        {
            var px = LoadPixel(input, p, baseX, baseY, tib);
            sh[gi].Pixel = px;
            sh[gi].EndPointLow = px;
            sh[gi].EndPointHigh = px;
        }
        Group.Barrier();
        ReduceEndpoints(sh, gi, tib);

        UInt4 ep0 = sh[threadBase].EndPointLow;
        UInt4 ep1 = sh[threadBase].EndPointHigh;

        uint error = 0xFFFFFFFFu;
        uint mode = 0, indexSelector = 0, rotation = 0;

        uint precX, precY;
        if (tib < 8)
        {
            if ((tib & 1) == 0) { indexSelector = 0; precX = 2; precY = 1; }
            else { indexSelector = 1; precX = 1; precY = 2; }
        }
        else
        {
            precX = 2;
            precY = 2;
        }

        if (tib < 12)
        {
            if (tib < 2 || tib == 8) rotation = 0;
            else if (tib < 4 || tib == 9) rotation = 1;
            else if (tib < 6 || tib == 10) rotation = 2;
            else rotation = 3; // tib < 8 || tib == 11
            ep0 = Rotate(ep0, rotation);
            ep1 = Rotate(ep1, rotation);

            mode = tib < 8 ? 4u : 5u;
            CompressEndpoint(mode, ref ep0, 0);
            CompressEndpoint(mode, ref ep1, 0);

            var pixel = Rotate(sh[threadBase].Pixel, rotation);
            var span = I4.From(ep1) - I4.From(ep0);
            int snsX = Dot3(span, span), snsY = span.W * span.W;

            var d0 = I4.From(pixel) - I4.From(ep0);
            var d1 = I4.From(pixel) - I4.From(ep1);
            if (Dot3(d0, d0) > Dot3(d1, d1))
            {
                span = new I4(-span.X, -span.Y, -span.Z, span.W);
                var t = ep0;
                ep0 = new UInt4(ep1.X, ep1.Y, ep1.Z, t.W);
                ep1 = new UInt4(t.X, t.Y, t.Z, ep1.W);
            }
            int aw0 = (int)pixel.W - (int)ep0.W, aw1 = (int)pixel.W - (int)ep1.W;
            if (aw0 * aw0 > aw1 * aw1)
            {
                span.W = -span.W;
                uint tw = ep0.W;
                ep0.W = ep1.W;
                ep1.W = tw;
            }

            error = 0;
            for (int i = 0; i < 16; i++)
            {
                pixel = Rotate(sh[threadBase + i].Pixel, rotation);
                uint ci = StepIndex(tbl, snsX, Dot3(span, I4.From(pixel) - I4.From(ep0)), precX);
                uint ai = StepIndex(tbl, snsY, span.W * ((int)pixel.W - (int)ep0.W), precY);
                uint wc = tbl[(int)(OffAWeight + (precX * 16 + ci))];
                uint wa = tbl[(int)(OffAWeight + (precY * 16 + ai))];
                var rgb = Interp(ep0, ep1, wc);
                var pixelR = new UInt4(rgb.X, rgb.Y, rgb.Z, ((64u - wa) * ep0.W + wa * ep1.W + 32u) >> 6);
                var diff = Rotate(AbsDiff(pixelR, pixel), rotation);
                error += ComputeError(diff, p.AlphaWeight);
            }
        }
        else
        {
            uint pp = (uint)tib - 12;
            CompressEndpoint(6, ref ep0, pp & 1);
            CompressEndpoint(6, ref ep1, (pp >> 1) & 1);

            var pixel = sh[threadBase].Pixel;
            var span = I4.From(ep1) - I4.From(ep0);
            int sns = Dot4(span, span);
            int dp0 = Dot4(span, I4.From(pixel) - I4.From(ep0));
            // Note: HLSL uses dp0 >= 0 here (vs > 0 elsewhere); preserved.
            if (sns > 0 && dp0 >= 0 && (uint)((float)dp0 * 63.49999f) > (uint)(32 * sns))
            {
                span = -span;
                (ep0, ep1) = (ep1, ep0);
            }

            error = 0;
            for (int i = 0; i < 16; i++)
            {
                pixel = sh[threadBase + i].Pixel;
                uint ci = StepIndex(tbl, sns, Dot4(span, I4.From(pixel) - I4.From(ep0)), 0);
                var pixelR = Interp(ep0, ep1, tbl[(int)(OffAWeight + (ci))]);
                error += ComputeError(AbsDiff(pixelR, pixel), p.AlphaWeight);
            }
            mode = 6;
            rotation = pp;
        }

        sh[gi].Error = error;
        sh[gi].Mode = mode;
        sh[gi].IndexSelector = indexSelector;
        sh[gi].Rotation = rotation;
        Group.Barrier();

        for (int step = 8; step >= 1; step >>= 1)
        {
            if (tib < step && sh[gi].Error > sh[gi + step].Error)
            {
                sh[gi].Error = sh[gi + step].Error;
                sh[gi].Mode = sh[gi + step].Mode;
                sh[gi].IndexSelector = sh[gi + step].IndexSelector;
                sh[gi].Rotation = sh[gi + step].Rotation;
            }
            Group.Barrier();
        }

        if (tib < 1)
            outBuf[(int)blockID] = new UInt4(sh[gi].Error, (sh[gi].IndexSelector << 31) | sh[gi].Mode, 0, sh[gi].Rotation);
    }

    //---------------------------------------------------------------------------------
    // Pass 2: modes 1, 3, 7 (2 subsets) — one thread per partition, 1 block per group
    //---------------------------------------------------------------------------------
    private static void TryMode137CS(Bc7Params p, ArrayView<uint> input, ArrayView<uint> tbl, ArrayView<UInt4> inBuf, ArrayView<UInt4> outBuf)
    {
        var sh = SharedMemory.Allocate<Bc7Shared>(ThreadGroupSize);
        int gi = Group.IdxX;
        uint blockID = p.StartBlockId + (uint)Grid.IdxX;
        const int threadBase = 0;
        int tib = gi;

        uint blockY = blockID / p.NumBlockX;
        uint blockX = blockID - blockY * p.NumBlockX;
        uint baseX = blockX * 4, baseY = blockY * 4;

        if (tib < 16) sh[gi].Pixel = LoadPixel(input, p, baseX, baseY, tib);
        Group.Barrier();

        uint modeId = p.ModeId;
        uint part = (uint)tib;
        uint bits = tbl[(int)(OffSectionBit + (part))];

        var b0lo = new UInt4(0xFFFFFFFFu);
        var b0hi = new UInt4(0u);
        var b1lo = new UInt4(0xFFFFFFFFu);
        var b1hi = new UInt4(0u);
        for (int i = 0; i < 16; i++)
        {
            var pixel = sh[threadBase + i].Pixel;
            if (((bits >> i) & 1u) == 1u)
            {
                b1lo = UInt4.Min(b1lo, pixel);
                b1hi = UInt4.Max(b1hi, pixel);
            }
            else
            {
                b0lo = UInt4.Min(b0lo, pixel);
                b0hi = UInt4.Max(b0hi, pixel);
            }
        }

        uint maxP = modeId == 1 ? 2u : 4u;
        uint stepSelector = modeId != 1 ? 2u : 1u;
        uint finalP0 = 0, finalP1 = 0;
        uint errs0 = 0xFFFFFFFFu, errs1 = 0xFFFFFFFFu;
        uint fix1 = tbl[(int)(OffFixUp + (part * 2))];

        for (uint pv = 0; pv < maxP; pv++)
        {
            var e0lo = b0lo;
            var e0hi = b0hi;
            var e1lo = b1lo;
            var e1hi = b1hi;

            // Mode 1 has one shared p-bit per subset; modes 3/7 one per endpoint.
            uint pLo = modeId == 1 ? pv : pv & 1;
            uint pHi = modeId == 1 ? pv : (pv >> 1) & 1;
            CompressEndpoint(modeId, ref e0lo, pLo);
            CompressEndpoint(modeId, ref e0hi, pHi);
            CompressEndpoint(modeId, ref e1lo, pLo);
            CompressEndpoint(modeId, ref e1hi, pHi);

            var span0 = I4.From(e0hi) - I4.From(e0lo);
            var span1 = I4.From(e1hi) - I4.From(e1lo);
            if (modeId != 7)
            {
                span0.W = 0;
                span1.W = 0;
            }
            int sns0 = Dot4(span0, span0);
            int sns1 = Dot4(span1, span1);

            if (ShouldSwap(sns0, Dot4(span0, I4.From(sh[threadBase].Pixel) - I4.From(e0lo))))
            {
                span0 = -span0;
                (e0lo, e0hi) = (e0hi, e0lo);
            }
            if (ShouldSwap(sns1, Dot4(span1, I4.From(sh[threadBase + (int)fix1].Pixel) - I4.From(e1lo))))
            {
                span1 = -span1;
                (e1lo, e1hi) = (e1hi, e1lo);
            }

            uint pErr0 = 0, pErr1 = 0;
            for (int i = 0; i < 16; i++)
            {
                var pixel = sh[threadBase + i].Pixel;
                uint subset = (bits >> i) & 1u;
                UInt4 pixelR;
                if (subset == 1)
                {
                    uint ci = StepIndex(tbl, sns1, Dot4(span1, I4.From(pixel) - I4.From(e1lo)), stepSelector);
                    pixelR = Interp(e1lo, e1hi, tbl[(int)(OffAWeight + (stepSelector * 16 + ci))]);
                }
                else
                {
                    uint ci = StepIndex(tbl, sns0, Dot4(span0, I4.From(pixel) - I4.From(e0lo)), stepSelector);
                    pixelR = Interp(e0lo, e0hi, tbl[(int)(OffAWeight + (stepSelector * 16 + ci))]);
                }
                if (modeId != 7) pixelR.W = 255;

                uint pe = ComputeError(AbsDiff(pixelR, pixel), p.AlphaWeight);
                if (subset == 1) pErr1 += pe;
                else pErr0 += pe;
            }

            if (pErr0 < errs0) { errs0 = pErr0; finalP0 = pv; }
            if (pErr1 < errs1) { errs1 = pErr1; finalP1 = pv; }
        }

        sh[gi].Error = errs0 + errs1;
        sh[gi].Mode = modeId;
        sh[gi].Part = part;
        sh[gi].Rotation = modeId == 1 ? (finalP1 << 1) | finalP0 : (finalP1 << 2) | finalP0;
        Group.Barrier();

        for (int step = 32; step >= 1; step >>= 1)
        {
            if (tib < step && sh[gi].Error > sh[gi + step].Error)
            {
                sh[gi].Error = sh[gi + step].Error;
                sh[gi].Mode = sh[gi + step].Mode;
                sh[gi].Part = sh[gi + step].Part;
                sh[gi].Rotation = sh[gi + step].Rotation;
            }
            Group.Barrier();
        }

        if (tib < 1)
        {
            var prev = inBuf[(int)blockID];
            outBuf[(int)blockID] = prev.X > sh[gi].Error
                ? new UInt4(sh[gi].Error, sh[gi].Mode, sh[gi].Part, sh[gi].Rotation)
                : prev;
        }
    }

    //---------------------------------------------------------------------------------
    // Pass 3: modes 0, 2 (3 subsets) — one thread per partition, 1 block per group
    //---------------------------------------------------------------------------------
    private static void TryMode02CS(Bc7Params p, ArrayView<uint> input, ArrayView<uint> tbl, ArrayView<UInt4> inBuf, ArrayView<UInt4> outBuf)
    {
        var sh = SharedMemory.Allocate<Bc7Shared>(ThreadGroupSize);
        int gi = Group.IdxX;
        uint blockID = p.StartBlockId + (uint)Grid.IdxX;
        const int threadBase = 0;
        int tib = gi;

        uint blockY = blockID / p.NumBlockX;
        uint blockX = blockID - blockY * p.NumBlockX;
        uint baseX = blockX * 4, baseY = blockY * 4;

        if (tib < 16) sh[gi].Pixel = LoadPixel(input, p, baseX, baseY, tib);
        Group.Barrier();

        sh[gi].Error = 0xFFFFFFFFu;

        uint modeId = p.ModeId;
        uint numPartitions = modeId == 0 ? 16u : 64u;

        if ((uint)tib < numPartitions)
        {
            uint part = (uint)tib + 64;
            uint bits2 = tbl[(int)(OffSectionBit2 + (part - 64))];

            var b0lo = new UInt4(0xFFFFFFFFu);
            var b0hi = new UInt4(0u);
            var b1lo = new UInt4(0xFFFFFFFFu);
            var b1hi = new UInt4(0u);
            var b2lo = new UInt4(0xFFFFFFFFu);
            var b2hi = new UInt4(0u);
            for (int i = 0; i < 16; i++)
            {
                var pixel = sh[threadBase + i].Pixel;
                uint subset = (bits2 >> (i * 2)) & 3u;
                if (subset == 2)
                {
                    b2lo = UInt4.Min(b2lo, pixel);
                    b2hi = UInt4.Max(b2hi, pixel);
                }
                else if (subset == 1)
                {
                    b1lo = UInt4.Min(b1lo, pixel);
                    b1hi = UInt4.Max(b1hi, pixel);
                }
                else
                {
                    b0lo = UInt4.Min(b0lo, pixel);
                    b0hi = UInt4.Max(b0hi, pixel);
                }
            }

            uint maxP = modeId == 0 ? 4u : 1u;
            uint stepSelector = modeId == 2 ? 2u : 1u;
            uint finalP0 = 0, finalP1 = 0, finalP2 = 0;
            uint errs0 = 0xFFFFFFFFu, errs1 = 0xFFFFFFFFu, errs2 = 0xFFFFFFFFu;
            int fix1 = (int)tbl[(int)(OffFixUp + (part * 2))];
            int fix2 = (int)tbl[(int)(OffFixUp + (part * 2 + 1))];

            for (uint pv = 0; pv < maxP; pv++)
            {
                var e0lo = b0lo;
                var e0hi = b0hi;
                var e1lo = b1lo;
                var e1hi = b1hi;
                var e2lo = b2lo;
                var e2hi = b2hi;

                uint pLo = pv & 1, pHi = (pv >> 1) & 1; // ignored by mode 2
                CompressEndpoint(modeId, ref e0lo, pLo);
                CompressEndpoint(modeId, ref e0hi, pHi);
                CompressEndpoint(modeId, ref e1lo, pLo);
                CompressEndpoint(modeId, ref e1hi, pHi);
                CompressEndpoint(modeId, ref e2lo, pLo);
                CompressEndpoint(modeId, ref e2hi, pHi);

                var span0 = I4.From(e0hi) - I4.From(e0lo);
                var span1 = I4.From(e1hi) - I4.From(e1lo);
                var span2 = I4.From(e2hi) - I4.From(e2lo);
                span0.W = 0;
                span1.W = 0;
                span2.W = 0;
                int sns0 = Dot4(span0, span0), sns1 = Dot4(span1, span1), sns2 = Dot4(span2, span2);

                if (ShouldSwap(sns0, Dot4(span0, I4.From(sh[threadBase].Pixel) - I4.From(e0lo))))
                {
                    span0 = -span0;
                    (e0lo, e0hi) = (e0hi, e0lo);
                }
                if (ShouldSwap(sns1, Dot4(span1, I4.From(sh[threadBase + fix1].Pixel) - I4.From(e1lo))))
                {
                    span1 = -span1;
                    (e1lo, e1hi) = (e1hi, e1lo);
                }
                if (ShouldSwap(sns2, Dot4(span2, I4.From(sh[threadBase + fix2].Pixel) - I4.From(e2lo))))
                {
                    span2 = -span2;
                    (e2lo, e2hi) = (e2hi, e2lo);
                }

                uint pErr0 = 0, pErr1 = 0, pErr2 = 0;
                for (int i = 0; i < 16; i++)
                {
                    var pixel = sh[threadBase + i].Pixel;
                    uint subset = (bits2 >> (i * 2)) & 3u;
                    I4 span = subset == 2 ? span2 : subset == 1 ? span1 : span0;
                    UInt4 lo = subset == 2 ? e2lo : subset == 1 ? e1lo : e0lo;
                    UInt4 hi = subset == 2 ? e2hi : subset == 1 ? e1hi : e0hi;
                    int sns = subset == 2 ? sns2 : subset == 1 ? sns1 : sns0;

                    uint ci = StepIndex(tbl, sns, Dot4(span, I4.From(pixel) - I4.From(lo)), stepSelector);
                    var pixelR = Interp(lo, hi, tbl[(int)(OffAWeight + (stepSelector * 16 + ci))]);
                    pixelR.W = 255;

                    uint pe = ComputeError(AbsDiff(pixelR, pixel), p.AlphaWeight);
                    if (subset == 2) pErr2 += pe;
                    else if (subset == 1) pErr1 += pe;
                    else pErr0 += pe;
                }

                if (pErr0 < errs0) { errs0 = pErr0; finalP0 = pv; }
                if (pErr1 < errs1) { errs1 = pErr1; finalP1 = pv; }
                if (pErr2 < errs2) { errs2 = pErr2; finalP2 = pv; }
            }

            sh[gi].Error = errs0 + errs1 + errs2;
            sh[gi].Part = part;
            sh[gi].Rotation = (finalP2 << 4) | (finalP1 << 2) | finalP0;
        }
        Group.Barrier();

        for (int step = 32; step >= 1; step >>= 1)
        {
            if (tib < step && sh[gi].Error > sh[gi + step].Error)
            {
                sh[gi].Error = sh[gi + step].Error;
                sh[gi].Part = sh[gi + step].Part;
                sh[gi].Rotation = sh[gi + step].Rotation;
            }
            Group.Barrier();
        }

        if (tib < 1)
        {
            var prev = inBuf[(int)blockID];
            outBuf[(int)blockID] = prev.X > sh[gi].Error
                ? new UInt4(sh[gi].Error, modeId, sh[gi].Part, sh[gi].Rotation)
                : prev;
        }
    }

    //---------------------------------------------------------------------------------
    // Pass 4: emit the final block for the winning mode — 16 threads per block
    //---------------------------------------------------------------------------------
    private static void EncodeBlockCS(Bc7Params p, ArrayView<uint> input, ArrayView<uint> tbl, ArrayView<UInt4> inBuf, ArrayView<UInt4> outBuf)
    {
        var sh = SharedMemory.Allocate<Bc7Shared>(ThreadGroupSize);
        const int MaxUsedThread = 16;
        int gi = Group.IdxX;
        int blockInGroup = gi / MaxUsedThread;
        uint blockID = p.StartBlockId + (uint)(Grid.IdxX * (ThreadGroupSize / MaxUsedThread) + blockInGroup);
        int threadBase = blockInGroup * MaxUsedThread;
        int tib = gi - threadBase;

        uint blockY = blockID / p.NumBlockX;
        uint blockX = blockID - blockY * p.NumBlockX;
        uint baseX = blockX * 4, baseY = blockY * 4;

        var best = inBuf[(int)blockID];
        uint mode = best.Y & 0x7FFFFFFFu;
        uint part = best.Z;
        uint indexSelector = (best.Y >> 31) & 1u;
        uint rotation = best.W;

        if (tib < 16)
        {
            var pixel = LoadPixel(input, p, baseX, baseY, tib);
            if (mode == 4 || mode == 5) pixel = Rotate(pixel, rotation);
            sh[gi].Pixel = pixel;
        }
        Group.Barrier();

        // Partition tables are only meaningful for the modes that use them; guard the
        // lookups the WGSL relied on index clamping for.
        uint bits = part < 64 ? tbl[(int)(OffSectionBit + (part))] : 0u;
        uint bits2 = part >= 64 && part < 128 ? tbl[(int)(OffSectionBit2 + (part - 64))] : 0u;

        var ep0 = new UInt4(0xFFFFFFFFu);
        var ep1 = new UInt4(0u);

        for (int ii = 2; ii >= 0; ii--)
        {
            if (tib < 16)
            {
                var lep0 = new UInt4(0xFFFFFFFFu);
                var lep1 = new UInt4(0u);
                var pixel = sh[gi].Pixel;
                uint subset = (bits >> tib) & 1u;
                uint subset2 = (bits2 >> (tib * 2)) & 3u;
                bool take;
                if (mode == 0 || mode == 2) take = subset2 == (uint)ii;
                else if (mode == 1 || mode == 3 || mode == 7) take = ii < 2 && subset == (uint)ii;
                else take = ii == 0;
                if (take)
                {
                    lep0 = pixel;
                    lep1 = pixel;
                }
                sh[gi].EndPointLow = lep0;
                sh[gi].EndPointHigh = lep1;
            }
            Group.Barrier();
            ReduceEndpoints(sh, gi, tib);

            if (ii == tib)
            {
                ep0 = sh[threadBase].EndPointLow;
                ep1 = sh[threadBase].EndPointHigh;
            }
            // Extra barrier (not in the HLSL): the next round overwrites slot threadBase.
            Group.Barrier();
        }

        if (tib < 3)
        {
            uint pLo, pHi;
            if (mode == 1)
            {
                pLo = (rotation >> tib) & 1u;
                pHi = pLo;
            }
            else
            {
                pLo = (rotation >> (tib * 2)) & 1u;
                pHi = (rotation >> (tib * 2 + 1)) & 1u;
            }

            var q0 = CompressEndpoint(mode, ref ep0, pLo);
            var q1 = CompressEndpoint(mode, ref ep1, pHi);

            var span = I4.From(ep1) - I4.From(ep0);
            if (mode < 4) span.W = 0;

            if (mode == 4 || mode == 5)
            {
                if (tib == 0)
                {
                    int snsX = Dot3(span, span), snsY = span.W * span.W;
                    var p0 = sh[threadBase].Pixel;
                    int dpX = Dot3(span, I4.From(p0) - I4.From(ep0));
                    int dpY = span.W * ((int)p0.W - (int)ep0.W);
                    if (ShouldSwap(snsX, dpX))
                    {
                        var t = ep0;
                        ep0 = new UInt4(ep1.X, ep1.Y, ep1.Z, t.W);
                        ep1 = new UInt4(t.X, t.Y, t.Z, ep1.W);
                        var tq = q0;
                        q0 = new UInt4(q1.X, q1.Y, q1.Z, tq.W);
                        q1 = new UInt4(tq.X, tq.Y, tq.Z, q1.W);
                    }
                    if (ShouldSwap(snsY, dpY))
                    {
                        uint tw = ep0.W;
                        ep0.W = ep1.W;
                        ep1.W = tw;
                        uint tqw = q0.W;
                        q0.W = q1.W;
                        q1.W = tqw;
                    }
                }
            }
            else
            {
                int pp = tib == 0 ? 0 : tib == 1 ? (int)tbl[(int)(OffFixUp + ((part & 127) * 2))] : (int)tbl[(int)(OffFixUp + ((part & 127) * 2 + 1))];
                int sns = Dot4(span, span);
                int dp = Dot4(span, I4.From(sh[threadBase + pp].Pixel) - I4.From(ep0));
                if (ShouldSwap(sns, dp))
                {
                    (ep0, ep1) = (ep1, ep0);
                    (q0, q1) = (q1, q0);
                }
            }

            sh[gi].EndPointLow = ep0;
            sh[gi].EndPointHigh = ep1;
            sh[gi].EndPointLowQ = q0;
            sh[gi].EndPointHighQ = q1;
        }
        Group.Barrier();

        if (tib < 16)
        {
            uint colorIndex = 0, alphaIndex = 0;

            uint precX, precY;
            if (mode == 0 || mode == 1) { precX = 1; precY = 1; }
            else if (mode == 6) { precX = 0; precY = 0; }
            else if (mode == 4)
            {
                if (indexSelector == 0) { precX = 2; precY = 1; }
                else { precX = 1; precY = 2; }
            }
            else { precX = 2; precY = 2; }

            int subsetIndex;
            if (mode == 0 || mode == 2) subsetIndex = (int)((bits2 >> (tib * 2)) & 3u);
            else if (mode == 1 || mode == 3 || mode == 7) subsetIndex = (int)((bits >> tib) & 1u);
            else subsetIndex = 0;

            var lo = sh[threadBase + subsetIndex].EndPointLow;
            var hi = sh[threadBase + subsetIndex].EndPointHigh;
            var span = I4.From(hi) - I4.From(lo);
            if (mode < 4) span.W = 0;

            var px = sh[threadBase + tib].Pixel;
            if (mode == 4 || mode == 5)
            {
                int snsX = Dot3(span, span), snsY = span.W * span.W;
                colorIndex = StepIndex(tbl, snsX, Dot3(span, I4.From(px) - I4.From(lo)), precX);
                alphaIndex = StepIndex(tbl, snsY, span.W * ((int)px.W - (int)lo.W), precY);
                if (indexSelector != 0) (colorIndex, alphaIndex) = (alphaIndex, colorIndex);
            }
            else
            {
                colorIndex = StepIndex(tbl, Dot4(span, span), Dot4(span, I4.From(px) - I4.From(lo)), precX);
            }

            sh[gi].Error = colorIndex;
            sh[gi].Mode = alphaIndex;
        }
        Group.Barrier();

        if (tib == 0)
        {
            UInt4 block;
            if (mode == 0) block = BlockPackage0(sh, tbl, part, threadBase);
            else if (mode == 1) block = BlockPackage1(sh, tbl, part, threadBase);
            else if (mode == 2) block = BlockPackage2(sh, tbl, part, threadBase);
            else if (mode == 3) block = BlockPackage3(sh, tbl, part, threadBase);
            else if (mode == 4) block = BlockPackage4(sh, tbl, rotation, indexSelector, threadBase);
            else if (mode == 5) block = BlockPackage5(sh, tbl, rotation, threadBase);
            else if (mode == 6) block = BlockPackage6(sh, tbl, threadBase);
            else block = BlockPackage7(sh, tbl, part, threadBase);
            outBuf[(int)blockID] = block;
        }
    }

    //---------------------------------------------------------------------------------
    // block_package* (bit packing of the final block, verbatim from the HLSL/WGSL)
    //---------------------------------------------------------------------------------

    private static UInt4 BlockPackage0(ArrayView<Bc7Shared> sh, ArrayView<uint> tbl, uint part, int tb)
    {
        uint bx = 0x01u | ((part - 64u) << 1)
          | ((sh[tb].EndPointLowQ.X & 0xF0u) << 1) | ((sh[tb].EndPointHighQ.X & 0xF0u) << 5)
          | ((sh[tb + 1].EndPointLowQ.X & 0xF0u) << 9) | ((sh[tb + 1].EndPointHighQ.X & 0xF0u) << 13)
          | ((sh[tb + 2].EndPointLowQ.X & 0xF0u) << 17) | ((sh[tb + 2].EndPointHighQ.X & 0xF0u) << 21)
          | ((sh[tb].EndPointLowQ.Y & 0xF0u) << 25);
        uint by = ((sh[tb].EndPointLowQ.Y & 0xF0u) >> 7) | ((sh[tb].EndPointHighQ.Y & 0xF0u) >> 3)
          | ((sh[tb + 1].EndPointLowQ.Y & 0xF0u) << 1) | ((sh[tb + 1].EndPointHighQ.Y & 0xF0u) << 5)
          | ((sh[tb + 2].EndPointLowQ.Y & 0xF0u) << 9) | ((sh[tb + 2].EndPointHighQ.Y & 0xF0u) << 13)
          | ((sh[tb].EndPointLowQ.Z & 0xF0u) << 17) | ((sh[tb].EndPointHighQ.Z & 0xF0u) << 21)
          | ((sh[tb + 1].EndPointLowQ.Z & 0xF0u) << 25);
        uint bz = ((sh[tb + 1].EndPointLowQ.Z & 0xF0u) >> 7) | ((sh[tb + 1].EndPointHighQ.Z & 0xF0u) >> 3)
          | ((sh[tb + 2].EndPointLowQ.Z & 0xF0u) << 1) | ((sh[tb + 2].EndPointHighQ.Z & 0xF0u) << 5)
          | ((sh[tb].EndPointLowQ.X & 0x08u) << 10) | ((sh[tb].EndPointHighQ.X & 0x08u) << 11)
          | ((sh[tb + 1].EndPointLowQ.X & 0x08u) << 12) | ((sh[tb + 1].EndPointHighQ.X & 0x08u) << 13)
          | ((sh[tb + 2].EndPointLowQ.X & 0x08u) << 14) | ((sh[tb + 2].EndPointHighQ.X & 0x08u) << 15)
          | (sh[tb].Error << 19);
        uint bw = 0u;
        uint f0 = tbl[(int)(OffFixUpOrdered + (part * 2))];
        uint f1 = tbl[(int)(OffFixUpOrdered + (part * 2 + 1))];
        uint i = 1u;
        for (; i <= Math.Min(f0, 4u); i++) {
          bz |= sh[tb + (int)i].Error << (int)(i * 3u + 18u);
        }
        if (f0 < 4u) {
          bz |= sh[tb + 4].Error << 29;
          i += 1u;
        } else {
          bw |= (sh[tb + 4].Error & 0x04u) >> 2;
          for (; i <= f0; i++) {
            bw |= sh[tb + (int)i].Error << (int)(i * 3u - 14u);
          }
        }
        for (; i <= f1; i++) {
          bw |= sh[tb + (int)i].Error << (int)(i * 3u - 15u);
        }
        for (; i < 16u; i++) {
          bw |= sh[tb + (int)i].Error << (int)(i * 3u - 16u);
        }
        return new UInt4(bx, by, bz, bw);
    }

    private static UInt4 BlockPackage1(ArrayView<Bc7Shared> sh, ArrayView<uint> tbl, uint part, int tb)
    {
        uint bx = 0x02u | (part << 2)
          | ((sh[tb].EndPointLowQ.X & 0xFCu) << 6) | ((sh[tb].EndPointHighQ.X & 0xFCu) << 12)
          | ((sh[tb + 1].EndPointLowQ.X & 0xFCu) << 18) | ((sh[tb + 1].EndPointHighQ.X & 0xFCu) << 24);
        uint by = ((sh[tb].EndPointLowQ.Y & 0xFCu) >> 2) | ((sh[tb].EndPointHighQ.Y & 0xFCu) << 4)
          | ((sh[tb + 1].EndPointLowQ.Y & 0xFCu) << 10) | ((sh[tb + 1].EndPointHighQ.Y & 0xFCu) << 16)
          | ((sh[tb].EndPointLowQ.Z & 0xFCu) << 22) | ((sh[tb].EndPointHighQ.Z & 0xFCu) << 28);
        uint bz = ((sh[tb].EndPointHighQ.Z & 0xFCu) >> 4) | ((sh[tb + 1].EndPointLowQ.Z & 0xFCu) << 2)
          | ((sh[tb + 1].EndPointHighQ.Z & 0xFCu) << 8)
          | ((sh[tb].EndPointLowQ.X & 0x02u) << 15) | ((sh[tb + 1].EndPointLowQ.X & 0x02u) << 16)
          | (sh[tb].Error << 18);
        uint bw = 0u;
        uint f0 = tbl[(int)(OffFixUpOrdered + (part * 2))];
        if (f0 == 15u) {
          bw = (sh[tb + 15].Error << 30) | (sh[tb + 14].Error << 27) | (sh[tb + 13].Error << 24) | (sh[tb + 12].Error << 21) | (sh[tb + 11].Error << 18) | (sh[tb + 10].Error << 15)
            | (sh[tb + 9].Error << 12) | (sh[tb + 8].Error << 9) | (sh[tb + 7].Error << 6) | (sh[tb + 6].Error << 3) | sh[tb + 5].Error;
          bz |= (sh[tb + 4].Error << 29) | (sh[tb + 3].Error << 26) | (sh[tb + 2].Error << 23) | (sh[tb + 1].Error << 20) | (sh[tb].Error << 18);
        } else if (f0 == 2u) {
          bw = (sh[tb + 15].Error << 29) | (sh[tb + 14].Error << 26) | (sh[tb + 13].Error << 23) | (sh[tb + 12].Error << 20) | (sh[tb + 11].Error << 17) | (sh[tb + 10].Error << 14)
            | (sh[tb + 9].Error << 11) | (sh[tb + 8].Error << 8) | (sh[tb + 7].Error << 5) | (sh[tb + 6].Error << 2) | (sh[tb + 5].Error >> 1);
          bz |= (sh[tb + 5].Error << 31) | (sh[tb + 4].Error << 28) | (sh[tb + 3].Error << 25) | (sh[tb + 2].Error << 23) | (sh[tb + 1].Error << 20) | (sh[tb].Error << 18);
        } else if (f0 == 8u) {
          bw = (sh[tb + 15].Error << 29) | (sh[tb + 14].Error << 26) | (sh[tb + 13].Error << 23) | (sh[tb + 12].Error << 20) | (sh[tb + 11].Error << 17) | (sh[tb + 10].Error << 14)
            | (sh[tb + 9].Error << 11) | (sh[tb + 8].Error << 9) | (sh[tb + 7].Error << 6) | (sh[tb + 6].Error << 3) | sh[tb + 5].Error;
          bz |= (sh[tb + 4].Error << 29) | (sh[tb + 3].Error << 26) | (sh[tb + 2].Error << 23) | (sh[tb + 1].Error << 20) | (sh[tb].Error << 18);
        } else {
          bw = (sh[tb + 15].Error << 29) | (sh[tb + 14].Error << 26) | (sh[tb + 13].Error << 23) | (sh[tb + 12].Error << 20) | (sh[tb + 11].Error << 17) | (sh[tb + 10].Error << 14)
            | (sh[tb + 9].Error << 11) | (sh[tb + 8].Error << 8) | (sh[tb + 7].Error << 5) | (sh[tb + 6].Error << 3) | sh[tb + 5].Error;
          bz |= (sh[tb + 4].Error << 29) | (sh[tb + 3].Error << 26) | (sh[tb + 2].Error << 23) | (sh[tb + 1].Error << 20) | (sh[tb].Error << 18);
        }
        return new UInt4(bx, by, bz, bw);
    }

    private static UInt4 BlockPackage2(ArrayView<Bc7Shared> sh, ArrayView<uint> tbl, uint part, int tb)
    {
        uint bx = 0x04u | ((part - 64u) << 3)
          | ((sh[tb].EndPointLowQ.X & 0xF8u) << 6) | ((sh[tb].EndPointHighQ.X & 0xF8u) << 11)
          | ((sh[tb + 1].EndPointLowQ.X & 0xF8u) << 16) | ((sh[tb + 1].EndPointHighQ.X & 0xF8u) << 21)
          | ((sh[tb + 2].EndPointLowQ.X & 0xF8u) << 26);
        uint by = ((sh[tb + 2].EndPointLowQ.X & 0xF8u) >> 6) | ((sh[tb + 2].EndPointHighQ.X & 0xF8u) >> 1)
          | ((sh[tb].EndPointLowQ.Y & 0xF8u) << 4) | ((sh[tb].EndPointHighQ.Y & 0xF8u) << 9)
          | ((sh[tb + 1].EndPointLowQ.Y & 0xF8u) << 14) | ((sh[tb + 1].EndPointHighQ.Y & 0xF8u) << 19)
          | ((sh[tb + 2].EndPointLowQ.Y & 0xF8u) << 24);
        uint bz = ((sh[tb + 2].EndPointHighQ.Y & 0xF8u) >> 3) | ((sh[tb].EndPointLowQ.Z & 0xF8u) << 2)
          | ((sh[tb].EndPointHighQ.Z & 0xF8u) << 7) | ((sh[tb + 1].EndPointLowQ.Z & 0xF8u) << 12)
          | ((sh[tb + 1].EndPointHighQ.Z & 0xF8u) << 17) | ((sh[tb + 2].EndPointLowQ.Z & 0xF8u) << 22)
          | ((sh[tb + 2].EndPointHighQ.Z & 0xF8u) << 27);
        uint bw = ((sh[tb + 2].EndPointHighQ.Z & 0xF8u) >> 5) | (sh[tb].Error << 3);
        uint f0 = tbl[(int)(OffFixUpOrdered + (part * 2))];
        uint f1 = tbl[(int)(OffFixUpOrdered + (part * 2 + 1))];
        uint i = 1u;
        for (; i <= f0; i++) { bw |= sh[tb + (int)i].Error << (int)(i * 2u + 2u); }
        for (; i <= f1; i++) { bw |= sh[tb + (int)i].Error << (int)(i * 2u + 1u); }
        for (; i < 16u; i++) { bw |= sh[tb + (int)i].Error << (int)(i * 2u); }
        return new UInt4(bx, by, bz, bw);
    }

    private static UInt4 BlockPackage3(ArrayView<Bc7Shared> sh, ArrayView<uint> tbl, uint part, int tb)
    {
        uint bx = 0x08u | (part << 4)
          | ((sh[tb].EndPointLowQ.X & 0xFEu) << 9) | ((sh[tb].EndPointHighQ.X & 0xFEu) << 16)
          | ((sh[tb + 1].EndPointLowQ.X & 0xFEu) << 23) | ((sh[tb + 1].EndPointHighQ.X & 0xFEu) << 30);
        uint by = ((sh[tb + 1].EndPointHighQ.X & 0xFEu) >> 2) | ((sh[tb].EndPointLowQ.Y & 0xFEu) << 5)
          | ((sh[tb].EndPointHighQ.Y & 0xFEu) << 12) | ((sh[tb + 1].EndPointLowQ.Y & 0xFEu) << 19)
          | ((sh[tb + 1].EndPointHighQ.Y & 0xFEu) << 26);
        uint bz = ((sh[tb + 1].EndPointHighQ.Y & 0xFEu) >> 6) | ((sh[tb].EndPointLowQ.Z & 0xFEu) << 1)
          | ((sh[tb].EndPointHighQ.Z & 0xFEu) << 8) | ((sh[tb + 1].EndPointLowQ.Z & 0xFEu) << 15)
          | ((sh[tb + 1].EndPointHighQ.Z & 0xFEu) << 22)
          | ((sh[tb].EndPointLowQ.X & 0x01u) << 30) | ((sh[tb].EndPointHighQ.X & 0x01u) << 31);
        uint bw = ((sh[tb + 1].EndPointLowQ.X & 0x01u) << 0) | ((sh[tb + 1].EndPointHighQ.X & 0x01u) << 1)
          | (sh[tb].Error << 2);
        uint f0 = tbl[(int)(OffFixUpOrdered + (part * 2))];
        uint i = 1u;
        for (; i <= f0; i++) { bw |= sh[tb + (int)i].Error << (int)(i * 2u + 1u); }
        for (; i < 16u; i++) { bw |= sh[tb + (int)i].Error << (int)(i * 2u); }
        return new UInt4(bx, by, bz, bw);
    }

    private static UInt4 BlockPackage4(ArrayView<Bc7Shared> sh, ArrayView<uint> tbl, uint rotation, uint index_selector, int tb)
    {
        uint bx = 0x10u | ((rotation & 3u) << 5) | ((index_selector & 1u) << 7)
          | ((sh[tb].EndPointLowQ.X & 0xF8u) << 5) | ((sh[tb].EndPointHighQ.X & 0xF8u) << 10)
          | ((sh[tb].EndPointLowQ.Y & 0xF8u) << 15) | ((sh[tb].EndPointHighQ.Y & 0xF8u) << 20)
          | ((sh[tb].EndPointLowQ.Z & 0xF8u) << 25);
        uint by = ((sh[tb].EndPointLowQ.Z & 0xF8u) >> 7) | ((sh[tb].EndPointHighQ.Z & 0xF8u) >> 2)
          | ((sh[tb].EndPointLowQ.W & 0xFCu) << 4) | ((sh[tb].EndPointHighQ.W & 0xFCu) << 10)
          | ((sh[tb].Error & 1u) << 18) | (sh[tb + 1].Error << 19) | (sh[tb + 2].Error << 21) | (sh[tb + 3].Error << 23)
          | (sh[tb + 4].Error << 25) | (sh[tb + 5].Error << 27) | (sh[tb + 6].Error << 29) | (sh[tb + 7].Error << 31);
        uint bz = (sh[tb + 7].Error >> 1) | (sh[tb + 8].Error << 1) | (sh[tb + 9].Error << 3) | (sh[tb + 10].Error << 5)
          | (sh[tb + 11].Error << 7) | (sh[tb + 12].Error << 9) | (sh[tb + 13].Error << 11) | (sh[tb + 14].Error << 13)
          | (sh[tb + 15].Error << 15) | ((sh[tb].Mode & 3u) << 17) | (sh[tb + 1].Mode << 19) | (sh[tb + 2].Mode << 22)
          | (sh[tb + 3].Mode << 25) | (sh[tb + 4].Mode << 28) | (sh[tb + 5].Mode << 31);
        uint bw = (sh[tb + 5].Mode >> 1) | (sh[tb + 6].Mode << 2) | (sh[tb + 7].Mode << 5) | (sh[tb + 8].Mode << 8)
          | (sh[tb + 9].Mode << 11) | (sh[tb + 10].Mode << 14) | (sh[tb + 11].Mode << 17) | (sh[tb + 12].Mode << 20)
          | (sh[tb + 13].Mode << 23) | (sh[tb + 14].Mode << 26) | (sh[tb + 15].Mode << 29);
        return new UInt4(bx, by, bz, bw);
    }

    private static UInt4 BlockPackage5(ArrayView<Bc7Shared> sh, ArrayView<uint> tbl, uint rotation, int tb)
    {
        uint bx = 0x20u | (rotation << 6)
          | ((sh[tb].EndPointLowQ.X & 0xFEu) << 7) | ((sh[tb].EndPointHighQ.X & 0xFEu) << 14)
          | ((sh[tb].EndPointLowQ.Y & 0xFEu) << 21) | ((sh[tb].EndPointHighQ.Y & 0xFEu) << 28);
        uint by = ((sh[tb].EndPointHighQ.Y & 0xFEu) >> 4) | ((sh[tb].EndPointLowQ.Z & 0xFEu) << 3)
          | ((sh[tb].EndPointHighQ.Z & 0xFEu) << 10) | (sh[tb].EndPointLowQ.W << 18) | (sh[tb].EndPointHighQ.W << 26);
        uint bz = (sh[tb].EndPointHighQ.W >> 6)
          | (sh[tb].Error << 2) | (sh[tb + 1].Error << 3) | (sh[tb + 2].Error << 5) | (sh[tb + 3].Error << 7)
          | (sh[tb + 4].Error << 9) | (sh[tb + 5].Error << 11) | (sh[tb + 6].Error << 13) | (sh[tb + 7].Error << 15)
          | (sh[tb + 8].Error << 17) | (sh[tb + 9].Error << 19) | (sh[tb + 10].Error << 21) | (sh[tb + 11].Error << 23)
          | (sh[tb + 12].Error << 25) | (sh[tb + 13].Error << 27) | (sh[tb + 14].Error << 29) | (sh[tb + 15].Error << 31);
        uint bw = (sh[tb + 15].Error >> 1) | (sh[tb].Mode << 1) | (sh[tb + 1].Mode << 2) | (sh[tb + 2].Mode << 4)
          | (sh[tb + 3].Mode << 6) | (sh[tb + 4].Mode << 8) | (sh[tb + 5].Mode << 10) | (sh[tb + 6].Mode << 12)
          | (sh[tb + 7].Mode << 14) | (sh[tb + 8].Mode << 16) | (sh[tb + 9].Mode << 18) | (sh[tb + 10].Mode << 20)
          | (sh[tb + 11].Mode << 22) | (sh[tb + 12].Mode << 24) | (sh[tb + 13].Mode << 26) | (sh[tb + 14].Mode << 28)
          | (sh[tb + 15].Mode << 30);
        return new UInt4(bx, by, bz, bw);
    }

    private static UInt4 BlockPackage6(ArrayView<Bc7Shared> sh, ArrayView<uint> tbl, int tb)
    {
        uint bx = 0x40u
          | ((sh[tb].EndPointLowQ.X & 0xFEu) << 6) | ((sh[tb].EndPointHighQ.X & 0xFEu) << 13)
          | ((sh[tb].EndPointLowQ.Y & 0xFEu) << 20) | ((sh[tb].EndPointHighQ.Y & 0xFEu) << 27);
        uint by = ((sh[tb].EndPointHighQ.Y & 0xFEu) >> 5) | ((sh[tb].EndPointLowQ.Z & 0xFEu) << 2)
          | ((sh[tb].EndPointHighQ.Z & 0xFEu) << 9) | ((sh[tb].EndPointLowQ.W & 0xFEu) << 16)
          | ((sh[tb].EndPointHighQ.W & 0xFEu) << 23)
          | ((sh[tb].EndPointLowQ.X & 0x01u) << 31);
        uint bz = (sh[tb].EndPointHighQ.X & 0x01u)
          | (sh[tb].Error << 1) | (sh[tb + 1].Error << 4) | (sh[tb + 2].Error << 8) | (sh[tb + 3].Error << 12)
          | (sh[tb + 4].Error << 16) | (sh[tb + 5].Error << 20) | (sh[tb + 6].Error << 24) | (sh[tb + 7].Error << 28);
        uint bw = (sh[tb + 8].Error << 0) | (sh[tb + 9].Error << 4) | (sh[tb + 10].Error << 8) | (sh[tb + 11].Error << 12)
          | (sh[tb + 12].Error << 16) | (sh[tb + 13].Error << 20) | (sh[tb + 14].Error << 24) | (sh[tb + 15].Error << 28);
        return new UInt4(bx, by, bz, bw);
    }

    private static UInt4 BlockPackage7(ArrayView<Bc7Shared> sh, ArrayView<uint> tbl, uint part, int tb)
    {
        uint bx = 0x80u | (part << 8)
          | ((sh[tb].EndPointLowQ.X & 0xF8u) << 11) | ((sh[tb].EndPointHighQ.X & 0xF8u) << 16)
          | ((sh[tb + 1].EndPointLowQ.X & 0xF8u) << 21) | ((sh[tb + 1].EndPointHighQ.X & 0xF8u) << 26);
        uint by = ((sh[tb + 1].EndPointHighQ.X & 0xF8u) >> 6) | ((sh[tb].EndPointLowQ.Y & 0xF8u) >> 1)
          | ((sh[tb].EndPointHighQ.Y & 0xF8u) << 4) | ((sh[tb + 1].EndPointLowQ.Y & 0xF8u) << 9)
          | ((sh[tb + 1].EndPointHighQ.Y & 0xF8u) << 14) | ((sh[tb].EndPointLowQ.Z & 0xF8u) << 19)
          | ((sh[tb].EndPointHighQ.Z & 0xF8u) << 24);
        uint bz = ((sh[tb + 1].EndPointLowQ.Z & 0xF8u) >> 3) | ((sh[tb + 1].EndPointHighQ.Z & 0xF8u) << 2)
          | ((sh[tb].EndPointLowQ.W & 0xF8u) << 7) | ((sh[tb].EndPointHighQ.W & 0xF8u) << 12)
          | ((sh[tb + 1].EndPointLowQ.W & 0xF8u) << 17) | ((sh[tb + 1].EndPointHighQ.W & 0xF8u) << 22)
          | ((sh[tb].EndPointLowQ.X & 0x04u) << 28) | ((sh[tb].EndPointHighQ.X & 0x04u) << 29);
        uint bw = ((sh[tb + 1].EndPointLowQ.X & 0x04u) >> 2) | ((sh[tb + 1].EndPointHighQ.X & 0x04u) >> 1)
          | (sh[tb].Error << 2);
        uint f0 = tbl[(int)(OffFixUpOrdered + (part * 2))];
        uint i = 1u;
        for (; i <= f0; i++) { bw |= sh[tb + (int)i].Error << (int)(i * 2u + 1u); }
        for (; i < 16u; i++) { bw |= sh[tb + (int)i].Error << (int)(i * 2u); }
        return new UInt4(bx, by, bz, bw);
    }
}
