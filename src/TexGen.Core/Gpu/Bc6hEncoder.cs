//-------------------------------------------------------------------------------------
// Gpu/Bc6hEncoder.cs
//
// BC6H GPU encoder — ILGPU port of texconv-js bc6h.wgsl, itself a faithful port of
// DirectXTex/Shaders/BC6HEncode.hlsl (the REF_DEVICE path: barriers on, no early-out),
// orchestrated like BCDirectCompute.cpp / compressBC6.ts:
//
//   TryModeG10     probe single-region modes 11..14            (4 blocks / group) -> err1
//   TryModeLE10    two-region modes 1..10 via mode id 0..9      (2 blocks / group), ping-pong
//                  err1 -> err2 -> err1 ... -> err1
//   EncodeBlock    re-derive endpoints for the winner, pack bits (2 blocks / group) -> output
//
// Translation notes (beyond the WGSL's own notes):
//   * workgroup shared_temp -> SharedMemory.Allocate<SharedData>(64); workgroupBarrier ->
//     Group.Barrier(). The cbuffer is a kernel struct parameter per launch.
//   * HLSL mat2x3 / int2x3 values are carried as pairs of Int3 (Ep struct).
//   * f32tof16 / f16tof32 are portable integer bit conversions (identical on CUDA and the
//     CPU accelerator). f32tof16 rounds toward zero, which is the D3D f32tof16 behaviour
//     this shader was written for (and why start_quantize special-cases 0x7BFF: finite
//     overflow truncates to the max finite half rather than rounding up to infinity).
//     The WebGPU port used pack2x16float, whose rounding is implementation-defined.
//   * block_package_one / block_package_two (~1100 lines of per-bit statements) are
//     expressed as a 14x82 table generated mechanically from bc6h.wgsl: entry =
//     (field << 8) | sourceBit for each output bit position; field 0..11 = endpoint
//     [region*6 + (0=low,1=high)*3 + channel], 12 = mode bits, 13 = partition bits.
//     The packed output is bit-identical to the unrolled statements.
//   * Texel loads clamp to the image edge (Tint's textureLoad clamps out-of-bounds
//     coordinates), so partial edge blocks replicate edge pixels.
//-------------------------------------------------------------------------------------

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace TexGen.Gpu;

public struct Bc6hParams
{
    public uint TexWidth;
    public uint TexHeight;
    public uint NumBlockX;
    public uint Format;
    public uint ModeId;
    public uint StartBlockId;
    public uint NumTotalBlocks;
}

internal sealed class Bc6hEncoder
{
    private const int ThreadGroupSize = 64;
    private const uint SignedF16 = 96;   // DXGI_FORMAT_BC6H_SF16
    private const uint UnsignedF16 = 95; // DXGI_FORMAT_BC6H_UF16

    private readonly Accelerator _acc;
    private readonly Action<KernelConfig, Bc6hParams, ArrayView<Float4>, ArrayView<UInt4>, ArrayView<UInt4>, ArrayView<uint>> _tryModeG10;
    private readonly Action<KernelConfig, Bc6hParams, ArrayView<Float4>, ArrayView<UInt4>, ArrayView<UInt4>, ArrayView<uint>> _tryModeLE10;
    private readonly Action<KernelConfig, Bc6hParams, ArrayView<Float4>, ArrayView<UInt4>, ArrayView<UInt4>, ArrayView<uint>> _encodeBlock;

    // Constant tables live in one device buffer. (ILGPU's inlined static arrays are
    // materialized into per-thread local memory at every use, which made the passes
    // ~100x slower; a read-only global buffer is served from the L1/texture cache.)
    private readonly MemoryBuffer1D<uint, Stride1D.Dense> _tableBuffer;

    public Bc6hEncoder(Accelerator acc)
    {
        _acc = acc;
        _tableBuffer = acc.Allocate1D(Tables.Build());
        _tryModeG10 = acc.LoadStreamKernel<Bc6hParams, ArrayView<Float4>, ArrayView<UInt4>, ArrayView<UInt4>, ArrayView<uint>>(TryModeG10Kernel);
        _tryModeLE10 = acc.LoadStreamKernel<Bc6hParams, ArrayView<Float4>, ArrayView<UInt4>, ArrayView<UInt4>, ArrayView<uint>>(TryModeLE10Kernel);
        _encodeBlock = acc.LoadStreamKernel<Bc6hParams, ArrayView<Float4>, ArrayView<UInt4>, ArrayView<UInt4>, ArrayView<uint>>(EncodeBlockKernel);
    }

    /// <summary>
    /// Encode an R32G32B32A32_FLOAT texture to BC6H_UF16 / BC6H_SF16. Like upstream
    /// (BC6HEncode.hlsl does max(pixel, 0) for both formats), negative inputs clamp to
    /// zero even for SF16 — texconv -gpu behaves the same way.
    /// </summary>
    public GpuBlocks Encode(GpuTexture src, DxgiFormat format)
    {
        if (!src.IsFloat) throw new ArgumentException("Bc6hEncoder: source must be R32G32B32A32_FLOAT");
        if (format is not (DxgiFormat.BC6H_UF16 or DxgiFormat.BC6H_SF16))
            throw new ArgumentException("Bc6hEncoder: target must be BC6H_UF16 or BC6H_SF16");

        int xblocks = Math.Max(1, (src.Width + 3) >> 2);
        int yblocks = Math.Max(1, (src.Height + 3) >> 2);
        int numBlocks = xblocks * yblocks;
        // 4 blocks per group in the G10 pass: pad so edge groups write into padding.
        int paddedBlocks = (numBlocks + 3) / 4 * 4;

        var output = _acc.Allocate1D<uint>((long)paddedBlocks * 4);
        EnsureScratch(paddedBlocks);
        ArrayView<UInt4> e1 = ((ArrayView<UInt4>)_err1!.View).SubView(0, paddedBlocks);
        ArrayView<UInt4> e2 = ((ArrayView<UInt4>)_err2!.View).SubView(0, paddedBlocks);
        var outView = ((ArrayView<uint>)output.View).Cast<UInt4>();
        var input = src.Rgba32FView;
        ArrayView<uint> tb = _tableBuffer.View;

        var p = new Bc6hParams
        {
            TexWidth = (uint)src.Width,
            TexHeight = (uint)src.Height,
            NumBlockX = (uint)xblocks,
            Format = (uint)format,
            ModeId = 0,
            StartBlockId = 0,
            NumTotalBlocks = (uint)numBlocks,
        };
        var groups4 = new KernelConfig(paddedBlocks / 4, ThreadGroupSize);
        var groups2 = new KernelConfig(paddedBlocks / 2, ThreadGroupSize);

        // Pass 1: probe single-region modes 11..14 -> err1.
        _tryModeG10(groups4, p, input, e2, e1, tb);

        // Pass 2: two-region modes 1..10 (mode id 0..9), ping-pong err1 <-> err2.
        for (uint i = 0; i < 10; ++i)
        {
            p.ModeId = i;
            if ((i & 1) != 0) _tryModeLE10(groups2, p, input, e2, e1, tb);
            else _tryModeLE10(groups2, p, input, e1, e2, tb);
        }

        // Pass 3: emit final blocks (the best result is in err1).
        p.ModeId = 0;
        _encodeBlock(groups2, p, input, e1, outView, tb);

        // No sync: launches are ordered on the accelerator's default stream, so the
        // cached err buffers can be reused by the next Encode (e.g. the next mip level).
        return new GpuBlocks(src.Width, src.Height, format, output);
    }

    // Ping-pong error buffers, kept across calls (grown on demand) to avoid a
    // device allocation + implicit synchronization per mip level.
    private MemoryBuffer1D<UInt4, Stride1D.Dense>? _err1, _err2;

    private void EnsureScratch(int blocks)
    {
        if (_err1 is not null && _err1.Length >= blocks) return;
        _err1?.Dispose();
        _err2?.Dispose();
        _err1 = _acc.Allocate1D<UInt4>(blocks);
        _err2 = _acc.Allocate1D<UInt4>(blocks);
    }

    //---------------------------------------------------------------------------------
    // Shared state and small value types
    //---------------------------------------------------------------------------------

    internal struct SharedData
    {
        public Float3 Pixel;
        public Int3 PixelPh;
        public Float3 PixelHr;
        public float PixelLum;
        public float Error;
        public uint BestMode;
        public uint BestPartition;
        public Int3 EndPointLow;
        public Int3 EndPointHigh;
        public float EndPointLumLow;
        public float EndPointLumHigh;
        public uint ScratchX; // index-bit accumulator in EncodeBlock (was pixel_hr.xy)
        public uint ScratchY;
    }

    /// <summary>An endpoint pair (HLSL int2x3 / WGSL mat2x3i).</summary>
    internal struct Ep
    {
        public Int3 Lo, Hi;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Ep(Int3 lo, Int3 hi) { Lo = lo; Hi = hi; }
    }

    //---------------------------------------------------------------------------------
    // Constant tables (host data; uploaded to the device by Tables.Build)
    //---------------------------------------------------------------------------------

    private static readonly uint[] CandidateModeMemory =
        { 0x00, 0x01, 0x02, 0x06, 0x0A, 0x0E, 0x12, 0x16, 0x1A, 0x1E, 0x03, 0x07, 0x0B, 0x0F };

    // candidateModeTransformed as 0/1.
    private static readonly uint[] CandidateModeTransformed = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1 };

    // candidateModePrec (x, y, z, w) flattened.
    private static readonly uint[] CandidateModePrec =
    {
        10, 5, 5, 5, 7, 6, 6, 6,
        11, 5, 4, 4, 11, 4, 5, 4, 11, 4, 4, 5, 9, 5, 5, 5,
        8, 6, 5, 5, 8, 5, 6, 5, 8, 5, 5, 6, 6, 6, 6, 6,
        10, 10, 10, 10, 11, 9, 9, 9, 12, 8, 8, 8, 16, 4, 4, 4,
    };

    private static readonly uint[] CandidateSectionBit =
    {
        0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80,
        0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000,
        0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
        0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
    };

    private static readonly uint[] CandidateFixUpIndex1D =
    {
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
    };

    // 0, 9, 18, 27, 37, 46, 55, 64
    private static readonly uint[] AStep1 =
    {
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
    };

    // 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64
    private static readonly uint[] AStep2 =
    {
        0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7,
        7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11,
        11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15,
    };

    private static readonly int[] AWeight3 = { 0, 9, 18, 27, 37, 46, 55, 64 };
    private static readonly int[] AWeight4 = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

    private const float LumR = 0.2126f, LumG = 0.7152f, LumB = 0.0722f;

    // Header bit layout per mode (see the file comment). Generated from bc6h.wgsl
    // block_package_two (mode index 0..9, 82 bits) / block_package_one (10..13, 65 bits).
    private const int PackStride = 82;
    private static readonly ushort[] PackTable =
    {
        // mode index 0
        0x0C00, 0x0C01, 0x0704, 0x0804, 0x0B04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0108, 0x0109, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0208, 0x0209, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x0A04, 0x0700, 0x0701, 0x0702, 0x0703, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x0B00, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x0B01, 0x0800, 0x0801, 0x0802, 0x0803, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604,
        0x0B02, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0B03, 0x0D00, 0x0D01, 0x0D02,
        0x0D03, 0x0D04,
        // mode index 1
        0x0C00, 0x0C01, 0x0705, 0x0A04, 0x0A05, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0B00, 0x0B01, 0x0804, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0805, 0x0B02, 0x0704, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0B03, 0x0B05, 0x0B04, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x0305, 0x0700, 0x0701, 0x0702, 0x0703, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x0405, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x0505, 0x0800, 0x0801, 0x0802, 0x0803, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604,
        0x0605, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0905, 0x0D00, 0x0D01, 0x0D02,
        0x0D03, 0x0D04,
        // mode index 2
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0108, 0x0109, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0208, 0x0209, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x000A, 0x0700, 0x0701, 0x0702, 0x0703, 0x0400, 0x0401, 0x0402, 0x0403, 0x010A,
        0x0B00, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0500, 0x0501, 0x0502, 0x0503, 0x020A,
        0x0B01, 0x0800, 0x0801, 0x0802, 0x0803, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604,
        0x0B02, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0B03, 0x0D00, 0x0D01, 0x0D02,
        0x0D03, 0x0D04,
        // mode index 3
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0108, 0x0109, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0208, 0x0209, 0x0300, 0x0301, 0x0302, 0x0303, 0x000A,
        0x0A04, 0x0700, 0x0701, 0x0702, 0x0703, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x010A, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0500, 0x0501, 0x0502, 0x0503, 0x020A,
        0x0B01, 0x0800, 0x0801, 0x0802, 0x0803, 0x0600, 0x0601, 0x0602, 0x0603, 0x0B00,
        0x0B02, 0x0900, 0x0901, 0x0902, 0x0903, 0x0704, 0x0B03, 0x0D00, 0x0D01, 0x0D02,
        0x0D03, 0x0D04,
        // mode index 4
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0108, 0x0109, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0208, 0x0209, 0x0300, 0x0301, 0x0302, 0x0303, 0x000A,
        0x0804, 0x0700, 0x0701, 0x0702, 0x0703, 0x0400, 0x0401, 0x0402, 0x0403, 0x010A,
        0x0B00, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x020A, 0x0800, 0x0801, 0x0802, 0x0803, 0x0600, 0x0601, 0x0602, 0x0603, 0x0B01,
        0x0B02, 0x0900, 0x0901, 0x0902, 0x0903, 0x0B04, 0x0B03, 0x0D00, 0x0D01, 0x0D02,
        0x0D03, 0x0D04,
        // mode index 5
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0008, 0x0804, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0108, 0x0704, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0208, 0x0B04, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x0A04, 0x0700, 0x0701, 0x0702, 0x0703, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x0B00, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x0B01, 0x0800, 0x0801, 0x0802, 0x0803, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604,
        0x0B02, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0B03, 0x0D00, 0x0D01, 0x0D02,
        0x0D03, 0x0D04,
        // mode index 6
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0A04, 0x0804, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0B02, 0x0704, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0B03, 0x0B04, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x0305, 0x0700, 0x0701, 0x0702, 0x0703, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x0B00, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x0B01, 0x0800, 0x0801, 0x0802, 0x0803, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604,
        0x0605, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0905, 0x0D00, 0x0D01, 0x0D02,
        0x0D03, 0x0D04,
        // mode index 7
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0B00, 0x0804, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0705, 0x0704, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0A05, 0x0B04, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x0A04, 0x0700, 0x0701, 0x0702, 0x0703, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x0405, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x0B01, 0x0800, 0x0801, 0x0802, 0x0803, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604,
        0x0B02, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0B03, 0x0D00, 0x0D01, 0x0D02,
        0x0D03, 0x0D04,
        // mode index 8
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0B01, 0x0804, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0805, 0x0704, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0B05, 0x0B04, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x0A04, 0x0700, 0x0701, 0x0702, 0x0703, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x0B00, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x0505, 0x0800, 0x0801, 0x0802, 0x0803, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604,
        0x0B02, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0B03, 0x0D00, 0x0D01, 0x0D02,
        0x0D03, 0x0D04,
        // mode index 9
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0A04, 0x0B00, 0x0B01, 0x0804, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0705, 0x0805, 0x0B02, 0x0704, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0A05, 0x0B03, 0x0B05, 0x0B04, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x0305, 0x0700, 0x0701, 0x0702, 0x0703, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x0405, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x0505, 0x0800, 0x0801, 0x0802, 0x0803, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604,
        0x0605, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0905, 0x0D00, 0x0D01, 0x0D02,
        0x0D03, 0x0D04,
        // mode index 10
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0108, 0x0109, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0208, 0x0209, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x0305, 0x0306, 0x0307, 0x0308, 0x0309, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x0405, 0x0406, 0x0407, 0x0408, 0x0409, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x0505, 0x0506, 0x0507, 0x0508, 0x0509, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF,
        // mode index 11
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0108, 0x0109, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0208, 0x0209, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x0305, 0x0306, 0x0307, 0x0308, 0x000A, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x0405, 0x0406, 0x0407, 0x0408, 0x010A, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x0505, 0x0506, 0x0507, 0x0508, 0x020A, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF,
        // mode index 12
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0108, 0x0109, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0208, 0x0209, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
        0x0305, 0x0306, 0x0307, 0x000B, 0x000A, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
        0x0405, 0x0406, 0x0407, 0x010B, 0x010A, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
        0x0505, 0x0506, 0x0507, 0x020B, 0x020A, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF,
        // mode index 13
        0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
        0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
        0x0105, 0x0106, 0x0107, 0x0108, 0x0109, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
        0x0205, 0x0206, 0x0207, 0x0208, 0x0209, 0x0300, 0x0301, 0x0302, 0x0303, 0x000F,
        0x000E, 0x000D, 0x000C, 0x000B, 0x000A, 0x0400, 0x0401, 0x0402, 0x0403, 0x010F,
        0x010E, 0x010D, 0x010C, 0x010B, 0x010A, 0x0500, 0x0501, 0x0502, 0x0503, 0x020F,
        0x020E, 0x020D, 0x020C, 0x020B, 0x020A, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF,
    };

    /// <summary>
    /// View over the constant-table buffer (kernels take the raw ArrayView&lt;uint&gt; —
    /// kernel signatures must use public types for the CPU accelerator — and wrap it). Layout: fixed offsets per table (all uint).
    /// </summary>
    internal struct Tables
    {
        public ArrayView<uint> Data;

        private const int OffModeMemory = 0;                 // 14
        private const int OffTransformed = OffModeMemory + 14; // 14
        private const int OffPrec = OffTransformed + 14;      // 56 (x, y, z, w per mode)
        private const int OffSectionBit = OffPrec + 56;       // 32
        private const int OffFixUp = OffSectionBit + 32;      // 32
        private const int OffStep1 = OffFixUp + 32;           // 64
        private const int OffStep2 = OffStep1 + 64;           // 64
        private const int OffWeight3 = OffStep2 + 64;         // 8
        private const int OffWeight4 = OffWeight3 + 8;        // 16
        private const int OffPack = OffWeight4 + 16;          // 14 * 82
        private const int Length = OffPack + 14 * PackStride;

        public static uint[] Build()
        {
            var t = new uint[Length];
            CandidateModeMemory.CopyTo(t, OffModeMemory);
            CandidateModeTransformed.CopyTo(t, OffTransformed);
            CandidateModePrec.CopyTo(t, OffPrec);
            CandidateSectionBit.CopyTo(t, OffSectionBit);
            CandidateFixUpIndex1D.CopyTo(t, OffFixUp);
            AStep1.CopyTo(t, OffStep1);
            AStep2.CopyTo(t, OffStep2);
            for (int i = 0; i < 8; i++) t[OffWeight3 + i] = (uint)AWeight3[i];
            for (int i = 0; i < 16; i++) t[OffWeight4 + i] = (uint)AWeight4[i];
            for (int i = 0; i < PackTable.Length; i++) t[OffPack + i] = PackTable[i];
            return t;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly uint ModeMemory(long i) => Data[OffModeMemory + i];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly uint Transformed(long i) => Data[OffTransformed + i];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly uint Prec(long i) => Data[OffPrec + i];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly uint SectionBit(long i) => Data[OffSectionBit + i];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly uint FixUp(long i) => Data[OffFixUp + i];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly uint Step1(long i) => Data[OffStep1 + i];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly uint Step2(long i) => Data[OffStep2 + i];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly int Weight3(long i) => (int)Data[OffWeight3 + i];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly int Weight4(long i) => (int)Data[OffWeight4 + i];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly uint Pack(long i) => Data[OffPack + i];
    }

    //---------------------------------------------------------------------------------
    // Half-float helpers (portable, bit-exact)
    //---------------------------------------------------------------------------------

    /// <summary>HLSL f32tof16: round toward zero; finite overflow -> max finite half.</summary>
    internal static uint F32ToF16(float f)
    {
        uint x = Interop.FloatAsInt(f);
        uint sign = (x >> 16) & 0x8000u;
        uint abs = x & 0x7FFFFFFFu;
        if (abs >= 0x7F800000u) return sign | (abs > 0x7F800000u ? 0x7FFFu : 0x7C00u);
        int e = (int)(abs >> 23) - 127;
        if (e > 15) return sign | 0x7BFFu;
        if (e >= -14) return sign | ((uint)(e + 15) << 10) | ((abs & 0x7FFFFFu) >> 13);
        int shift = -1 - e;
        if (shift > 24) return sign;
        uint mant = (abs & 0x7FFFFFu) | 0x800000u;
        return sign | (mant >> shift);
    }

    /// <summary>HLSL f16tof32 of the low 16 bits.</summary>
    internal static float F16ToF32(uint x)
    {
        uint h = x & 0xFFFFu;
        uint sign = (h & 0x8000u) << 16;
        uint e = (h >> 10) & 0x1Fu;
        uint m = h & 0x3FFu;
        if (e == 0)
        {
            float v = m * 5.9604644775390625e-8f; // m * 2^-24, exact
            return sign != 0 ? -v : v;
        }
        if (e == 31) return Interop.IntAsFloat(sign | 0x7F800000u | (m << 13));
        return Interop.IntAsFloat(sign | ((e + 112) << 23) | (m << 13));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Float3 ToF(Int3 v) => new(v.X, v.Y, v.Z);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Lum(Float3 c) => c.X * LumR + c.Y * LumG + c.Z * LumB;

    //---------------------------------------------------------------------------------
    // Quantization
    //---------------------------------------------------------------------------------

    private static int StartQuantize(uint h, uint format)
    {
        if (format == UnsignedF16) return (int)((h << 6) / 31u);
        if (h < 0x8000u) return h == 0x7BFFu ? 0x7FFF : (int)((h << 5) / 31u);
        return -(int)(((h & 0x7FFFu) << 5) / 31u);
    }

    private static Int3 StartQuantize(Float3 px, uint format) => new(
        StartQuantize(F32ToF16(px.X), format),
        StartQuantize(F32ToF16(px.Y), format),
        StartQuantize(F32ToF16(px.Z), format));

    private static Float3 HalfRound(Float3 px) =>
        new(F16ToF32(F32ToF16(px.X)), F16ToF32(F32ToF16(px.Y)), F16ToF32(F32ToF16(px.Z)));

    private static int Quantize(int ep, uint prec, uint format)
    {
        if (format == UnsignedF16)
        {
            if (prec >= 15) return ep;
            if (ep == 0) return ep;
            return ep == 0xFFFF ? (1 << (int)prec) - 1 : (ep << (int)prec) >> 16;
        }
        if (prec >= 16) return ep;
        if (ep == 0) return ep;
        int h = (1 << (int)(prec - 1)) - 1;
        if (ep >= 0) return ep == 0x7FFF ? h : (ep << (int)(prec - 1)) >> 15;
        return -ep == 0x7FFF ? -h : -(((-ep) << (int)(prec - 1)) >> 15);
    }

    private static Int3 Quantize(Int3 ep, uint prec, uint format) =>
        new(Quantize(ep.X, prec, format), Quantize(ep.Y, prec, format), Quantize(ep.Z, prec, format));

    private static Ep Quantize(Ep ep, uint prec, uint format) =>
        new(Quantize(ep.Lo, prec, format), Quantize(ep.Hi, prec, format));

    /// <summary>1 &lt;&lt; (prec - 1) for a delta channel.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Half(uint prec) => 1 << (int)(prec - 1);

    /// <summary>Is a transformed delta out of range for its precision?</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool BadComponent(int e, int h) => e >= 0 ? e >= h : -e > h;

    /// <summary>Clamp a transformed delta into range (finish_quantize's select chain).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ClampDelta(int e, int h, uint prec) =>
        e >= 0 ? (e >= h ? h - 1 : e) : (-e > h ? h : e & ((1 << (int)prec) - 1));

    private static Int3 ClampDelta(Int3 e, uint py, uint pz, uint pw) =>
        new(ClampDelta(e.X, Half(py), py), ClampDelta(e.Y, Half(pz), pz), ClampDelta(e.Z, Half(pw), pw));

    private static bool BadDelta(Int3 e, uint py, uint pz, uint pw) =>
        BadComponent(e.X, Half(py)) | BadComponent(e.Y, Half(pz)) | BadComponent(e.Z, Half(pw));

    /// <summary>HLSL finish_quantize_0 / finish_quantize: returns 1 in <paramref name="bad"/> if unrepresentable.</summary>
    private static Ep FinishQuantize0(Tables tb, Ep ep, uint mode, out int bad)
    {
        uint px = tb.Prec(mode * 4), py = tb.Prec(mode * 4 + 1);
        uint pz = tb.Prec(mode * 4 + 2), pw = tb.Prec(mode * 4 + 3);
        int m = (1 << (int)px) - 1;
        if (tb.Transformed(mode) != 0)
        {
            bad = BadDelta(ep.Hi, py, pz, pw) ? 1 : 0;
            return new Ep(ep.Lo & m, ClampDelta(ep.Hi, py, pz, pw));
        }
        bad = 0;
        return new Ep(ep.Lo & m, ep.Hi & m);
    }

    /// <summary>HLSL finish_quantize_1 (second region: both endpoints are deltas).</summary>
    private static Ep FinishQuantize1(Tables tb, Ep ep, uint mode, out int bad)
    {
        uint px = tb.Prec(mode * 4), py = tb.Prec(mode * 4 + 1);
        uint pz = tb.Prec(mode * 4 + 2), pw = tb.Prec(mode * 4 + 3);
        if (tb.Transformed(mode) != 0)
        {
            bad = BadDelta(ep.Lo, py, pz, pw) | BadDelta(ep.Hi, py, pz, pw) ? 1 : 0;
            return new Ep(ClampDelta(ep.Lo, py, pz, pw), ClampDelta(ep.Hi, py, pz, pw));
        }
        bad = 0;
        int m = (1 << (int)px) - 1;
        return new Ep(ep.Lo & m, ep.Hi & m);
    }

    //---------------------------------------------------------------------------------
    // Unquantization
    //---------------------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SignExtend(uint prec, int c)
    {
        int p = 1 << (int)(prec - 1);
        return (c & p) != 0 ? (c & (p - 1)) - p : c;
    }

    private static Int3 SignExtend(uint px, uint py, uint pz, Int3 c) =>
        new(SignExtend(px, c.X), SignExtend(py, c.Y), SignExtend(pz, c.Z));

    private static Ep StartUnquantizeOne(Tables tb, Ep ep, uint mode, uint format)
    {
        uint px = tb.Prec(mode * 4), py = tb.Prec(mode * 4 + 1);
        uint pz = tb.Prec(mode * 4 + 2), pw = tb.Prec(mode * 4 + 3);
        bool transformed = tb.Transformed(mode) != 0;
        if (format == SignedF16) ep.Lo = SignExtend(px, px, px, ep.Lo);
        if (format == SignedF16 || transformed) ep.Hi = SignExtend(py, pz, pw, ep.Hi);
        if (transformed) ep.Hi = ep.Hi + ep.Lo;
        return ep;
    }

    private static int Unquantize(int c, uint prec, uint format)
    {
        if (format == UnsignedF16)
        {
            if (prec < 15 && c != 0)
                return c == (1 << (int)prec) - 1 ? 0xFFFF : ((c << 16) + 0x8000) >> (int)prec;
            return c;
        }
        if (prec < 16)
        {
            bool s = c < 0;
            int a = s ? -c : c;
            if (a != 0)
                a = a >= (1 << (int)(prec - 1)) - 1 ? 0x7FFF : ((a << 15) + 0x4000) >> (int)(prec - 1);
            c = s ? -a : a;
        }
        return c;
    }

    private static Int3 Unquantize(Int3 c, uint prec, uint format) =>
        new(Unquantize(c.X, prec, format), Unquantize(c.Y, prec, format), Unquantize(c.Z, prec, format));

    private static Ep Unquantize(Ep ep, uint prec, uint format) =>
        new(Unquantize(ep.Lo, prec, format), Unquantize(ep.Hi, prec, format));

    private static uint FinishUnquantize(int color, uint format)
    {
        if (format == UnsignedF16) return (uint)((color * 31) >> 6);
        color = color < 0 ? -((-color * 31) >> 5) : (color * 31) >> 5;
        return (uint)(color < 0 ? (-color) | 0x8000 : color);
    }

    /// <summary>generate_palette_unquantized8/16 followed by half2float.</summary>
    private static Float3 PaletteColor(Int3 low, Int3 high, int w, uint format)
    {
        int iw = 64 - w;
        return new Float3(
            F16ToF32(FinishUnquantize((low.X * iw + high.X * w + 32) >> 6, format)),
            F16ToF32(FinishUnquantize((low.Y * iw + high.Y * w + 32) >> 6, format)),
            F16ToF32(FinishUnquantize((low.Z * iw + high.Z * w + 32) >> 6, format)));
    }

    //---------------------------------------------------------------------------------
    // Shared helpers
    //---------------------------------------------------------------------------------

    /// <summary>Edge-clamped texel load with the encoder's max(pixel, 0) input clamp.</summary>
    private static Float3 LoadPixel(Bc6hParams p, ArrayView<Float4> input, uint blockID, uint threadInBlock)
    {
        uint blockY = blockID / p.NumBlockX;
        uint blockX = blockID - blockY * p.NumBlockX;
        uint x = Math.Min(blockX * 4 + threadInBlock % 4, p.TexWidth - 1);
        uint y = Math.Min(blockY * 4 + threadInBlock / 4, p.TexHeight - 1);
        var t = input[(long)y * p.TexWidth + x];
        return new Float3(MathF.Max(t.X, 0f), MathF.Max(t.Y, 0f), MathF.Max(t.Z, 0f));
    }

    /// <summary>One step of the min/max-luminance endpoint reduction.</summary>
    private static void ReduceEndpoints(ArrayView<SharedData> sh, int gi, int stride)
    {
        if (sh[gi].EndPointLumLow > sh[gi + stride].EndPointLumLow)
        {
            sh[gi].EndPointLow = sh[gi + stride].EndPointLow;
            sh[gi].EndPointLumLow = sh[gi + stride].EndPointLumLow;
        }
        if (sh[gi].EndPointLumHigh < sh[gi + stride].EndPointLumHigh)
        {
            sh[gi].EndPointHigh = sh[gi + stride].EndPointHigh;
            sh[gi].EndPointLumHigh = sh[gi + stride].EndPointLumHigh;
        }
    }

    /// <summary>One step of the min-error reduction.</summary>
    private static void ReduceError(ArrayView<SharedData> sh, int gi, int stride)
    {
        if (sh[gi].Error > sh[gi + stride].Error)
        {
            sh[gi].Error = sh[gi + stride].Error;
            sh[gi].BestMode = sh[gi + stride].BestMode;
            sh[gi].BestPartition = sh[gi + stride].BestPartition;
        }
    }

    /// <summary>
    /// compute_index's endpoint-order fixup: swap so the fix-up pixel's index has a 0 MSB.
    /// </summary>
    private static Ep OrderEndpoints(Ep ep, Int3 fixupPixel)
    {
        var span = ToF(ep.Hi - ep.Lo);
        float spanNormSqr = Float3.Dot(span, span);
        float dotProduct = Float3.Dot(span, ToF(fixupPixel - ep.Lo));
        if (spanNormSqr > 0f && dotProduct >= 0f && (uint)(dotProduct * 63.49999f / spanNormSqr) > 32u)
            return new Ep(ep.Hi, ep.Lo);
        return ep;
    }

    /// <summary>Palette index of a pixel along an endpoint span (aStep1 = 3-bit, aStep2 = 4-bit).</summary>
    private static uint ProjectIndex(Tables tb, Float3 span, float spanNormSqr, float dotProduct, bool fourBit)
    {
        uint index = 0;
        if (spanNormSqr > 0f && dotProduct > 0f)
        {
            uint step = dotProduct < spanNormSqr ? (uint)(dotProduct * 63.49999f / spanNormSqr) : 63u;
            index = fourBit ? tb.Step2(step) : tb.Step1(step);
        }
        return index;
    }

    //---------------------------------------------------------------------------------
    // Pass 1: TryModeG10 — probe modes 11..14 (single region), 4 blocks per group
    //---------------------------------------------------------------------------------

    private static void TryModeG10Kernel(Bc6hParams p, ArrayView<Float4> input, ArrayView<UInt4> inBuf, ArrayView<UInt4> outBuf, ArrayView<uint> tables)
    {
        var tb = new Tables { Data = tables };
        var sh = SharedMemory.Allocate<SharedData>(ThreadGroupSize);
        int gi = Group.IdxX;
        const int maxUsedThread = 16;
        const int blockInGroupCount = ThreadGroupSize / maxUsedThread; // 4
        int blockInGroup = gi / maxUsedThread;
        uint blockID = p.StartBlockId + (uint)(Grid.IdxX * blockInGroupCount + blockInGroup);
        int threadBase = blockInGroup * maxUsedThread;
        int threadInBlock = gi - threadBase;
        uint format = p.Format;

        if (threadInBlock < 16)
        {
            var px = LoadPixel(p, input, blockID, (uint)threadInBlock);
            sh[gi].Pixel = px;
            var hr = HalfRound(px);
            sh[gi].PixelHr = hr;
            float lum = Lum(hr);
            sh[gi].PixelLum = lum;
            var ph = StartQuantize(px, format);
            sh[gi].PixelPh = ph;

            sh[gi].EndPointLow = ph;
            sh[gi].EndPointHigh = ph;
            sh[gi].EndPointLumLow = lum;
            sh[gi].EndPointLumHigh = lum;
        }
        Group.Barrier();

        if (threadInBlock < 8) ReduceEndpoints(sh, gi, 8);
        Group.Barrier();
        if (threadInBlock < 4) ReduceEndpoints(sh, gi, 4);
        Group.Barrier();
        if (threadInBlock < 2) ReduceEndpoints(sh, gi, 2);
        Group.Barrier();
        if (threadInBlock < 1) ReduceEndpoints(sh, gi, 1);
        Group.Barrier();

        // ergod mode_type 11:14
        if (threadInBlock == 0)
        {
            var ep = new Ep(sh[threadBase].EndPointLow, sh[threadBase].EndPointHigh);
            ep = OrderEndpoints(ep, sh[threadBase].PixelPh);
            sh[gi].EndPointLow = ep.Lo;
            sh[gi].EndPointHigh = ep.Hi;
        }
        Group.Barrier();

        if (threadInBlock < 4)
        {
            var endPoint = new Ep(sh[threadBase].EndPointLow, sh[threadBase].EndPointHigh);
            var span = ToF(endPoint.Hi - endPoint.Lo);
            float spanNormSqr = Float3.Dot(span, span);

            uint mode = (uint)threadInBlock + 10;
            uint precX = tb.Prec(mode * 4);
            var q = Quantize(endPoint, precX, format);
            if (tb.Transformed(mode) != 0) q.Hi = q.Hi - q.Lo;

            q = FinishQuantize0(tb, q, mode, out int bad);
            q = StartUnquantizeOne(tb, q, mode, format);
            q = Unquantize(q, precX, format);

            float error = 0f;
            for (int j = 0; j < 16; j++)
            {
                float dotProduct = Float3.Dot(span, ToF(sh[threadBase + j].PixelPh - endPoint.Lo));
                uint index = ProjectIndex(tb, span, spanNormSqr, dotProduct, fourBit: true);
                var r = PaletteColor(q.Lo, q.Hi, tb.Weight4(index), format) - sh[threadBase + j].PixelHr;
                error += Float3.Dot(r, r);
            }
            if (bad != 0) error = 1e20f;

            sh[gi].Error = error;
            sh[gi].BestMode = mode + 1; // candidateModeFlag
        }
        Group.Barrier();

        if (threadInBlock < 2)
        {
            if (sh[gi].Error > sh[gi + 2].Error)
            {
                sh[gi].Error = sh[gi + 2].Error;
                sh[gi].BestMode = sh[gi + 2].BestMode;
            }
        }
        Group.Barrier();
        if (threadInBlock < 1)
        {
            if (sh[gi].Error > sh[gi + 1].Error)
            {
                sh[gi].Error = sh[gi + 1].Error;
                sh[gi].BestMode = sh[gi + 1].BestMode;
            }
            outBuf[(long)blockID] = new UInt4(Interop.FloatAsInt(sh[gi].Error), sh[gi].BestMode, 0, 0);
        }
    }

    //---------------------------------------------------------------------------------
    // Pass 2: TryModeLE10 — probe two-region modes 1..10 (ModeId selects), 2 blocks/group
    //---------------------------------------------------------------------------------

    private static void TryModeLE10Kernel(Bc6hParams p, ArrayView<Float4> input, ArrayView<UInt4> inBuf, ArrayView<UInt4> outBuf, ArrayView<uint> tables)
    {
        var tb = new Tables { Data = tables };
        var sh = SharedMemory.Allocate<SharedData>(ThreadGroupSize);
        int gi = Group.IdxX;
        const int maxUsedThread = 32;
        const int blockInGroupCount = ThreadGroupSize / maxUsedThread; // 2
        int blockInGroup = gi / maxUsedThread;
        uint blockID = p.StartBlockId + (uint)(Grid.IdxX * blockInGroupCount + blockInGroup);
        int threadBase = blockInGroup * maxUsedThread;
        int threadInBlock = gi - threadBase;
        uint format = p.Format;

        if (threadInBlock < 16)
        {
            var px = LoadPixel(p, input, blockID, (uint)threadInBlock);
            sh[gi].Pixel = px;
            var hr = HalfRound(px);
            sh[gi].PixelHr = hr;
            sh[gi].PixelLum = Lum(hr);
            sh[gi].PixelPh = StartQuantize(px, format);
        }
        Group.Barrier();

        // ergod mode_type 1:10 (every thread is one of the 32 partitions)
        {
            // find_axis
            var ep0 = new Ep(new Int3(int.MaxValue), new Int3(int.MinValue));
            var ep1 = new Ep(new Int3(int.MaxValue), new Int3(int.MinValue));
            float lum0Lo = float.MaxValue, lum0Hi = float.MinValue;
            float lum1Lo = float.MaxValue, lum1Hi = float.MinValue;

            uint bit = tb.SectionBit(threadInBlock);
            for (int i = 0; i < 16; i++)
            {
                var pixelPh = sh[threadBase + i].PixelPh;
                float pixelLum = sh[threadBase + i].PixelLum;
                if (((bit >> i) & 1) != 0)
                {
                    if (lum1Lo > pixelLum) { ep1.Lo = pixelPh; lum1Lo = pixelLum; }
                    if (lum1Hi < pixelLum) { ep1.Hi = pixelPh; lum1Hi = pixelLum; }
                }
                else
                {
                    if (lum0Lo > pixelLum) { ep0.Lo = pixelPh; lum0Lo = pixelLum; }
                    if (lum0Hi < pixelLum) { ep0.Hi = pixelPh; lum0Hi = pixelLum; }
                }
            }

            // compute_index (region 0 fix-up pixel is 0; region 1 uses the partition's anchor)
            ep0 = OrderEndpoints(ep0, sh[threadBase].PixelPh);
            ep1 = OrderEndpoints(ep1, sh[threadBase + (int)tb.FixUp(threadInBlock)].PixelPh);
            var span0 = ToF(ep0.Hi - ep0.Lo);
            var span1 = ToF(ep1.Hi - ep1.Lo);
            float norm0 = Float3.Dot(span0, span0);
            float norm1 = Float3.Dot(span1, span1);

            uint mode = p.ModeId;
            uint precX = tb.Prec(mode * 4);
            var q0 = Quantize(ep0, precX, format);
            var q1 = Quantize(ep1, precX, format);

            bool transformed = tb.Transformed(mode) != 0;
            if (transformed)
            {
                q0.Hi = q0.Hi - q0.Lo;
                q1.Lo = q1.Lo - q0.Lo;
                q1.Hi = q1.Hi - q0.Lo;
            }

            q0 = FinishQuantize0(tb, q0, mode, out int bad0);
            q1 = FinishQuantize1(tb, q1, mode, out int bad1);

            // start_unquantize_two
            uint py = tb.Prec(mode * 4 + 1), pz = tb.Prec(mode * 4 + 2), pw = tb.Prec(mode * 4 + 3);
            if (format == SignedF16) q0.Lo = SignExtend(precX, precX, precX, q0.Lo);
            if (format == SignedF16 || transformed)
            {
                q0.Hi = SignExtend(py, pz, pw, q0.Hi);
                q1.Lo = SignExtend(py, pz, pw, q1.Lo);
                q1.Hi = SignExtend(py, pz, pw, q1.Hi);
            }
            if (transformed)
            {
                q0.Hi = q0.Hi + q0.Lo;
                q1.Lo = q1.Lo + q0.Lo;
                q1.Hi = q1.Hi + q0.Lo;
            }

            q0 = Unquantize(q0, precX, format);
            q1 = Unquantize(q1, precX, format);

            float error = 0f;
            for (int j = 0; j < 16; j++)
            {
                Float3 pixelR;
                if (((bit >> j) & 1) != 0)
                {
                    float dotProduct = Float3.Dot(span1, ToF(sh[threadBase + j].PixelPh - ep1.Lo));
                    uint index = ProjectIndex(tb, span1, norm1, dotProduct, fourBit: false);
                    pixelR = PaletteColor(q1.Lo, q1.Hi, tb.Weight3(index), format);
                }
                else
                {
                    float dotProduct = Float3.Dot(span0, ToF(sh[threadBase + j].PixelPh - ep0.Lo));
                    uint index = ProjectIndex(tb, span0, norm0, dotProduct, fourBit: false);
                    pixelR = PaletteColor(q0.Lo, q0.Hi, tb.Weight3(index), format);
                }
                var r = pixelR - sh[threadBase + j].PixelHr;
                error += Float3.Dot(r, r);
            }
            if ((bad0 | bad1) != 0) error = 1e20f;

            sh[gi].Error = error;
            sh[gi].BestMode = mode + 1; // candidateModeFlag
            sh[gi].BestPartition = (uint)threadInBlock;
        }
        Group.Barrier();

        if (threadInBlock < 16) ReduceError(sh, gi, 16);
        Group.Barrier();
        if (threadInBlock < 8) ReduceError(sh, gi, 8);
        Group.Barrier();
        if (threadInBlock < 4) ReduceError(sh, gi, 4);
        Group.Barrier();
        if (threadInBlock < 2) ReduceError(sh, gi, 2);
        Group.Barrier();
        if (threadInBlock < 1)
        {
            ReduceError(sh, gi, 1);
            var prev = inBuf[(long)blockID];
            if (Interop.IntAsFloat(prev.X) > sh[gi].Error)
                outBuf[(long)blockID] = new UInt4(Interop.FloatAsInt(sh[gi].Error), sh[gi].BestMode, sh[gi].BestPartition, 0);
            else
                outBuf[(long)blockID] = prev;
        }
    }

    //---------------------------------------------------------------------------------
    // Pass 3: EncodeBlock — re-derive endpoints for the winning mode and pack bits
    //---------------------------------------------------------------------------------

    private static void EncodeBlockKernel(Bc6hParams p, ArrayView<Float4> input, ArrayView<UInt4> inBuf, ArrayView<UInt4> outBuf, ArrayView<uint> tables)
    {
        var tb = new Tables { Data = tables };
        var sh = SharedMemory.Allocate<SharedData>(ThreadGroupSize);
        int gi = Group.IdxX;
        const int maxUsedThread = 32;
        const int blockInGroupCount = ThreadGroupSize / maxUsedThread; // 2
        int blockInGroup = gi / maxUsedThread;
        uint blockID = p.StartBlockId + (uint)(Grid.IdxX * blockInGroupCount + blockInGroup);
        int threadBase = blockInGroup * maxUsedThread;
        int threadInBlock = gi - threadBase;
        uint format = p.Format;

        if (threadInBlock < 16)
        {
            var px = LoadPixel(p, input, blockID, (uint)threadInBlock);
            sh[gi].Pixel = px;
            sh[gi].PixelLum = Lum(px);
            sh[gi].PixelPh = StartQuantize(px, format);
        }
        Group.Barrier();

        var best = inBuf[(long)blockID];
        uint bestMode = best.Y;
        uint bestPartition = best.Z;

        var blk = new UInt4(0);

        // Each of the 32 threads seeds a candidate endpoint: threads 0..15 region 0,
        // threads 16..31 region 1 (pixel threadInBlock & 0xF), or the identity (empty).
        {
            var lo = new Int3(int.MaxValue);
            var hi = new Int3(int.MinValue);
            float lumLo = float.MaxValue, lumHi = float.MinValue;

            var pixelPh = sh[threadBase + (threadInBlock & 0xF)].PixelPh;
            float pixelLum = sh[threadBase + (threadInBlock & 0xF)].PixelLum;
            bool take;
            if (threadInBlock < 16)
                take = bestMode > 10 || ((tb.SectionBit(bestPartition) >> threadInBlock) & 1) == 0;
            else
                take = bestMode <= 10 && ((tb.SectionBit(bestPartition) >> (threadInBlock & 0xF)) & 1) == 1;
            if (take)
            {
                lo = pixelPh;
                hi = pixelPh;
                lumLo = pixelLum;
                lumHi = pixelLum;
            }

            sh[gi].EndPointLow = lo;
            sh[gi].EndPointHigh = hi;
            sh[gi].EndPointLumLow = lumLo;
            sh[gi].EndPointLumHigh = lumHi;
        }
        Group.Barrier();
        if ((threadInBlock & 0xF) < 8) ReduceEndpoints(sh, gi, 8);
        Group.Barrier();
        if ((threadInBlock & 0xF) < 4) ReduceEndpoints(sh, gi, 4);
        Group.Barrier();
        if ((threadInBlock & 0xF) < 2) ReduceEndpoints(sh, gi, 2);
        Group.Barrier();
        if ((threadInBlock & 0xF) < 1) ReduceEndpoints(sh, gi, 1);
        Group.Barrier();

        if (threadInBlock < 2)
        {
            // find_axis: region endpoints from threads 0 / 16
            var ep = new Ep(sh[threadBase + threadInBlock * 16].EndPointLow, sh[threadBase + threadInBlock * 16].EndPointHigh);
            int fixup = 0;
            if (threadInBlock == 1 && bestMode <= 10) fixup = (int)tb.FixUp(bestPartition);
            ep = OrderEndpoints(ep, sh[threadBase + fixup].PixelPh);
            sh[gi].EndPointLow = ep.Lo;
            sh[gi].EndPointHigh = ep.Hi;
        }
        Group.Barrier();

        if (threadInBlock < 16)
        {
            uint bits = bestMode <= 10 ? tb.SectionBit(bestPartition) : 0u;
            int region = (int)((bits >> threadInBlock) & 1);
            var rLo = sh[threadBase + region].EndPointLow;
            var span = ToF(sh[threadBase + region].EndPointHigh - rLo);
            float dotProduct = Float3.Dot(span, ToF(sh[threadBase + threadInBlock].PixelPh - rLo));
            float spanNormSqr = Float3.Dot(span, span);

            if (bestMode > 10)
            {
                uint index = ProjectIndex(tb, span, spanNormSqr, dotProduct, fourBit: true);
                if (threadInBlock == 0) blk.Z |= index << 1;
                else if (threadInBlock < 8) blk.Z |= index << (threadInBlock * 4);
                else blk.W |= index << ((threadInBlock - 8) * 4);
            }
            else
            {
                uint index = ProjectIndex(tb, span, spanNormSqr, dotProduct, fourBit: false);
                uint fixup = tb.FixUp(bestPartition);
                int offsetX = fixup != 2 ? 1 : 0;
                int offsetY = fixup == 15 ? 1 : 0;

                if (threadInBlock == 0) blk.Z |= index << 18;
                else if (threadInBlock < 3) blk.Z |= index << (20 + (threadInBlock - 1) * 3);
                else if (threadInBlock < 5) blk.Z |= index << (25 + (threadInBlock - 3) * 3 + offsetX);
                else if (threadInBlock == 5)
                {
                    blk.W |= index >> (1 - offsetX); // index >> !offset.x
                    if (offsetX == 0) blk.Z |= index << 31;
                }
                else if (threadInBlock < 9) blk.W |= index << (2 + (threadInBlock - 6) * 3 + offsetX);
                else blk.W |= index << (11 + (threadInBlock - 9) * 3 + offsetY);
            }

            sh[gi].ScratchX = blk.Z;
            sh[gi].ScratchY = blk.W;
        }
        Group.Barrier();
        if (threadInBlock < 8) { sh[gi].ScratchX |= sh[gi + 8].ScratchX; sh[gi].ScratchY |= sh[gi + 8].ScratchY; }
        Group.Barrier();
        if (threadInBlock < 4) { sh[gi].ScratchX |= sh[gi + 4].ScratchX; sh[gi].ScratchY |= sh[gi + 4].ScratchY; }
        Group.Barrier();
        if (threadInBlock < 2) { sh[gi].ScratchX |= sh[gi + 2].ScratchX; sh[gi].ScratchY |= sh[gi + 2].ScratchY; }
        Group.Barrier();
        if (threadInBlock < 1)
        {
            sh[gi].ScratchX |= sh[gi + 1].ScratchX;
            sh[gi].ScratchY |= sh[gi + 1].ScratchY;
            blk.Z = sh[gi].ScratchX;
            blk.W = sh[gi].ScratchY;
        }
        Group.Barrier();

        uint mode = bestMode - 1;
        bool transformed = tb.Transformed(mode) != 0;
        uint precX = tb.Prec(mode * 4);
        if (threadInBlock == 2)
        {
            var q = Quantize(new Ep(sh[threadBase].EndPointLow, sh[threadBase].EndPointHigh), precX, format);
            if (transformed) q.Hi = q.Hi - q.Lo;
            sh[gi].EndPointLow = q.Lo;
            sh[gi].EndPointHigh = q.Hi;
        }
        Group.Barrier();
        if (threadInBlock == 3)
        {
            var ep0 = sh[threadBase + 2].EndPointLow;
            if (bestMode <= 10)
            {
                var q = Quantize(new Ep(sh[threadBase + 1].EndPointLow, sh[threadBase + 1].EndPointHigh), precX, format);
                if (transformed)
                {
                    q.Lo = q.Lo - ep0;
                    q.Hi = q.Hi - ep0;
                }
                sh[gi].EndPointLow = q.Lo;
                sh[gi].EndPointHigh = q.Hi;
            }
        }
        Group.Barrier();

        if (threadInBlock < 2)
        {
            var q = new Ep(sh[threadBase + threadInBlock + 2].EndPointLow, sh[threadBase + threadInBlock + 2].EndPointHigh);
            if (threadInBlock == 0)
            {
                // best_mode > 10 used HLSL finish_quantize, <= 10 finish_quantize_0;
                // their bodies are identical (only the out/inout flag differed).
                q = FinishQuantize0(tb, q, mode, out _);
            }
            else if (bestMode <= 10)
            {
                q = FinishQuantize1(tb, q, mode, out _);
            }
            sh[gi].EndPointLow = q.Lo;
            sh[gi].EndPointHigh = q.Hi;
        }
        Group.Barrier();

        if (threadInBlock == 0)
        {
            var e0 = new Ep(sh[threadBase].EndPointLow, sh[threadBase].EndPointHigh);
            var e1 = new Ep(sh[threadBase + 1].EndPointLow, sh[threadBase + 1].EndPointHigh);
            outBuf[(long)blockID] = BlockPackage(tb, blk, e0, e1, mode, bestPartition);
        }
    }

    //---------------------------------------------------------------------------------
    // block_package_one / block_package_two (table-driven; see the file comment)
    //---------------------------------------------------------------------------------

    private static int Field(Ep e0, Ep e1, int field) => field switch
    {
        0 => e0.Lo.X, 1 => e0.Lo.Y, 2 => e0.Lo.Z,
        3 => e0.Hi.X, 4 => e0.Hi.Y, 5 => e0.Hi.Z,
        6 => e1.Lo.X, 7 => e1.Lo.Y, 8 => e1.Lo.Z,
        9 => e1.Hi.X, 10 => e1.Hi.Y, _ => e1.Hi.Z,
    };

    /// <param name="mode">Mode index 0..13 (best_mode - 1).</param>
    private static UInt4 BlockPackage(Tables tb, UInt4 blk, Ep e0, Ep e1, uint mode, uint partition)
    {
        bool twoRegion = mode < 10;
        blk.X = 0;
        blk.Y = 0;
        blk.Z &= twoRegion ? 0xFFFC0000u : 0xFFFFFFFEu;
        int headerBits = twoRegion ? 82 : 65;
        uint modeMemory = tb.ModeMemory(mode);
        int tableBase = (int)mode * PackStride;
        for (int pos = 0; pos < headerBits; pos++)
        {
            uint entry = tb.Pack(tableBase + pos);
            int field = (int)(entry >> 8);
            int bit = (int)(entry & 0xFF);
            uint v = field == 12 ? (modeMemory >> bit) & 1u
                : field == 13 ? (partition >> bit) & 1u
                : (uint)((Field(e0, e1, field) >> bit) & 1);
            int word = pos >> 5;
            uint mask = v << (pos & 31);
            if (word == 0) blk.X |= mask;
            else if (word == 1) blk.Y |= mask;
            else blk.Z |= mask;
        }
        return blk;
    }
}
