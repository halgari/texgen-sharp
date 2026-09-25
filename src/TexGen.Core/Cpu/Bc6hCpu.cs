//-------------------------------------------------------------------------------------
// Cpu/Bc6hCpu.cs
//
// DirectXTex CPU BC6H encoder (port of D3DX_BC6H::Encode in BC6HBC7.cpp). For each of
// the 14 modes it scores every shape with a rough least-squares fit (RoughMSE), then
// refines the best quarter: quantize endpoints, assign indices, perturb endpoints
// channel by channel, and emit the best block. Unlike the GPU shader, negative inputs
// are preserved for BC6H_SF16.
//
// Two DirectXTex quirks are kept for bit-exact parity with texconv:
//   * OptimizeEndPoints collects region pixels with g_aPartitionTable[p] (not
//     [uPartitions]), so region 0 of 2-region modes perturbs against all 16 pixels.
//   * Errors are computed in float on half-float integer values.
//-------------------------------------------------------------------------------------

using System.Numerics;
using System.Runtime.CompilerServices;

namespace TexGen.Cpu;

[SkipLocalsInit]
internal static class Bc6hCpu
{
    private const int NumModes = 14;
    private const int MaxShapes = 32;
    private const int F16Max = IntColor.F16Max;

    // EField
    private const byte NA = 0, M = 1, D = 2, RW = 3, RX = 4, RY = 5, RZ = 6, GW = 7, GX = 8, GY = 9, GZ = 10,
        BW = 11, BX = 12, BY = 13, BZ = 14;

    private readonly struct ModeInfo(byte mode, byte partitions, bool transformed, byte indexPrec,
        LdrColorA prec00, LdrColorA prec01, LdrColorA prec10, LdrColorA prec11)
    {
        public readonly byte Mode = mode;
        public readonly byte Partitions = partitions;
        public readonly bool Transformed = transformed;
        public readonly byte IndexPrec = indexPrec;
        /// <summary>RGBAPrec[0][0]: endpoint A precision (the one used for quantization).</summary>
        public readonly LdrColorA Prec00 = prec00;
        public readonly LdrColorA Prec01 = prec01;
        public readonly LdrColorA Prec10 = prec10;
        public readonly LdrColorA Prec11 = prec11;
    }

    private static LdrColorA P(byte r, byte g, byte b) => new(r, g, b, 0);

    // Mode, Partitions, Transformed, IndexPrec, RGBAPrec
    private static readonly ModeInfo[] Info =
    [
        new(0x00, 1, true, 3, P(10, 10, 10), P(5, 5, 5), P(5, 5, 5), P(5, 5, 5)), // Mode 1
        new(0x01, 1, true, 3, P(7, 7, 7), P(6, 6, 6), P(6, 6, 6), P(6, 6, 6)), // Mode 2
        new(0x02, 1, true, 3, P(11, 11, 11), P(5, 4, 4), P(5, 4, 4), P(5, 4, 4)), // Mode 3
        new(0x06, 1, true, 3, P(11, 11, 11), P(4, 5, 4), P(4, 5, 4), P(4, 5, 4)), // Mode 4
        new(0x0a, 1, true, 3, P(11, 11, 11), P(4, 4, 5), P(4, 4, 5), P(4, 4, 5)), // Mode 5
        new(0x0e, 1, true, 3, P(9, 9, 9), P(5, 5, 5), P(5, 5, 5), P(5, 5, 5)), // Mode 6
        new(0x12, 1, true, 3, P(8, 8, 8), P(6, 5, 5), P(6, 5, 5), P(6, 5, 5)), // Mode 7
        new(0x16, 1, true, 3, P(8, 8, 8), P(5, 6, 5), P(5, 6, 5), P(5, 6, 5)), // Mode 8
        new(0x1a, 1, true, 3, P(8, 8, 8), P(5, 5, 6), P(5, 5, 6), P(5, 5, 6)), // Mode 9
        new(0x1e, 1, false, 3, P(6, 6, 6), P(6, 6, 6), P(6, 6, 6), P(6, 6, 6)), // Mode 10
        new(0x03, 0, false, 4, P(10, 10, 10), P(10, 10, 10), P(0, 0, 0), P(0, 0, 0)), // Mode 11
        new(0x07, 0, true, 4, P(11, 11, 11), P(9, 9, 9), P(0, 0, 0), P(0, 0, 0)), // Mode 12
        new(0x0b, 0, true, 4, P(12, 12, 12), P(8, 8, 8), P(0, 0, 0), P(0, 0, 0)), // Mode 13
        new(0x0f, 0, true, 4, P(16, 16, 16), P(4, 4, 4), P(0, 0, 0), P(0, 0, 0)), // Mode 14
    ];

    [InlineArray(MaxShapes * 2)]
    private struct UnqEndPtsArray
    {
        private IntEndPntPair _element0;
    }

    private struct EncodeParams
    {
        public float BestErr;
        public bool Signed;
        public int Mode;
        public int Shape;
        /// <summary>aUnqEndPts[shape][region] flattened to [shape * 2 + region].</summary>
        public UnqEndPtsArray UnqEndPts;
        public Buffer16<IntColor> IPixels;
        public Buffer16<HdrColorA> HdrPixels;
    }

    /// <summary>Encode 16 linear-float pixels (row-major) into one BC6H block.</summary>
    public static void Encode(Span<byte> bc, ReadOnlySpan<Vector4> color, bool signed)
    {
        EncodeParams ep = default;
        ep.BestErr = float.MaxValue;
        ep.Signed = signed;
        for (int i = 0; i < 16; i++)
        {
            ep.HdrPixels[i] = HdrColorA.FromVector(color[i]);
            ep.IPixels[i].Set(ep.HdrPixels[i], signed);
        }

        Bits128 block = default;
        Span<float> roughMse = stackalloc float[MaxShapes];
        Span<int> shapes = stackalloc int[MaxShapes];

        for (ep.Mode = 0; ep.Mode < NumModes && ep.BestErr > 0; ++ep.Mode)
        {
            int uShapes = Info[ep.Mode].Partitions != 0 ? 32 : 1;
            // Number of rough cases to look at. reasonable values of this are 1, uShapes/4, and uShapes
            // uShapes/4 gets nearly all the cases; you can increase that a bit (say by 3 or 4) if you really want to squeeze the last bit out
            int items = Math.Max(1, uShapes >> 2);

            // pick the best uItems shapes and refine these.
            for (ep.Shape = 0; ep.Shape < uShapes; ++ep.Shape)
            {
                roughMse[ep.Shape] = RoughMse(ref ep);
                shapes[ep.Shape] = ep.Shape;
            }

            // Bubble up the first uItems items
            for (int i = 0; i < items; i++)
            {
                for (int j = i + 1; j < uShapes; j++)
                {
                    if (roughMse[i] > roughMse[j])
                    {
                        (roughMse[i], roughMse[j]) = (roughMse[j], roughMse[i]);
                        (shapes[i], shapes[j]) = (shapes[j], shapes[i]);
                    }
                }
            }

            for (int i = 0; i < items && ep.BestErr > 0; i++)
            {
                ep.Shape = shapes[i];
                Refine(ref ep, ref block);
            }
        }

        ((ReadOnlySpan<byte>)block).CopyTo(bc);
    }

    //---------------------------------------------------------------------------------
    // Helpers
    //---------------------------------------------------------------------------------

    private static int Quantize(int value, int prec, bool signed)
    {
        int q;
        if (signed)
        {
            bool s = false;
            if (value < 0)
            {
                s = true;
                value = -value;
            }
            q = prec >= 16 ? value : (value << (prec - 1)) / (F16Max + 1);
            if (s) q = -q;
        }
        else
        {
            q = prec >= 15 ? value : (value << prec) / (F16Max + 1);
        }
        return q;
    }

    private static int Unquantize(int comp, int bitsPerComp, bool signed)
    {
        int unq;
        if (signed)
        {
            if (bitsPerComp >= 16)
            {
                unq = comp;
            }
            else
            {
                bool s = false;
                if (comp < 0)
                {
                    s = true;
                    comp = -comp;
                }

                if (comp == 0) unq = 0;
                else if (comp >= ((1 << (bitsPerComp - 1)) - 1)) unq = 0x7FFF;
                else unq = ((comp << 15) + 0x4000) >> (bitsPerComp - 1);

                if (s) unq = -unq;
            }
        }
        else
        {
            if (bitsPerComp >= 15) unq = comp;
            else if (comp == 0) unq = 0;
            else if (comp == ((1 << bitsPerComp) - 1)) unq = 0xFFFF;
            else unq = ((comp << 16) + 0x8000) >> bitsPerComp;
        }
        return unq;
    }

    private static int FinishUnquantize(int comp, bool signed) =>
        signed
            ? (comp < 0 ? -(((-comp) * 31) >> 5) : (comp * 31) >> 5) // scale the magnitude by 31/32
            : (comp * 31) >> 6; // scale the magnitude by 31/64

    private static void TransformForward(ref Buffer2<IntEndPntPair> e)
    {
        e[0].B.Sub(e[0].A);
        e[1].A.Sub(e[0].A);
        e[1].B.Sub(e[0].A);
    }

    private static void TransformInverse(ref Buffer2<IntEndPntPair> e, in LdrColorA prec, bool signed)
    {
        var wrapMask = new IntColor((1 << prec.R) - 1, (1 << prec.G) - 1, (1 << prec.B) - 1);
        e[0].B.Add(e[0].A);
        e[0].B.And(wrapMask);
        e[1].A.Add(e[0].A);
        e[1].A.And(wrapMask);
        e[1].B.Add(e[0].A);
        e[1].B.And(wrapMask);
        if (signed)
        {
            e[0].B.SignExtend(prec);
            e[1].A.SignExtend(prec);
            e[1].B.SignExtend(prec);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Norm(in IntColor a, in IntColor b)
    {
        float dr = (float)a.R - b.R;
        float dg = (float)a.G - b.G;
        float db = (float)a.B - b.B;
        return dr * dr + dg * dg + db * db;
    }

    /// <summary>Number of bits needed to store n, handling signed or unsigned cases.</summary>
    private static int NBits(int n, bool isSigned)
    {
        int nb;
        if (n == 0) return 0; // no bits needed for 0, signed or not
        if (n > 0)
        {
            for (nb = 0; n != 0; ++nb, n >>= 1) { }
            return nb + (isSigned ? 1 : 0);
        }
        for (nb = 0; n < -1; ++nb, n >>= 1) { }
        return nb + 1;
    }

    private static bool EndPointsFit(ref EncodeParams ep, ref Buffer2<IntEndPntPair> e)
    {
        ref readonly var info = ref Info[ep.Mode];
        bool transformed = info.Transformed;
        bool isSigned = ep.Signed;
        ref readonly var prec0 = ref info.Prec00;
        ref readonly var prec1 = ref info.Prec01;
        ref readonly var prec2 = ref info.Prec10;
        ref readonly var prec3 = ref info.Prec11;

        if (NBits(e[0].A.R, isSigned) > prec0.R || NBits(e[0].B.R, transformed || isSigned) > prec1.R
            || NBits(e[0].A.G, isSigned) > prec0.G || NBits(e[0].B.G, transformed || isSigned) > prec1.G
            || NBits(e[0].A.B, isSigned) > prec0.B || NBits(e[0].B.B, transformed || isSigned) > prec1.B)
            return false;

        if (info.Partitions != 0)
        {
            if (NBits(e[1].A.R, transformed || isSigned) > prec2.R || NBits(e[1].B.R, transformed || isSigned) > prec3.R
                || NBits(e[1].A.G, transformed || isSigned) > prec2.G || NBits(e[1].B.G, transformed || isSigned) > prec3.G
                || NBits(e[1].A.B, transformed || isSigned) > prec2.B || NBits(e[1].B.B, transformed || isSigned) > prec3.B)
                return false;
        }
        return true;
    }

    private static void GeneratePaletteQuantized(ref EncodeParams ep, in IntEndPntPair endPts, Span<IntColor> palette)
    {
        ref readonly var info = ref Info[ep.Mode];
        int indexPrec = info.IndexPrec;
        int numIndices = 1 << indexPrec;
        ref readonly var prec = ref info.Prec00;
        bool signed = ep.Signed;

        // scale endpoints
        int ar = Unquantize(endPts.A.R, prec.R, signed);
        int ag = Unquantize(endPts.A.G, prec.G, signed);
        int ab = Unquantize(endPts.A.B, prec.B, signed);
        int br = Unquantize(endPts.B.R, prec.R, signed);
        int bg = Unquantize(endPts.B.G, prec.G, signed);
        int bb = Unquantize(endPts.B.B, prec.B, signed);

        // interpolate
        var weights = Bc67Tables.Weights(indexPrec);
        for (int i = 0; i < numIndices; ++i)
        {
            int w = weights[i];
            int iw = Bc67Tables.WeightMax - w;
            palette[i].R = FinishUnquantize((ar * iw + br * w + Bc67Tables.WeightRound) >> Bc67Tables.WeightShift, signed);
            palette[i].G = FinishUnquantize((ag * iw + bg * w + Bc67Tables.WeightRound) >> Bc67Tables.WeightShift, signed);
            palette[i].B = FinishUnquantize((ab * iw + bb * w + Bc67Tables.WeightRound) >> Bc67Tables.WeightShift, signed);
        }
    }

    /// <summary>
    /// Given a collection of colors and quantized endpoints, generate a palette, choose
    /// best entries, and return a single total error.
    /// </summary>
    private static float MapColorsQuantized(ref EncodeParams ep, ReadOnlySpan<IntColor> colors, int np, in IntEndPntPair endPts)
    {
        int numIndices = 1 << Info[ep.Mode].IndexPrec;
        Buffer16<IntColor> palette;
        Unsafe.SkipInit(out palette);
        GeneratePaletteQuantized(ref ep, endPts, palette);
        return Bc67Simd.HdrErrorSum(colors, np, palette, numIndices, Bc67Simd.Level);
    }

    private static float PerturbOne(ref EncodeParams ep, ReadOnlySpan<IntColor> colors, int np, int ch,
        in IntEndPntPair oldEndPts, out IntEndPntPair newEndPts, float oldErr, int doB)
    {
        int prec = Info[ep.Mode].Prec00[ch];
        float minErr = oldErr;
        int bestStep = 0;

        // copy real endpoints so we can perturb them
        newEndPts = oldEndPts;
        IntEndPntPair tmpEndPts = oldEndPts;

        // do a logarithmic search for the best error for this endpoint (which)
        for (int step = 1 << (prec - 1); step != 0; step >>= 1)
        {
            bool improved = false;
            for (int sign = -1; sign <= 1; sign += 2)
            {
                if (doB == 0)
                {
                    tmpEndPts.A[ch] = newEndPts.A[ch] + sign * step;
                    if (tmpEndPts.A[ch] < 0 || tmpEndPts.A[ch] >= (1 << prec)) continue;
                }
                else
                {
                    tmpEndPts.B[ch] = newEndPts.B[ch] + sign * step;
                    if (tmpEndPts.B[ch] < 0 || tmpEndPts.B[ch] >= (1 << prec)) continue;
                }

                float err = MapColorsQuantized(ref ep, colors, np, tmpEndPts);
                if (err < minErr)
                {
                    improved = true;
                    minErr = err;
                    bestStep = sign * step;
                }
            }
            // if this was an improvement, move the endpoint and continue search from there
            if (improved)
            {
                if (doB == 0) newEndPts.A[ch] += bestStep;
                else newEndPts.B[ch] += bestStep;
            }
        }
        return minErr;
    }

    private static void OptimizeOne(ref EncodeParams ep, ReadOnlySpan<IntColor> colors, int np, float orgErr,
        in IntEndPntPair orgEndPts, out IntEndPntPair optEndPts)
    {
        float optErr = orgErr;
        optEndPts = orgEndPts;

        // now optimize each channel separately
        for (int ch = 0; ch < 3; ++ch)
        {
            // figure out which endpoint when perturbed gives the most improvement and start there
            // if we just alternate, we can easily end up in a local minima
            float err0 = PerturbOne(ref ep, colors, np, ch, optEndPts, out var newA, optErr, 0); // perturb endpt A
            float err1 = PerturbOne(ref ep, colors, np, ch, optEndPts, out var newB, optErr, 1); // perturb endpt B

            int doB;
            if (err0 < err1)
            {
                if (err0 >= optErr) continue;
                optEndPts.A[ch] = newA.A[ch];
                optErr = err0;
                doB = 1; // do B next
            }
            else
            {
                if (err1 >= optErr) continue;
                optEndPts.B[ch] = newB.B[ch];
                optErr = err1;
                doB = 0; // do A next
            }

            // now alternate endpoints and keep trying until there is no improvement
            for (;;)
            {
                float err = PerturbOne(ref ep, colors, np, ch, optEndPts, out var newEndPts, optErr, doB);
                if (err >= optErr) break;
                if (doB == 0) optEndPts.A[ch] = newEndPts.A[ch];
                else optEndPts.B[ch] = newEndPts.B[ch];
                optErr = err;
                doB = 1 - doB; // now move the other endpoint
            }
        }
    }

    private static void OptimizeEndPoints(ref EncodeParams ep, ReadOnlySpan<float> orgErr,
        ref Buffer2<IntEndPntPair> orgEndPts, ref Buffer2<IntEndPntPair> optEndPts)
    {
        int partitions = Info[ep.Mode].Partitions;
        Buffer16<IntColor> pixels;
        Unsafe.SkipInit(out pixels);

        for (int p = 0; p <= partitions; ++p)
        {
            // collect the pixels in the region
            // (DirectXTex indexes g_aPartitionTable with the region p here, not uPartitions.)
            int np = 0;
            for (int i = 0; i < 16; ++i)
            {
                if (Bc67Tables.Partition(p, ep.Shape, i) == p) pixels[np++] = ep.IPixels[i];
            }

            OptimizeOne(ref ep, pixels, np, orgErr[p], orgEndPts[p], out optEndPts[p]);
        }
    }

    /// <summary>Swap endpoints as needed to ensure that the indices at fix up have a 0 high-order bit.</summary>
    private static void SwapIndices(ref EncodeParams ep, ref Buffer2<IntEndPntPair> endPts, Span<int> indices)
    {
        int partitions = Info[ep.Mode].Partitions;
        int numIndices = 1 << Info[ep.Mode].IndexPrec;
        int highIndexBit = numIndices >> 1;

        for (int p = 0; p <= partitions; ++p)
        {
            int i = Bc67Tables.FixUpIndex(partitions, ep.Shape, p);
            if ((indices[i] & highIndexBit) != 0)
            {
                // high bit is set, swap the aEndPts and indices for this region
                (endPts[p].A, endPts[p].B) = (endPts[p].B, endPts[p].A);

                for (int j = 0; j < 16; ++j)
                {
                    if (Bc67Tables.Partition(partitions, ep.Shape, j) == p) indices[j] = numIndices - 1 - indices[j];
                }
            }
        }
    }

    /// <summary>Assign indices given a tile, shape, and quantized endpoints; return total error for each region.</summary>
    private static void AssignIndices(ref EncodeParams ep, ref Buffer2<IntEndPntPair> endPts, Span<int> indices, Span<float> totErr)
    {
        int partitions = Info[ep.Mode].Partitions;
        int numIndices = 1 << Info[ep.Mode].IndexPrec;

        // build list of possibles
        Buffer16<IntColor> palette0, palette1;
        Unsafe.SkipInit(out palette0);
        Unsafe.SkipInit(out palette1);
        GeneratePaletteQuantized(ref ep, endPts[0], palette0);
        totErr[0] = 0;
        if (partitions > 0)
        {
            GeneratePaletteQuantized(ref ep, endPts[1], palette1);
            totErr[1] = 0;
        }

        ReadOnlySpan<IntColor> pal0 = palette0;
        ReadOnlySpan<IntColor> pal1 = palette1;
        for (int i = 0; i < 16; ++i)
        {
            int region = Bc67Tables.Partition(partitions, ep.Shape, i);
            var palette = region == 0 ? pal0 : pal1;
            float bestErr = Norm(ep.IPixels[i], palette[0]);
            indices[i] = 0;

            for (int j = 1; j < numIndices && bestErr > 0; ++j)
            {
                float err = Norm(ep.IPixels[i], palette[j]);
                if (err > bestErr) break; // error increased, so we're done searching
                if (err < bestErr)
                {
                    bestErr = err;
                    indices[i] = j;
                }
            }
            totErr[region] += bestErr;
        }
    }

    private static void QuantizeEndPts(ref EncodeParams ep, ref Buffer2<IntEndPntPair> qnt)
    {
        ref readonly var prec = ref Info[ep.Mode].Prec00;
        int partitions = Info[ep.Mode].Partitions;
        bool signed = ep.Signed;

        for (int p = 0; p <= partitions; ++p)
        {
            ref readonly var unq = ref ep.UnqEndPts[ep.Shape * 2 + p];
            qnt[p].A.R = Quantize(unq.A.R, prec.R, signed);
            qnt[p].A.G = Quantize(unq.A.G, prec.G, signed);
            qnt[p].A.B = Quantize(unq.A.B, prec.B, signed);
            qnt[p].B.R = Quantize(unq.B.R, prec.R, signed);
            qnt[p].B.G = Quantize(unq.B.G, prec.G, signed);
            qnt[p].B.B = Quantize(unq.B.B, prec.B, signed);
        }
    }

    private static void EmitBlock(ref EncodeParams ep, ref Buffer2<IntEndPntPair> e, ReadOnlySpan<int> indices, ref Bits128 block)
    {
        ref readonly var info = ref Info[ep.Mode];
        int realMode = info.Mode;
        int partitions = info.Partitions;
        int indexPrec = info.IndexPrec;
        int headerBits = partitions > 0 ? 82 : 65;
        var fields = DescField.Slice(ep.Mode * 82, 82);
        var bits = DescBit.Slice(ep.Mode * 82, 82);
        int startBit = 0;

        while (startBit < headerBits)
        {
            int b = bits[startBit];
            int v = fields[startBit] switch
            {
                M => realMode >> b,
                D => ep.Shape >> b,
                RW => e[0].A.R >> b,
                RX => e[0].B.R >> b,
                RY => e[1].A.R >> b,
                RZ => e[1].B.R >> b,
                GW => e[0].A.G >> b,
                GX => e[0].B.G >> b,
                GY => e[1].A.G >> b,
                GZ => e[1].B.G >> b,
                BW => e[0].A.B >> b,
                BX => e[0].B.B >> b,
                BY => e[1].A.B >> b,
                BZ => e[1].B.B >> b,
                _ => 0,
            };
            block.SetBit(ref startBit, v & 0x01);
        }

        for (int i = 0; i < 16; ++i)
        {
            if (Bc67Tables.IsFixUpOffset(partitions, ep.Shape, i))
                block.SetBits(ref startBit, indexPrec - 1, indices[i]);
            else
                block.SetBits(ref startBit, indexPrec, indices[i]);
        }
    }

    private static void Refine(ref EncodeParams ep, ref Bits128 block)
    {
        ref readonly var info = ref Info[ep.Mode];
        int partitions = info.Partitions;
        bool transformed = info.Transformed;

        Span<float> orgErr = stackalloc float[2];
        Span<float> optErr = stackalloc float[2];
        Buffer2<IntEndPntPair> orgEndPts = default, optEndPts = default;
        Buffer16<int> orgIdx, optIdx;
        Unsafe.SkipInit(out orgIdx);
        Unsafe.SkipInit(out optIdx);

        QuantizeEndPts(ref ep, ref orgEndPts);
        AssignIndices(ref ep, ref orgEndPts, orgIdx, orgErr);
        SwapIndices(ref ep, ref orgEndPts, orgIdx);

        if (transformed) TransformForward(ref orgEndPts);
        if (EndPointsFit(ref ep, ref orgEndPts))
        {
            if (transformed) TransformInverse(ref orgEndPts, info.Prec00, ep.Signed);
            OptimizeEndPoints(ref ep, orgErr, ref orgEndPts, ref optEndPts);
            AssignIndices(ref ep, ref optEndPts, optIdx, optErr);
            SwapIndices(ref ep, ref optEndPts, optIdx);

            float orgTotErr = 0, optTotErr = 0;
            for (int p = 0; p <= partitions; ++p)
            {
                orgTotErr += orgErr[p];
                optTotErr += optErr[p];
            }

            if (transformed) TransformForward(ref optEndPts);
            if (EndPointsFit(ref ep, ref optEndPts) && optTotErr < orgTotErr && optTotErr < ep.BestErr)
            {
                ep.BestErr = optTotErr;
                EmitBlock(ref ep, ref optEndPts, optIdx, ref block);
            }
            else if (orgTotErr < ep.BestErr)
            {
                // either it stopped fitting when we optimized it, or there was no improvement
                // so go back to the unoptimized endpoints which we know will fit
                if (transformed) TransformForward(ref orgEndPts);
                ep.BestErr = orgTotErr;
                EmitBlock(ref ep, ref orgEndPts, orgIdx, ref block);
            }
        }
    }

    private static void GeneratePaletteUnquantized(ref EncodeParams ep, int region, Span<IntColor> palette)
    {
        ref readonly var endPts = ref ep.UnqEndPts[ep.Shape * 2 + region];
        int indexPrec = Info[ep.Mode].IndexPrec;
        int numIndices = 1 << indexPrec;
        var weights = Bc67Tables.Weights(indexPrec);

        for (int i = 0; i < numIndices; ++i)
        {
            int w = weights[i];
            int iw = Bc67Tables.WeightMax - w;
            palette[i].R = (endPts.A.R * iw + endPts.B.R * w + Bc67Tables.WeightRound) >> Bc67Tables.WeightShift;
            palette[i].G = (endPts.A.G * iw + endPts.B.G * w + Bc67Tables.WeightRound) >> Bc67Tables.WeightShift;
            palette[i].B = (endPts.A.B * iw + endPts.B.B * w + Bc67Tables.WeightRound) >> Bc67Tables.WeightShift;
        }
    }

    private static float MapColors(ref EncodeParams ep, int region, int np, ReadOnlySpan<int> pixIndex)
    {
        int numIndices = 1 << Info[ep.Mode].IndexPrec;
        Buffer16<IntColor> palette;
        Unsafe.SkipInit(out palette);
        GeneratePaletteUnquantized(ref ep, region, palette);

        float totalErr = 0.0f;
        for (int i = 0; i < np; ++i)
        {
            ref readonly var px = ref ep.IPixels[pixIndex[i]];
            float bestErr = Norm(px, palette[0]);
            for (int j = 1; j < numIndices && bestErr > 0.0f; ++j)
            {
                float err = Norm(px, palette[j]);
                if (err > bestErr) break; // error increased, so we're done searching
                if (err < bestErr) bestErr = err;
            }
            totalErr += bestErr;
        }
        return totalErr;
    }

    private static float RoughMse(ref EncodeParams ep)
    {
        int partitions = Info[ep.Mode].Partitions;
        Span<int> pixIdx = stackalloc int[16];

        float error = 0.0f;
        for (int p = 0; p <= partitions; ++p)
        {
            int np = 0;
            for (int i = 0; i < 16; ++i)
            {
                if (Bc67Tables.Partition(partitions, ep.Shape, i) == p) pixIdx[np++] = i;
            }

            ref var endPts = ref ep.UnqEndPts[ep.Shape * 2 + p];

            // handle simple cases
            if (np == 1)
            {
                endPts.A = ep.IPixels[pixIdx[0]];
                endPts.B = ep.IPixels[pixIdx[0]];
                continue;
            }
            if (np == 2)
            {
                endPts.A = ep.IPixels[pixIdx[0]];
                endPts.B = ep.IPixels[pixIdx[1]];
                continue;
            }

            HdrColorA epA = default, epB = default;
            Bc67Math.OptimizeRgb(ep.HdrPixels, ref epA, ref epB, 4, np, pixIdx);
            endPts.A.Set(epA, ep.Signed);
            endPts.B.Set(epB, ep.Signed);
            if (ep.Signed)
            {
                endPts.A.Clamp(-F16Max, F16Max);
                endPts.B.Clamp(-F16Max, F16Max);
            }
            else
            {
                endPts.A.Clamp(0, F16Max);
                endPts.B.Clamp(0, F16Max);
            }

            error += MapColors(ref ep, p, np, pixIdx);
        }

        return error;
    }

    // ms_aDesc[14][82] field (EField) and bit, flattened [mode * 82 + bit]
    internal static ReadOnlySpan<byte> DescField =>
    [
        1, 1, 9, 13, 14, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 4, 4, 4, 4, 4, 10,
        9, 9, 9, 9, 8, 8, 8, 8, 8, 14, 10, 10, 10, 10, 12, 12, 12, 12, 12, 14, 13, 13, 13, 13, 5, 5, 5, 5, 5, 14, 6, 6, 6, 6, 6, 14, 2, 2, 2, 2, 2,
        1, 1, 9, 10, 10, 3, 3, 3, 3, 3, 3, 3, 14, 14, 13, 7, 7, 7, 7, 7, 7, 7, 13, 14, 9, 11, 11, 11, 11, 11, 11, 11, 14, 14, 14, 4, 4, 4, 4, 4, 4,
        9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 4, 4, 4, 4, 4, 3,
        9, 9, 9, 9, 8, 8, 8, 8, 7, 14, 10, 10, 10, 10, 12, 12, 12, 12, 11, 14, 13, 13, 13, 13, 5, 5, 5, 5, 5, 14, 6, 6, 6, 6, 6, 14, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 4, 4, 4, 4, 3, 10,
        9, 9, 9, 9, 8, 8, 8, 8, 8, 7, 10, 10, 10, 10, 12, 12, 12, 12, 11, 14, 13, 13, 13, 13, 5, 5, 5, 5, 14, 14, 6, 6, 6, 6, 9, 14, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 4, 4, 4, 4, 3, 13,
        9, 9, 9, 9, 8, 8, 8, 8, 7, 14, 10, 10, 10, 10, 12, 12, 12, 12, 12, 11, 13, 13, 13, 13, 5, 5, 5, 5, 14, 14, 6, 6, 6, 6, 14, 14, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 13, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 14, 4, 4, 4, 4, 4, 10,
        9, 9, 9, 9, 8, 8, 8, 8, 8, 14, 10, 10, 10, 10, 12, 12, 12, 12, 12, 14, 13, 13, 13, 13, 5, 5, 5, 5, 5, 14, 6, 6, 6, 6, 6, 14, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 10, 13, 7, 7, 7, 7, 7, 7, 7, 7, 14, 9, 11, 11, 11, 11, 11, 11, 11, 11, 14, 14, 4, 4, 4, 4, 4, 4,
        9, 9, 9, 9, 8, 8, 8, 8, 8, 14, 10, 10, 10, 10, 12, 12, 12, 12, 12, 14, 13, 13, 13, 13, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 14, 13, 7, 7, 7, 7, 7, 7, 7, 7, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 10, 14, 4, 4, 4, 4, 4, 10,
        9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 12, 12, 12, 12, 12, 14, 13, 13, 13, 13, 5, 5, 5, 5, 5, 14, 6, 6, 6, 6, 6, 14, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 14, 13, 7, 7, 7, 7, 7, 7, 7, 7, 13, 9, 11, 11, 11, 11, 11, 11, 11, 11, 14, 14, 4, 4, 4, 4, 4, 10,
        9, 9, 9, 9, 8, 8, 8, 8, 8, 14, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 5, 5, 5, 5, 5, 14, 6, 6, 6, 6, 6, 14, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 10, 14, 14, 13, 7, 7, 7, 7, 7, 7, 9, 13, 14, 9, 11, 11, 11, 11, 11, 11, 10, 14, 14, 14, 4, 4, 4, 4, 4, 4,
        9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 4, 4, 4, 4, 4, 4,
        4, 4, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 4, 4, 4, 4, 3, 3,
        3, 3, 3, 3, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    internal static ReadOnlySpan<byte> DescBit =>
    [
        0, 1, 4, 4, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 4,
        0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 3, 0, 1, 2, 3, 4,
        0, 1, 5, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 4, 0, 1, 2, 3, 4, 5, 6, 5, 2, 4, 0, 1, 2, 3, 4, 5, 6, 3, 5, 4, 0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10,
        0, 1, 2, 3, 0, 1, 2, 3, 10, 0, 0, 1, 2, 3, 0, 1, 2, 3, 10, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 3, 0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 10, 4,
        0, 1, 2, 3, 0, 1, 2, 3, 4, 10, 0, 1, 2, 3, 0, 1, 2, 3, 10, 1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 0, 1, 2, 3, 4, 3, 0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 10, 4,
        0, 1, 2, 3, 0, 1, 2, 3, 10, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 10, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 0, 1, 2, 3, 4, 3, 0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 0, 1, 2, 3, 4, 4,
        0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 3, 0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 4, 4, 0, 1, 2, 3, 4, 5, 6, 7, 2, 4, 0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 0, 4, 0, 1, 2, 3, 4, 5, 6, 7, 5, 4, 0, 1, 2, 3, 4, 5, 6, 7, 5, 4, 0, 1, 2, 3, 4, 4,
        0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 3, 0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 1, 4, 0, 1, 2, 3, 4, 5, 6, 7, 5, 4, 0, 1, 2, 3, 4, 5, 6, 7, 5, 4, 0, 1, 2, 3, 4, 4,
        0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 3, 0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 4, 0, 1, 4, 0, 1, 2, 3, 4, 5, 5, 5, 2, 4, 0, 1, 2, 3, 4, 5, 5, 3, 5, 4, 0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5,
        6, 7, 8, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5,
        6, 7, 11, 10, 0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 15, 14,
        13, 12, 11, 10, 0, 1, 2, 3, 15, 14, 13, 12, 11, 10, 0, 1, 2, 3, 15, 14, 13, 12, 11, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
}
