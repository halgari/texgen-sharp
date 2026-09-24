//-------------------------------------------------------------------------------------
// Cpu/Bc7Cpu.cs
//
// DirectXTex CPU BC7 encoder (port of D3DX_BC7::Encode in BC6HBC7.cpp). For each
// enabled mode, rotation and index mode it scores every partition shape with a rough
// least-squares fit (RoughMSE), refines the best quarter (quantize, fix p-bits, assign
// indices, perturb and exhaustively search endpoints) and keeps the lowest-error block.
//
// Mode selection matches DirectXTex: modes 0/2 only with BcFlags.Use3Subsets, only mode 6
// with BcFlags.ForceBc7Mode6 (TEX_COMPRESS_BC7_QUICK), and mode 7 is skipped for fully
// opaque blocks. DirectXTex's arithmetic quirks are preserved for bit-exact parity with
// texconv (see the notes on Quantize and OptimizeOne).
//-------------------------------------------------------------------------------------

using System.Numerics;
using System.Runtime.CompilerServices;

namespace TexGen.Cpu;

[SkipLocalsInit]
internal static class Bc7Cpu
{
    private const int NumModes = 8;
    private const int MaxShapes = 64;
    private const int MaxRegions = 3;

    private readonly struct ModeInfo(byte partitions, byte partitionBits, byte pBits, byte rotationBits, byte indexModeBits,
        byte indexPrec, byte indexPrec2, LdrColorA rgbaPrec, LdrColorA rgbaPrecWithP)
    {
        public readonly byte Partitions = partitions;
        public readonly byte PartitionBits = partitionBits;
        public readonly byte PBits = pBits;
        public readonly byte RotationBits = rotationBits;
        public readonly byte IndexModeBits = indexModeBits;
        public readonly byte IndexPrec = indexPrec;
        public readonly byte IndexPrec2 = indexPrec2;
        public readonly LdrColorA RgbaPrec = rgbaPrec;
        public readonly LdrColorA RgbaPrecWithP = rgbaPrecWithP;
    }

    // uPartitions, uPartitionBits, uPBits, uRotationBits, uIndexModeBits, uIndexPrec, uIndexPrec2, RGBAPrec, RGBAPrecWithP
    private static readonly ModeInfo[] Info =
    [
        // Mode 0: Color only, 3 Subsets, RGBP 4441 (unique P-bit), 3-bit indices, 16 partitions
        new(2, 4, 6, 0, 0, 3, 0, new(4, 4, 4, 0), new(5, 5, 5, 0)),
        // Mode 1: Color only, 2 Subsets, RGBP 6661 (shared P-bit), 3-bit indices, 64 partitions
        new(1, 6, 2, 0, 0, 3, 0, new(6, 6, 6, 0), new(7, 7, 7, 0)),
        // Mode 2: Color only, 3 Subsets, RGB 555, 2-bit indices, 64 partitions
        new(2, 6, 0, 0, 0, 2, 0, new(5, 5, 5, 0), new(5, 5, 5, 0)),
        // Mode 3: Color only, 2 Subsets, RGBP 7771 (unique P-bit), 2-bits indices, 64 partitions
        new(1, 6, 4, 0, 0, 2, 0, new(7, 7, 7, 0), new(8, 8, 8, 0)),
        // Mode 4: Color w/ Separate Alpha, 1 Subset, RGB 555, A6, 16x2/16x3-bit indices, 2-bit rotation, 1-bit index selector
        new(0, 0, 0, 2, 1, 2, 3, new(5, 5, 5, 6), new(5, 5, 5, 6)),
        // Mode 5: Color w/ Separate Alpha, 1 Subset, RGB 777, A8, 16x2/16x2-bit indices, 2-bit rotation
        new(0, 0, 0, 2, 0, 2, 2, new(7, 7, 7, 8), new(7, 7, 7, 8)),
        // Mode 6: Color+Alpha, 1 Subset, RGBAP 77771 (unique P-bit), 16x4-bit indices
        new(0, 0, 2, 0, 0, 4, 0, new(7, 7, 7, 7), new(8, 8, 8, 8)),
        // Mode 7: Color+Alpha, 2 Subsets, RGBAP 55551 (unique P-bit), 2-bit indices, 64 partitions
        new(1, 6, 4, 0, 0, 2, 0, new(5, 5, 5, 5), new(6, 6, 6, 6)),
    ];

    [InlineArray(MaxShapes * MaxRegions)]
    private struct EndPtsArray
    {
        private LdrEndPntPair _element0;
    }

    [InlineArray(MaxRegions * 16)]
    private struct Palettes
    {
        private LdrColorA _element0;
    }

    private struct EncodeParams
    {
        public int Mode;
        /// <summary>aEndPts[shape][region] flattened to [shape * 3 + region].</summary>
        public EndPtsArray EndPts;
        public Buffer16<LdrColorA> LdrPixels;
        public Buffer16<HdrColorA> HdrPixels;
    }

    /// <summary>Encode 16 RGBA pixels in [0,1] (row-major) into one BC7 block.</summary>
    public static void Encode(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags)
    {
        EncodeParams ep;
        Unsafe.SkipInit(out ep);
        ep.Mode = 0;
        ep.EndPts = default;

        Bits128 current = default;
        Bits128 final = default;
        float mseBest = float.MaxValue;
        int alphaMask = 0xFF;

        for (int i = 0; i < 16; ++i)
        {
            var c = color[i];
            ep.HdrPixels[i] = HdrColorA.FromVector(c);
            ep.LdrPixels[i] = new LdrColorA(ToByte(c.X), ToByte(c.Y), ToByte(c.Z), ToByte(c.W));
            alphaMask &= ep.LdrPixels[i].A;
        }

        bool hasAlpha = alphaMask != 0xFF;
        Span<float> roughMse = stackalloc float[MaxShapes];
        Span<int> shapes = stackalloc int[MaxShapes];

        for (ep.Mode = 0; ep.Mode < NumModes && mseBest > 0; ++ep.Mode)
        {
            // 3 subset modes tend to be used rarely and add significant compression time
            if ((flags & BcFlags.Use3Subsets) == 0 && (ep.Mode == 0 || ep.Mode == 2)) continue;

            // Use only mode 6
            if ((flags & BcFlags.ForceBc7Mode6) != 0 && ep.Mode != 6) continue;

            // There is no value in using mode 7 for completely opaque blocks (the other 2 subset
            // modes handle this case for opaque blocks), so skip it for a small perf win.
            if (!hasAlpha && ep.Mode == 7) continue;

            ref readonly var info = ref Info[ep.Mode];
            int numShapes = 1 << info.PartitionBits;
            int numRots = 1 << info.RotationBits;
            int numIdxMode = 1 << info.IndexModeBits;
            // Number of rough cases to look at. reasonable values of this are 1, uShapes/4, and uShapes
            // uShapes/4 gets nearly all the cases; you can increase that a bit (say by 3 or 4) if you really want to squeeze the last bit out
            int items = Math.Max(1, numShapes >> 2);

            for (int r = 0; r < numRots && mseBest > 0; ++r)
            {
                SwapRotation(ref ep, r);

                for (int im = 0; im < numIdxMode && mseBest > 0; ++im)
                {
                    // pick the best uItems shapes and refine these.
                    for (int s = 0; s < numShapes; s++)
                    {
                        roughMse[s] = RoughMse(ref ep, s, im);
                        shapes[s] = s;
                    }

                    // Bubble up the first uItems items
                    for (int i = 0; i < items; i++)
                    {
                        for (int j = i + 1; j < numShapes; j++)
                        {
                            if (roughMse[i] > roughMse[j])
                            {
                                (roughMse[i], roughMse[j]) = (roughMse[j], roughMse[i]);
                                (shapes[i], shapes[j]) = (shapes[j], shapes[i]);
                            }
                        }
                    }

                    for (int i = 0; i < items && mseBest > 0; i++)
                    {
                        float mse = Refine(ref ep, shapes[i], r, im, ref current);
                        if (mse < mseBest)
                        {
                            final = current;
                            mseBest = mse;
                        }
                    }
                }

                SwapRotation(ref ep, r);
            }
        }

        ((ReadOnlySpan<byte>)final).CopyTo(bc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte ToByte(float v) => (byte)MathF.Max(0.0f, MathF.Min(255.0f, v * 255.0f + 0.01f));

    /// <summary>Rotation 1/2/3 swaps alpha with R/G/B (self-inverse).</summary>
    private static void SwapRotation(ref EncodeParams ep, int r)
    {
        if (r == 0) return;
        for (int i = 0; i < 16; i++)
        {
            ref var p = ref ep.LdrPixels[i];
            ref byte c = ref p[r - 1];
            (c, p.A) = (p.A, c);
        }
    }

    //---------------------------------------------------------------------------------
    // Quantization
    //---------------------------------------------------------------------------------

    /// <summary>
    /// D3DX_BC7::Quantize. Note DirectXTex casts comp + rounding to uint8 before the
    /// min(255, ...), so the rounding wraps for components near 255; preserved for parity
    /// (the endpoint optimizer recovers). For 8-bit precision the shift amount is -1,
    /// which on x86 leaves the component unchanged.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte Quantize(byte comp, int prec)
    {
        if (prec >= 8) return comp;
        byte rnd = (byte)(comp + (1 << (7 - prec)));
        return (byte)(rnd >> (8 - prec));
    }

    private static LdrColorA Quantize(in LdrColorA c, in LdrColorA prec) => new(
        Quantize(c.R, prec.R),
        Quantize(c.G, prec.G),
        Quantize(c.B, prec.B),
        prec.A != 0 ? Quantize(c.A, prec.A) : (byte)255);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte Unquantize(byte comp, int prec)
    {
        comp = (byte)(comp << (8 - prec));
        return (byte)(comp | (comp >> prec));
    }

    private static LdrColorA Unquantize(in LdrColorA c, in LdrColorA prec) => new(
        Unquantize(c.R, prec.R),
        Unquantize(c.G, prec.G),
        Unquantize(c.B, prec.B),
        prec.A > 0 ? Unquantize(c.A, prec.A) : (byte)255);

    //---------------------------------------------------------------------------------
    // Encoder steps
    //---------------------------------------------------------------------------------

    private static void GeneratePaletteQuantized(ref EncodeParams ep, int indexMode, in LdrEndPntPair endPts, Span<LdrColorA> palette)
    {
        ref readonly var info = ref Info[ep.Mode];
        int indexPrec = indexMode != 0 ? info.IndexPrec2 : info.IndexPrec;
        int indexPrec2 = indexMode != 0 ? info.IndexPrec : info.IndexPrec2;
        int numIndices = 1 << indexPrec;
        int numIndices2 = 1 << indexPrec2;

        var a = Unquantize(endPts.A, info.RgbaPrecWithP);
        var b = Unquantize(endPts.B, info.RgbaPrecWithP);
        if (indexPrec2 == 0)
        {
            for (int i = 0; i < numIndices; i++) LdrColorA.Interpolate(a, b, i, i, indexPrec, indexPrec, ref palette[i]);
        }
        else
        {
            for (int i = 0; i < numIndices; i++) LdrColorA.InterpolateRgb(a, b, i, indexPrec, ref palette[i]);
            for (int i = 0; i < numIndices2; i++) LdrColorA.InterpolateA(a, b, i, indexPrec2, ref palette[i]);
        }
    }

    private static float PerturbOne(ref EncodeParams ep, ReadOnlySpan<LdrColorA> colors, int np, int indexMode, int ch,
        in LdrEndPntPair oldEndPts, out LdrEndPntPair newEndPts, float oldErr, int doB)
    {
        int prec = Info[ep.Mode].RgbaPrecWithP[ch];
        newEndPts = oldEndPts;
        LdrEndPntPair tmpEndPts = oldEndPts;
        float minErr = oldErr;
        ref byte newC = ref doB != 0 ? ref newEndPts.B[ch] : ref newEndPts.A[ch];
        ref byte tmpC = ref doB != 0 ? ref tmpEndPts.B[ch] : ref tmpEndPts.A[ch];

        // do a logarithmic search for the best error for this endpoint (which)
        for (int step = 1 << (prec - 1); step != 0; step >>= 1)
        {
            bool improved = false;
            int bestStep = 0;
            for (int sign = -1; sign <= 1; sign += 2)
            {
                int tmp = newC + sign * step;
                if (tmp < 0 || tmp >= (1 << prec)) continue;
                tmpC = (byte)tmp;

                float totalErr = MapColors(ref ep, colors, np, indexMode, tmpEndPts, minErr);
                if (totalErr < minErr)
                {
                    improved = true;
                    minErr = totalErr;
                    bestStep = sign * step;
                }
            }

            // if this was an improvement, move the endpoint and continue search from there
            if (improved) newC = (byte)(newC + bestStep);
        }
        return minErr;
    }

    /// <summary>Perturb the endpoints at least -3 to 3, preserving endpoint ordering.</summary>
    private static void Exhaustive(ref EncodeParams ep, ReadOnlySpan<LdrColorA> colors, int np, int indexMode, int ch,
        ref float orgErr, ref LdrEndPntPair optEndPt)
    {
        int prec = Info[ep.Mode].RgbaPrecWithP[ch];
        if (orgErr == 0) return;

        const int delta = 5;

        // ok figure out the range of A and B
        LdrEndPntPair tmpEndPt = optEndPt;
        int alow = Math.Max(0, optEndPt.A[ch] - delta);
        int ahigh = Math.Min((1 << prec) - 1, optEndPt.A[ch] + delta);
        int blow = Math.Max(0, optEndPt.B[ch] - delta);
        int bhigh = Math.Min((1 << prec) - 1, optEndPt.B[ch] + delta);
        int amin = 0;
        int bmin = 0;

        float bestErr = orgErr;
        if (optEndPt.A[ch] <= optEndPt.B[ch])
        {
            // keep a <= b
            for (int a = alow; a <= ahigh; ++a)
            {
                for (int b = Math.Max(a, blow); b < bhigh; ++b)
                {
                    tmpEndPt.A[ch] = (byte)a;
                    tmpEndPt.B[ch] = (byte)b;

                    float err = MapColors(ref ep, colors, np, indexMode, tmpEndPt, bestErr);
                    if (err < bestErr)
                    {
                        amin = a;
                        bmin = b;
                        bestErr = err;
                    }
                }
            }
        }
        else
        {
            // keep b <= a
            for (int b = blow; b < bhigh; ++b)
            {
                for (int a = Math.Max(b, alow); a <= ahigh; ++a)
                {
                    tmpEndPt.A[ch] = (byte)a;
                    tmpEndPt.B[ch] = (byte)b;

                    float err = MapColors(ref ep, colors, np, indexMode, tmpEndPt, bestErr);
                    if (err < bestErr)
                    {
                        amin = a;
                        bmin = b;
                        bestErr = err;
                    }
                }
            }
        }

        if (bestErr < orgErr)
        {
            optEndPt.A[ch] = (byte)amin;
            optEndPt.B[ch] = (byte)bmin;
            orgErr = bestErr;
        }
    }

    /// <summary>
    /// D3DX_BC7::OptimizeOne. DirectXTex binds its "new B" reference to new_a.B rather than
    /// new_b.B, so moves of endpoint B come from the A-perturbation result; this is kept
    /// as-is for bit-exact parity with texconv.
    /// </summary>
    private static void OptimizeOne(ref EncodeParams ep, ReadOnlySpan<LdrColorA> colors, int np, int indexMode,
        float orgErr, in LdrEndPntPair org, out LdrEndPntPair opt)
    {
        float optErr = orgErr;
        opt = org;

        // now optimize each channel separately
        for (int ch = 0; ch < 4; ++ch)
        {
            if (Info[ep.Mode].RgbaPrecWithP[ch] == 0) continue;

            // figure out which endpoint when perturbed gives the most improvement and start there
            // if we just alternate, we can easily end up in a local minima
            float err0 = PerturbOne(ref ep, colors, np, indexMode, ch, opt, out var newA, optErr, 0); // perturb endpt A
            float err1 = PerturbOne(ref ep, colors, np, indexMode, ch, opt, out _, optErr, 1); // perturb endpt B

            byte cnewA = newA.A[ch];
            byte cnewB = newA.B[ch];

            int doB;
            if (err0 < err1)
            {
                if (err0 >= optErr) continue;
                opt.A[ch] = cnewA;
                optErr = err0;
                doB = 1; // do B next
            }
            else
            {
                if (err1 >= optErr) continue;
                opt.B[ch] = cnewB;
                optErr = err1;
                doB = 0; // do A next
            }

            // now alternate endpoints and keep trying until there is no improvement
            for (;;)
            {
                float err = PerturbOne(ref ep, colors, np, indexMode, ch, opt, out _, optErr, doB);
                if (err >= optErr) break;
                if (doB == 0) opt.A[ch] = cnewA;
                else opt.B[ch] = cnewB;
                optErr = err;
                doB = 1 - doB; // now move the other endpoint
            }
        }

        // finally, do a small exhaustive search around what we think is the global minima to be sure
        for (int ch = 0; ch < 4; ch++) Exhaustive(ref ep, colors, np, indexMode, ch, ref optErr, ref opt);
    }

    private static void OptimizeEndPoints(ref EncodeParams ep, int shape, int indexMode, ReadOnlySpan<float> orgErr,
        ref Buffer3<LdrEndPntPair> orgEndPts, ref Buffer3<LdrEndPntPair> optEndPts)
    {
        int partitions = Info[ep.Mode].Partitions;
        Buffer16<LdrColorA> pixels;
        Unsafe.SkipInit(out pixels);

        for (int p = 0; p <= partitions; ++p)
        {
            // collect the pixels in the region
            int np = 0;
            for (int i = 0; i < 16; ++i)
            {
                if (Bc67Tables.Partition(partitions, shape, i) == p) pixels[np++] = ep.LdrPixels[i];
            }

            OptimizeOne(ref ep, pixels, np, indexMode, orgErr[p], orgEndPts[p], out optEndPts[p]);
        }
    }

    private static void AssignIndices(ref EncodeParams ep, int shape, int indexMode, ref Buffer3<LdrEndPntPair> endPts,
        Span<int> indices, Span<int> indices2, Span<float> totErr)
    {
        ref readonly var info = ref Info[ep.Mode];
        int partitions = info.Partitions;
        int indexPrec = indexMode != 0 ? info.IndexPrec2 : info.IndexPrec;
        int indexPrec2 = indexMode != 0 ? info.IndexPrec : info.IndexPrec2;
        int numIndices = 1 << indexPrec;
        int numIndices2 = 1 << indexPrec2;
        int highestIndexBit = numIndices >> 1;
        int highestIndexBit2 = numIndices2 >> 1;

        Palettes palettes;
        Unsafe.SkipInit(out palettes);
        Span<LdrColorA> pal = palettes;

        // build list of possibles
        for (int p = 0; p <= partitions; p++)
        {
            GeneratePaletteQuantized(ref ep, indexMode, endPts[p], pal.Slice(p * 16, 16));
            totErr[p] = 0;
        }

        for (int i = 0; i < 16; i++)
        {
            int region = Bc67Tables.Partition(partitions, shape, i);
            totErr[region] += Bc67Math.ComputeError(ep.LdrPixels[i], pal.Slice(region * 16, 16), indexPrec, indexPrec2,
                out indices[i], out indices2[i]);
        }

        // swap endpoints as needed to ensure that the indices at index_positions have a 0 high-order bit
        if (indexPrec2 == 0)
        {
            for (int p = 0; p <= partitions; p++)
            {
                if ((indices[Bc67Tables.FixUpIndex(partitions, shape, p)] & highestIndexBit) != 0)
                {
                    (endPts[p].A, endPts[p].B) = (endPts[p].B, endPts[p].A);
                    for (int i = 0; i < 16; i++)
                    {
                        if (Bc67Tables.Partition(partitions, shape, i) == p) indices[i] = numIndices - 1 - indices[i];
                    }
                }
            }
        }
        else
        {
            for (int p = 0; p <= partitions; p++)
            {
                if ((indices[Bc67Tables.FixUpIndex(partitions, shape, p)] & highestIndexBit) != 0)
                {
                    ref var e = ref endPts[p];
                    (e.A.R, e.B.R) = (e.B.R, e.A.R);
                    (e.A.G, e.B.G) = (e.B.G, e.A.G);
                    (e.A.B, e.B.B) = (e.B.B, e.A.B);
                    for (int i = 0; i < 16; i++)
                    {
                        if (Bc67Tables.Partition(partitions, shape, i) == p) indices[i] = numIndices - 1 - indices[i];
                    }
                }

                if ((indices2[0] & highestIndexBit2) != 0)
                {
                    ref var e = ref endPts[p];
                    (e.A.A, e.B.A) = (e.B.A, e.A.A);
                    for (int i = 0; i < 16; i++) indices2[i] = numIndices2 - 1 - indices2[i];
                }
            }
        }
    }

    private static void EmitBlock(ref EncodeParams ep, int shape, int rotation, int indexMode,
        ref Buffer3<LdrEndPntPair> endPts, ReadOnlySpan<int> index, ReadOnlySpan<int> index2, ref Bits128 block)
    {
        ref readonly var info = ref Info[ep.Mode];
        int partitions = info.Partitions;
        int pBits = info.PBits;
        int indexPrec = info.IndexPrec;
        int indexPrec2 = info.IndexPrec2;
        var rgbaPrec = info.RgbaPrec;
        var rgbaPrecWithP = info.RgbaPrecWithP;
        int startBit = 0;
        block.SetBits(ref startBit, ep.Mode, 0);
        block.SetBits(ref startBit, 1, 1);
        block.SetBits(ref startBit, info.RotationBits, rotation);
        block.SetBits(ref startBit, info.IndexModeBits, indexMode);
        block.SetBits(ref startBit, info.PartitionBits, shape);

        if (pBits != 0)
        {
            int numEP = (partitions + 1) << 1;
            Span<int> pVote = stackalloc int[MaxRegions << 1];
            Span<int> count = stackalloc int[MaxRegions << 1];
            pVote.Clear();
            count.Clear();
            for (int ch = 0; ch < 4; ch++)
            {
                int epIdx = 0;
                for (int i = 0; i <= partitions; i++)
                {
                    if (rgbaPrec[ch] == rgbaPrecWithP[ch])
                    {
                        block.SetBits(ref startBit, rgbaPrec[ch], endPts[i].A[ch]);
                        block.SetBits(ref startBit, rgbaPrec[ch], endPts[i].B[ch]);
                    }
                    else
                    {
                        block.SetBits(ref startBit, rgbaPrec[ch], endPts[i].A[ch] >> 1);
                        block.SetBits(ref startBit, rgbaPrec[ch], endPts[i].B[ch] >> 1);
                        int idx = epIdx++ * pBits / numEP;
                        pVote[idx] += endPts[i].A[ch] & 0x01;
                        count[idx]++;
                        idx = epIdx++ * pBits / numEP;
                        pVote[idx] += endPts[i].B[ch] & 0x01;
                        count[idx]++;
                    }
                }
            }

            for (int i = 0; i < pBits; i++) block.SetBits(ref startBit, 1, pVote[i] > (count[i] >> 1) ? 1 : 0);
        }
        else
        {
            for (int ch = 0; ch < 4; ch++)
            {
                for (int i = 0; i <= partitions; i++)
                {
                    block.SetBits(ref startBit, rgbaPrec[ch], endPts[i].A[ch]);
                    block.SetBits(ref startBit, rgbaPrec[ch], endPts[i].B[ch]);
                }
            }
        }

        var i1 = indexMode != 0 ? index2 : index;
        var i2 = indexMode != 0 ? index : index2;
        for (int i = 0; i < 16; i++)
        {
            if (Bc67Tables.IsFixUpOffset(partitions, shape, i))
                block.SetBits(ref startBit, indexPrec - 1, i1[i]);
            else
                block.SetBits(ref startBit, indexPrec, i1[i]);
        }
        if (indexPrec2 != 0)
        {
            for (int i = 0; i < 16; i++) block.SetBits(ref startBit, i != 0 ? indexPrec2 : indexPrec2 - 1, i2[i]);
        }
    }

    private static void FixEndpointPBits(ref EncodeParams ep, ref Buffer3<LdrEndPntPair> orig, ref Buffer3<LdrEndPntPair> fixedEp)
    {
        ref readonly var info = ref Info[ep.Mode];
        int partitions = info.Partitions;

        fixedEp[0] = orig[0];
        fixedEp[1] = orig[1];
        fixedEp[2] = orig[2];

        int pBits = info.PBits;
        if (pBits == 0) return;

        int numEP = (1 + partitions) << 1;
        Span<int> pVote = stackalloc int[MaxRegions << 1];
        Span<int> count = stackalloc int[MaxRegions << 1];
        pVote.Clear();
        count.Clear();

        var rgbaPrec = info.RgbaPrec;
        var rgbaPrecWithP = info.RgbaPrecWithP;

        for (int ch = 0; ch < 4; ch++)
        {
            int epIdx = 0;
            for (int i = 0; i <= partitions; i++)
            {
                if (rgbaPrec[ch] == rgbaPrecWithP[ch])
                {
                    fixedEp[i].A[ch] = orig[i].A[ch];
                    fixedEp[i].B[ch] = orig[i].B[ch];
                }
                else
                {
                    fixedEp[i].A[ch] = (byte)(orig[i].A[ch] >> 1);
                    fixedEp[i].B[ch] = (byte)(orig[i].B[ch] >> 1);

                    int idx = epIdx++ * pBits / numEP;
                    pVote[idx] += orig[i].A[ch] & 0x01;
                    count[idx]++;
                    idx = epIdx++ * pBits / numEP;
                    pVote[idx] += orig[i].B[ch] & 0x01;
                    count[idx]++;
                }
            }
        }

        // Compute the actual pbits we'll use when we encode block. Note this is not
        // rounding the component indices correctly in cases the pbits != a component's LSB.
        Span<int> pbits = stackalloc int[MaxRegions << 1];
        for (int i = 0; i < pBits; i++) pbits[i] = pVote[i] > (count[i] >> 1) ? 1 : 0;

        // Now calculate the actual endpoints with proper pbits, so error calculations are accurate.
        if (ep.Mode == 1)
        {
            // shared pbits
            for (int ch = 0; ch < 4; ch++)
            {
                for (int i = 0; i <= partitions; i++)
                {
                    fixedEp[i].A[ch] = (byte)((fixedEp[i].A[ch] << 1) | pbits[i]);
                    fixedEp[i].B[ch] = (byte)((fixedEp[i].B[ch] << 1) | pbits[i]);
                }
            }
        }
        else
        {
            for (int ch = 0; ch < 4; ch++)
            {
                for (int i = 0; i <= partitions; i++)
                {
                    fixedEp[i].A[ch] = (byte)((fixedEp[i].A[ch] << 1) | pbits[i * 2 + 0]);
                    fixedEp[i].B[ch] = (byte)((fixedEp[i].B[ch] << 1) | pbits[i * 2 + 1]);
                }
            }
        }
    }

    private static float Refine(ref EncodeParams ep, int shape, int rotation, int indexMode, ref Bits128 block)
    {
        ref readonly var info = ref Info[ep.Mode];
        int partitions = info.Partitions;

        Buffer3<LdrEndPntPair> orgEndPts = default, optEndPts = default, newEndPts1 = default, newEndPts2 = default;
        Buffer16<int> orgIdx, orgIdx2, optIdx, optIdx2;
        Unsafe.SkipInit(out orgIdx);
        Unsafe.SkipInit(out orgIdx2);
        Unsafe.SkipInit(out optIdx);
        Unsafe.SkipInit(out optIdx2);
        Span<float> orgErr = stackalloc float[MaxRegions];
        Span<float> optErr = stackalloc float[MaxRegions];

        for (int p = 0; p <= partitions; p++)
        {
            ref readonly var e = ref ep.EndPts[shape * MaxRegions + p];
            orgEndPts[p].A = Quantize(e.A, info.RgbaPrecWithP);
            orgEndPts[p].B = Quantize(e.B, info.RgbaPrecWithP);
        }

        FixEndpointPBits(ref ep, ref orgEndPts, ref newEndPts1);
        AssignIndices(ref ep, shape, indexMode, ref newEndPts1, orgIdx, orgIdx2, orgErr);
        OptimizeEndPoints(ref ep, shape, indexMode, orgErr, ref newEndPts1, ref optEndPts);
        FixEndpointPBits(ref ep, ref optEndPts, ref newEndPts2);
        AssignIndices(ref ep, shape, indexMode, ref newEndPts2, optIdx, optIdx2, optErr);

        float orgTotErr = 0, optTotErr = 0;
        for (int p = 0; p <= partitions; p++)
        {
            orgTotErr += orgErr[p];
            optTotErr += optErr[p];
        }

        if (optTotErr < orgTotErr)
        {
            EmitBlock(ref ep, shape, rotation, indexMode, ref newEndPts2, optIdx, optIdx2, ref block);
            return optTotErr;
        }
        EmitBlock(ref ep, shape, rotation, indexMode, ref newEndPts1, orgIdx, orgIdx2, ref block);
        return orgTotErr;
    }

    private static float MapColors(ref EncodeParams ep, ReadOnlySpan<LdrColorA> colors, int np, int indexMode,
        in LdrEndPntPair endPts, float minErr)
    {
        ref readonly var info = ref Info[ep.Mode];
        int indexPrec = indexMode != 0 ? info.IndexPrec2 : info.IndexPrec;
        int indexPrec2 = indexMode != 0 ? info.IndexPrec : info.IndexPrec2;
        Buffer16<LdrColorA> palette;
        Unsafe.SkipInit(out palette);
        float totalErr = 0;

        GeneratePaletteQuantized(ref ep, indexMode, endPts, palette);
        for (int i = 0; i < np; ++i)
        {
            totalErr += Bc67Math.ComputeError(colors[i], palette, indexPrec, indexPrec2, out _, out _);
            if (totalErr > minErr) // check for early exit
            {
                totalErr = float.MaxValue;
                break;
            }
        }
        return totalErr;
    }

    private static float RoughMse(ref EncodeParams ep, int shape, int indexMode)
    {
        ref readonly var info = ref Info[ep.Mode];
        int partitions = info.Partitions;
        int indexPrec = indexMode != 0 ? info.IndexPrec2 : info.IndexPrec;
        int indexPrec2 = indexMode != 0 ? info.IndexPrec : info.IndexPrec2;
        int numIndices = 1 << indexPrec;
        int numIndices2 = 1 << indexPrec2;
        Span<int> pixIdx = stackalloc int[16];
        Palettes palettes;
        Unsafe.SkipInit(out palettes);
        Span<LdrColorA> pal = palettes;

        for (int p = 0; p <= partitions; p++)
        {
            int np = 0;
            for (int i = 0; i < 16; i++)
            {
                if (Bc67Tables.Partition(partitions, shape, i) == p) pixIdx[np++] = i;
            }

            ref var e = ref ep.EndPts[shape * MaxRegions + p];

            // handle simple cases
            if (np == 1)
            {
                e.A = ep.LdrPixels[pixIdx[0]];
                e.B = ep.LdrPixels[pixIdx[0]];
                continue;
            }
            if (np == 2)
            {
                e.A = ep.LdrPixels[pixIdx[0]];
                e.B = ep.LdrPixels[pixIdx[1]];
                continue;
            }

            if (indexPrec2 == 0)
            {
                Bc67Math.OptimizeRgba(ep.HdrPixels, out var epA, out var epB, 4, np, pixIdx);
                epA = epA.Clamp(0.0f, 1.0f) * 255.0f;
                epB = epB.Clamp(0.0f, 1.0f) * 255.0f;
                e.A = LdrColorA.FromScaled(epA);
                e.B = LdrColorA.FromScaled(epB);
            }
            else
            {
                byte minAlpha = 255, maxAlpha = 0;
                for (int i = 0; i < 16; ++i)
                {
                    minAlpha = Math.Min(minAlpha, ep.LdrPixels[pixIdx[i]].A);
                    maxAlpha = Math.Max(maxAlpha, ep.LdrPixels[pixIdx[i]].A);
                }

                HdrColorA epA = default, epB = default;
                Bc67Math.OptimizeRgb(ep.HdrPixels, ref epA, ref epB, 4, np, pixIdx);
                epA = epA.Clamp(0.0f, 1.0f) * 255.0f;
                epB = epB.Clamp(0.0f, 1.0f) * 255.0f;
                e.A = LdrColorA.FromScaled(epA);
                e.B = LdrColorA.FromScaled(epB);
                e.A.A = minAlpha;
                e.B.A = maxAlpha;
            }
        }

        if (indexPrec2 == 0)
        {
            for (int p = 0; p <= partitions; p++)
            {
                ref readonly var e = ref ep.EndPts[shape * MaxRegions + p];
                for (int i = 0; i < numIndices; i++)
                    LdrColorA.Interpolate(e.A, e.B, i, i, indexPrec, indexPrec, ref pal[p * 16 + i]);
            }
        }
        else
        {
            for (int p = 0; p <= partitions; p++)
            {
                ref readonly var e = ref ep.EndPts[shape * MaxRegions + p];
                for (int i = 0; i < numIndices; i++) LdrColorA.InterpolateRgb(e.A, e.B, i, indexPrec, ref pal[p * 16 + i]);
                for (int i = 0; i < numIndices2; i++) LdrColorA.InterpolateA(e.A, e.B, i, indexPrec2, ref pal[p * 16 + i]);
            }
        }

        float totalErr = 0;
        for (int i = 0; i < 16; i++)
        {
            int region = Bc67Tables.Partition(partitions, shape, i);
            totalErr += Bc67Math.ComputeError(ep.LdrPixels[i], pal.Slice(region * 16, 16), indexPrec, indexPrec2, out _, out _);
        }
        return totalErr;
    }
}
