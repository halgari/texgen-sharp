//-------------------------------------------------------------------------------------
// Cpu/Bc67Simd.cs
//
// SIMD versions of the innermost BC6H/BC7 CPU-encoder loop: "for every pixel of a
// region, scan the palette for the closest entry and add up the errors". The endpoint
// searches (PerturbOne / Exhaustive / OptimizeOne) call this hundreds of thousands of
// times per block, so it dominates encode time.
//
// All 16 pixels of a block are scored at once, one pixel per lane. Each lane keeps
// its own best error and an "active" flag, which reproduces the scalar loop exactly:
//   for (i = 0; i < n && best > 0; i++) { if (err > best) break; if (err < best) best = err; }
// so the output stays bit-identical to DirectXTex:
//   * BC7 errors are integer-valued (byte pixels and palettes, max 16*4*255^2 < 2^24), so
//     integer lanes give the exact float result regardless of summation order.
//   * BC6H errors are rounded floats; each lane performs the same float operations in
//     the same order as the scalar Norm(), and the per-pixel minima are summed in pixel
//     order on the scalar side.
//
// Width is picked once: Vector512 (AVX-512) -> Vector<T> (AVX2 256-bit, SSE/NEON
// 128-bit) -> scalar. TEXGEN_SIMD=scalar|vector|512 forces a lower level for testing.
// BC7 uses Vector<T> chunks at both SIMD levels (its early exit favors 8-pixel chunks);
// BC6H scores all 16 pixels in one Vector512 when available.
//-------------------------------------------------------------------------------------

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace TexGen.Cpu;

public enum SimdLevel
{
    Scalar,
    /// <summary>System.Numerics.Vector&lt;T&gt;: 256-bit on AVX2, 128-bit on SSE/NEON.</summary>
    Vector,
    /// <summary>Vector512 (AVX-512): all 16 pixels of a block in one register.</summary>
    Vector512,
}

[SkipLocalsInit]
internal static class Bc67Simd
{
    /// <summary>Best SIMD level this machine supports.</summary>
    public static SimdLevel Hardware { get; } =
        Vector512.IsHardwareAccelerated ? SimdLevel.Vector512
        : Vector.IsHardwareAccelerated ? SimdLevel.Vector
        : SimdLevel.Scalar;

    /// <summary>Level in use: the hardware level, optionally lowered by TEXGEN_SIMD.</summary>
    public static SimdLevel Level { get; } = Resolve(Environment.GetEnvironmentVariable("TEXGEN_SIMD"));

    private static SimdLevel Resolve(string? env)
    {
        SimdLevel requested = env?.Trim().ToLowerInvariant() switch
        {
            "scalar" or "0" or "off" => SimdLevel.Scalar,
            "vector" or "128" or "256" => SimdLevel.Vector,
            "512" or "vector512" => SimdLevel.Vector512,
            _ => SimdLevel.Vector512,
        };
        return (SimdLevel)Math.Min((int)requested, (int)Hardware);
    }

    //---------------------------------------------------------------------------------
    // BC7: LDR (byte) pixels, integer error metric
    //---------------------------------------------------------------------------------

    /// <summary>
    /// BC7 MapColors: total best-palette error of <paramref name="np"/> pixels against the
    /// palette interpolated between the unquantized endpoints <paramref name="a"/> and
    /// <paramref name="b"/>, or float.MaxValue once the total exceeds <paramref name="minErr"/>.
    /// The RGBA metric is used over 2^indexPrec entries, or (indexPrec2 != 0) RGB over
    /// 2^indexPrec plus alpha over 2^indexPrec2. Palette entries are computed on the fly.
    /// </summary>
    public static float LdrMapColors(ReadOnlySpan<LdrColorA> colors, int np, in LdrColorA a, in LdrColorA b,
        int indexPrec, int indexPrec2, float minErr, SimdLevel level)
    {
        if (np <= 0) return 0;
        int total;
        switch (level)
        {
            // Both SIMD levels use Vector<T> chunks (8 pixels on AVX2/AVX-512 machines): stopping
            // after a chunk once the running total passes minErr beats one 16-lane pass here
            // (measured: BC7 1024^2 4.9 s vs 5.2 s on a Zen 5 with AVX-512).
            case SimdLevel.Vector512 or SimdLevel.Vector when Vector.IsHardwareAccelerated:
                // Integer sums: sum > minErr <=> sum > floor(minErr). Clamp before converting
                // (float)int.MaxValue rounds up to 2^31, which would overflow.
                total = LdrErrorSumVector(colors, np, a, b, indexPrec, indexPrec2,
                    minErr >= int.MaxValue ? int.MaxValue : (int)minErr);
                break;
            default:
                return LdrMapColorsScalar(colors, np, a, b, indexPrec, indexPrec2, minErr);
        }
        // Per-pixel errors are non-negative integers, so "some prefix sum exceeds minErr"
        // (DirectXTex's per-pixel early exit) is the same as "the total exceeds minErr".
        return total > minErr ? float.MaxValue : total;
    }

    /// <summary>The original DirectXTex loop (GeneratePaletteQuantized + ComputeError).</summary>
    private static float LdrMapColorsScalar(ReadOnlySpan<LdrColorA> colors, int np, in LdrColorA a, in LdrColorA b,
        int indexPrec, int indexPrec2, float minErr)
    {
        Span<LdrColorA> palette = stackalloc LdrColorA[16];
        if (indexPrec2 == 0)
        {
            for (int i = 0; i < 1 << indexPrec; i++) LdrColorA.Interpolate(a, b, i, i, indexPrec, indexPrec, ref palette[i]);
        }
        else
        {
            for (int i = 0; i < 1 << indexPrec; i++) LdrColorA.InterpolateRgb(a, b, i, indexPrec, ref palette[i]);
            for (int i = 0; i < 1 << indexPrec2; i++) LdrColorA.InterpolateA(a, b, i, indexPrec2, ref palette[i]);
        }
        float totalErr = 0;
        for (int i = 0; i < np; ++i)
        {
            totalErr += Bc67Math.ComputeError(colors[i], palette, indexPrec, indexPrec2, out _, out _);
            if (totalErr > minErr) return float.MaxValue; // check for early exit
        }
        return totalErr;
    }

    /// <summary>LdrColorA::Interpolate's per-channel lerp (BC6H/BC7 6-bit weights).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Lerp(int c0, int c1, int w) => (c0 * (64 - w) + c1 * w + 32) >> 6;

    /// <summary>Packed RGBA ints for 16 lanes: read in place when the span has 16 entries.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ReadOnlySpan<int> Packed16(ReadOnlySpan<LdrColorA> colors, int np, Span<int> scratch)
    {
        if (colors.Length >= 16) return MemoryMarshal.Cast<LdrColorA, int>(colors[..16]);
        MemoryMarshal.Cast<LdrColorA, int>(colors[..np]).CopyTo(scratch); // lanes >= np are masked off
        return scratch;
    }

    /// <summary>
    /// Vector&lt;T&gt; version (4 or 8 pixels per chunk). Chunks run in pixel order and stop once
    /// the running total passes <paramref name="minErr"/>, like the scalar early exit.
    /// </summary>
    private static int LdrErrorSumVector(ReadOnlySpan<LdrColorA> colors, int np, in LdrColorA ea, in LdrColorA eb,
        int indexPrec, int indexPrec2, int minErr)
    {
        Span<int> scratch = stackalloc int[16];
        var packed = Packed16(colors, np, scratch);
        int width = Vector<int>.Count;
        var mask = new Vector<int>(0xFF);
        var npv = new Vector<int>(np);
        var w1 = Bc67Tables.Weights(indexPrec);
        var w2 = Bc67Tables.Weights(indexPrec2);
        bool rgba = indexPrec2 == 0;
        int sum = 0;
        for (int c = 0; c < np; c += width)
        {
            var v = new Vector<int>(packed[c..]);
            var r = v & mask;
            var g = Vector.ShiftRightLogical(v, 8) & mask;
            var b = Vector.ShiftRightLogical(v, 16) & mask;
            var a = Vector.ShiftRightLogical(v, 24);
            var valid = Vector.LessThan(Vector<int>.Indices + new Vector<int>(c), npv);

            var best = new Vector<int>(int.MaxValue);
            var active = valid;
            for (int i = 0; i < w1.Length; i++)
            {
                active &= Vector.GreaterThan(best, Vector<int>.Zero);
                if (active == Vector<int>.Zero) break;
                int w = w1[i];
                var dr = r - new Vector<int>(Lerp(ea.R, eb.R, w));
                var dg = g - new Vector<int>(Lerp(ea.G, eb.G, w));
                var db = b - new Vector<int>(Lerp(ea.B, eb.B, w));
                var err = dr * dr + dg * dg + db * db;
                if (rgba)
                {
                    var da = a - new Vector<int>(Lerp(ea.A, eb.A, w));
                    err += da * da;
                }
                var brk = active & Vector.GreaterThan(err, best);
                var upd = active & Vector.LessThan(err, best);
                best = Vector.ConditionalSelect(upd, err, best);
                active = Vector.AndNot(active, brk);
            }
            sum += Vector.Sum(best & valid);

            if (!rgba)
            {
                best = new Vector<int>(int.MaxValue);
                active = valid;
                for (int i = 0; i < w2.Length; i++)
                {
                    active &= Vector.GreaterThan(best, Vector<int>.Zero);
                    if (active == Vector<int>.Zero) break;
                    var da = a - new Vector<int>(Lerp(ea.A, eb.A, w2[i]));
                    var err = da * da;
                    var brk = active & Vector.GreaterThan(err, best);
                    var upd = active & Vector.LessThan(err, best);
                    best = Vector.ConditionalSelect(upd, err, best);
                    active = Vector.AndNot(active, brk);
                }
                sum += Vector.Sum(best & valid);
            }
            if (sum > minErr) break; // the caller maps any total above minErr to MaxValue
        }
        return sum;
    }

    //---------------------------------------------------------------------------------
    // BC6H: integer (half-bit-pattern) pixels, float error metric
    //---------------------------------------------------------------------------------

    /// <summary>
    /// BC6H MapColorsQuantized error: for each pixel the best of <paramref name="numIndices"/>
    /// palette entries under Norm() (float RGB squared distance), summed in pixel order.
    /// </summary>
    public static float HdrErrorSum(ReadOnlySpan<IntColor> colors, int np, ReadOnlySpan<IntColor> palette,
        int numIndices, SimdLevel level)
    {
        if (np <= 0) return 0;
        Span<float> best = stackalloc float[16];
        switch (level)
        {
            case SimdLevel.Vector512 when Vector512.IsHardwareAccelerated:
                HdrBest512(colors, np, palette, numIndices, best);
                break;
            case SimdLevel.Vector512 or SimdLevel.Vector when Vector.IsHardwareAccelerated:
                HdrBestVector(colors, np, palette, numIndices, best);
                break;
            default:
                for (int i = 0; i < np; ++i)
                {
                    float bestErr = Norm(colors[i], palette[0]);
                    for (int j = 1; j < numIndices && bestErr > 0; ++j)
                    {
                        float err = Norm(colors[i], palette[j]);
                        if (err > bestErr) break; // error increased, so we're done searching
                        if (err < bestErr) bestErr = err;
                    }
                    best[i] = bestErr;
                }
                break;
        }
        // Same left-to-right float accumulation as the scalar encoder.
        float totErr = 0;
        for (int i = 0; i < np; i++) totErr += best[i];
        return totErr;
    }

    /// <summary>Scalar BC6H Norm (identical to Bc6hCpu.Norm).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Norm(in IntColor a, in IntColor b)
    {
        float dr = (float)a.R - b.R;
        float dg = (float)a.G - b.G;
        float db = (float)a.B - b.B;
        return dr * dr + dg * dg + db * db;
    }

    /// <summary>Deinterleave up to 16 IntColors into float SoA lanes (int -> float is exact here).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LoadHdr(ReadOnlySpan<IntColor> colors, int np, Span<float> r, Span<float> g, Span<float> b)
    {
        for (int i = 0; i < np; i++)
        {
            ref readonly var c = ref colors[i];
            r[i] = c.R;
            g[i] = c.G;
            b[i] = c.B;
        }
    }

    private static void HdrBest512(ReadOnlySpan<IntColor> colors, int np, ReadOnlySpan<IntColor> palette, int n, Span<float> bestOut)
    {
        Span<float> rs = stackalloc float[16], gs = stackalloc float[16], bs = stackalloc float[16];
        LoadHdr(colors, np, rs, gs, bs);
        var r = Vector512.Create((ReadOnlySpan<float>)rs);
        var g = Vector512.Create((ReadOnlySpan<float>)gs);
        var b = Vector512.Create((ReadOnlySpan<float>)bs);

        var best = Norm512(r, g, b, palette[0]);
        var active = Vector512.LessThan(Vector512<int>.Indices, Vector512.Create(np)).AsSingle();
        for (int j = 1; j < n; j++)
        {
            active &= Vector512.GreaterThan(best, Vector512<float>.Zero);
            if (Vector512.ExtractMostSignificantBits(active) == 0) break;
            var err = Norm512(r, g, b, palette[j]);
            var brk = active & Vector512.GreaterThan(err, best);
            var upd = active & Vector512.LessThan(err, best);
            best = Vector512.ConditionalSelect(upd, err, best);
            active = Vector512.AndNot(active, brk);
        }
        best.CopyTo(bestOut);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Norm512(Vector512<float> r, Vector512<float> g, Vector512<float> b, in IntColor p)
    {
        // Same operation order as Norm(): ((dr*dr + dg*dg) + db*db), no fused multiply-add.
        var dr = r - Vector512.Create((float)p.R);
        var dg = g - Vector512.Create((float)p.G);
        var db = b - Vector512.Create((float)p.B);
        return dr * dr + dg * dg + db * db;
    }

    private static void HdrBestVector(ReadOnlySpan<IntColor> colors, int np, ReadOnlySpan<IntColor> palette, int n, Span<float> bestOut)
    {
        Span<float> rs = stackalloc float[16], gs = stackalloc float[16], bs = stackalloc float[16];
        LoadHdr(colors, np, rs, gs, bs);
        int w = Vector<float>.Count;
        var npv = new Vector<int>(np);
        for (int c = 0; c < np; c += w)
        {
            var r = new Vector<float>(rs[c..]);
            var g = new Vector<float>(gs[c..]);
            var b = new Vector<float>(bs[c..]);
            var best = NormVector(r, g, b, palette[0]);
            var active = Vector.LessThan(Vector<int>.Indices + new Vector<int>(c), npv);
            for (int j = 1; j < n; j++)
            {
                active &= Vector.GreaterThan(best, Vector<float>.Zero);
                if (active == Vector<int>.Zero) break;
                var err = NormVector(r, g, b, palette[j]);
                var brk = active & Vector.GreaterThan(err, best);
                var upd = active & Vector.LessThan(err, best);
                best = Vector.ConditionalSelect(upd, err, best);
                active = Vector.AndNot(active, brk);
            }
            best.CopyTo(bestOut[c..]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> NormVector(Vector<float> r, Vector<float> g, Vector<float> b, in IntColor p)
    {
        var dr = r - new Vector<float>(p.R);
        var dg = g - new Vector<float>(p.G);
        var db = b - new Vector<float>(p.B);
        return dr * dr + dg * dg + db * db;
    }
}
