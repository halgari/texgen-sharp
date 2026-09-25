//-------------------------------------------------------------------------------------
// The SIMD error kernels must match the scalar DirectXTex loops bit for bit at every
// width. Random palettes are deliberately unsorted so the "error increased -> stop"
// early exit fires at arbitrary points, and exact hits exercise the best == 0 stop.
//-------------------------------------------------------------------------------------

using TexGen.Cpu;

namespace TexGen.Tests.Cpu;

public class SimdTests
{
    private static IEnumerable<SimdLevel> Levels() =>
        Enum.GetValues<SimdLevel>().Where(l => l != SimdLevel.Scalar && l <= Bc67Simd.Hardware);

    [Fact]
    public void ReportsHardwareLevel() =>
        // Informational: which paths this machine exercises (all are compared to scalar below).
        Assert.True(Enum.IsDefined(Bc67Simd.Hardware), $"hardware={Bc67Simd.Hardware}, in use={Bc67Simd.Level}");

    [Fact]
    public void LdrMapColorsMatchesScalarAtEveryLevel()
    {
        var rng = new Random(7);
        var colors = new LdrColorA[16];
        for (int trial = 0; trial < 40000; trial++)
        {
            // Narrow value ranges make ties and exact (zero-error) matches common.
            int range = trial % 3 == 0 ? 6 : 256;
            LdrColorA Rand() => new((byte)rng.Next(range), (byte)rng.Next(range), (byte)rng.Next(range), (byte)rng.Next(range));
            for (int i = 0; i < 16; i++) colors[i] = Rand();
            var a = Rand();
            var b = Rand();
            if (trial % 7 == 0) a = colors[rng.Next(16)]; // an endpoint equal to a pixel -> exact hit
            int np = 1 + trial % 16;
            // Index precisions used by BC7: (2,0) (3,0) (4,0) and the mode 4/5 pairs.
            var (p1, p2) = (trial % 6) switch { 0 => (2, 0), 1 => (3, 0), 2 => (4, 0), 3 => (2, 3), 4 => (3, 2), _ => (2, 2) };
            // Early-exit thresholds: none, tight, fractional, zero.
            float minErr = (trial % 4) switch { 0 => float.MaxValue, 1 => rng.Next(0, 200000), 2 => rng.Next(0, 20000) + 0.5f, _ => 0 };
            // Short spans take the copy path, 16-long spans are read in place.
            ReadOnlySpan<LdrColorA> span = trial % 2 == 0 ? colors : colors.AsSpan(0, np);

            float expected = Bc67Simd.LdrMapColors(span, np, a, b, p1, p2, minErr, SimdLevel.Scalar);
            foreach (var level in Levels())
                Assert.Equal(expected, Bc67Simd.LdrMapColors(span, np, a, b, p1, p2, minErr, level));
        }
    }

    [Fact]
    public void HdrErrorSumMatchesScalarBitForBitAtEveryLevel()
    {
        var rng = new Random(11);
        var colors = new IntColor[16];
        var palette = new IntColor[16];
        for (int trial = 0; trial < 20000; trial++)
        {
            // Half-float bit patterns: full signed range, plus narrow ranges for ties.
            int lo = trial % 2 == 0 ? -0x7bff : 0, hi = trial % 3 == 0 ? 8 : 0x7bff;
            for (int i = 0; i < 16; i++)
            {
                colors[i] = new IntColor { R = rng.Next(lo, hi + 1), G = rng.Next(lo, hi + 1), B = rng.Next(lo, hi + 1) };
                palette[i] = new IntColor { R = rng.Next(lo, hi + 1), G = rng.Next(lo, hi + 1), B = rng.Next(lo, hi + 1) };
            }
            if (trial % 5 == 0) palette[rng.Next(16)] = colors[rng.Next(16)];
            int np = 1 + trial % 16;
            int n = trial % 2 == 0 ? 8 : 16; // BC6H: 3-bit (two-region) or 4-bit (one-region) indices

            float expected = Bc67Simd.HdrErrorSum(colors, np, palette, n, SimdLevel.Scalar);
            foreach (var level in Levels())
            {
                float actual = Bc67Simd.HdrErrorSum(colors, np, palette, n, level);
                Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual));
            }
        }
    }
}
