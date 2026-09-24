//-------------------------------------------------------------------------------------
// Cpu/BcCommon.cs
//
// Shared pieces of the DirectXTex CPU block-compression codec (port of BC.h):
// BC_FLAGS, the HDRColorA working color, and the OptimizeAlpha endpoint search used
// by the BC3/BC4/BC5 alpha-style blocks. Pixels are System.Numerics.Vector4 (the
// XMVECTOR of the original), 16 per 4x4 block in row-major order.
//-------------------------------------------------------------------------------------

using System.Runtime.CompilerServices;

namespace TexGen.Cpu;

/// <summary>DirectXTex BC_FLAGS (values match TEX_COMPRESS_*).</summary>
[Flags]
public enum BcFlags : uint
{
    None = 0,
    /// <summary>Enables dithering for RGB colors for BC1-3.</summary>
    DitherRgb = 0x10000,
    /// <summary>Enables dithering for Alpha channel for BC1-3.</summary>
    DitherA = 0x20000,
    /// <summary>By default, uses perceptual weighting for BC1-3; this flag makes it a uniform weighting.</summary>
    Uniform = 0x40000,
    /// <summary>By default, BC7 skips mode 0 &amp; 2; this flag adds those modes back.</summary>
    Use3Subsets = 0x80000,
    /// <summary>BC7 should only use mode 6; skip other modes.</summary>
    ForceBc7Mode6 = 0x100000,
}

internal static class BcConstants
{
    public const int NumPixelsPerBlock = 16;
}

/// <summary>Floating-point RGBA color used by the BC encoders (HDRColorA).</summary>
internal struct HdrColorA
{
    public float R, G, B, A;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public HdrColorA(float r, float g, float b, float a)
    {
        R = r;
        G = g;
        B = b;
        A = a;
    }

    public static HdrColorA operator +(HdrColorA x, HdrColorA c) => new(x.R + c.R, x.G + c.G, x.B + c.B, x.A + c.A);
    public static HdrColorA operator -(HdrColorA x, HdrColorA c) => new(x.R - c.R, x.G - c.G, x.B - c.B, x.A - c.A);
    public static HdrColorA operator *(HdrColorA x, float f) => new(x.R * f, x.G * f, x.B * f, x.A * f);

    public static HdrColorA operator /(HdrColorA x, float f)
    {
        float inv = 1.0f / f;
        return new(x.R * inv, x.G * inv, x.B * inv, x.A * inv);
    }

    /// <summary>4-component dot product (HDRColorA::operator*(const HDRColorA&amp;)).</summary>
    public static float Dot(HdrColorA x, HdrColorA c) => x.R * c.R + x.G * c.G + x.B * c.B + x.A * c.A;

    public HdrColorA Clamp(float min, float max) => new(
        MathF.Min(max, MathF.Max(min, R)),
        MathF.Min(max, MathF.Max(min, G)),
        MathF.Min(max, MathF.Max(min, B)),
        MathF.Min(max, MathF.Max(min, A)));

    public static HdrColorA Lerp(in HdrColorA c1, in HdrColorA c2, float s) => new(
        c1.R + s * (c2.R - c1.R),
        c1.G + s * (c2.G - c1.G),
        c1.B + s * (c2.B - c1.B),
        c1.A + s * (c2.A - c1.A));

    public static HdrColorA FromVector(System.Numerics.Vector4 v) => new(v.X, v.Y, v.Z, v.W);

    public readonly System.Numerics.Vector4 ToVector() => new(R, G, B, A);
}

internal static class BcAlpha
{
    private static readonly float[] C6 = [5f / 5f, 4f / 5f, 3f / 5f, 2f / 5f, 1f / 5f, 0f / 5f];
    private static readonly float[] D6 = [0f / 5f, 1f / 5f, 2f / 5f, 3f / 5f, 4f / 5f, 5f / 5f];
    private static readonly float[] C8 = [7f / 7f, 6f / 7f, 5f / 7f, 4f / 7f, 3f / 7f, 2f / 7f, 1f / 7f, 0f / 7f];
    private static readonly float[] D8 = [0f / 7f, 1f / 7f, 2f / 7f, 3f / 7f, 4f / 7f, 5f / 7f, 6f / 7f, 7f / 7f];

    /// <summary>
    /// Endpoint optimization for 6- or 8-step single-channel palettes (BC3 alpha, BC4, BC5).
    /// <paramref name="range"/> selects the signed [-1,1] domain (the template's bRange).
    /// </summary>
    public static void OptimizeAlpha(bool range, out float pX, out float pY, ReadOnlySpan<float> points, int steps)
    {
        var pC = steps == 6 ? C6 : C8;
        var pD = steps == 6 ? D6 : D8;

        const float maxValue = 1.0f;
        float minValue = range ? -1.0f : 0.0f;

        float fX = maxValue;
        float fY = minValue;

        if (steps == 8)
        {
            for (int i = 0; i < BcConstants.NumPixelsPerBlock; i++)
            {
                if (points[i] < fX) fX = points[i];
                if (points[i] > fY) fY = points[i];
            }
        }
        else
        {
            for (int i = 0; i < BcConstants.NumPixelsPerBlock; i++)
            {
                if (points[i] < fX && points[i] > minValue) fX = points[i];
                if (points[i] > fY && points[i] < maxValue) fY = points[i];
            }
            if (fX == fY) fY = maxValue;
        }

        float fSteps = steps - 1;
        Span<float> pSteps = stackalloc float[8];

        for (int iteration = 0; iteration < 8; iteration++)
        {
            if ((fY - fX) < (1.0f / 256.0f)) break;

            float fScale = fSteps / (fY - fX);

            for (int s = 0; s < steps; s++) pSteps[s] = pC[s] * fX + pD[s] * fY;
            if (steps == 6)
            {
                pSteps[6] = minValue;
                pSteps[7] = maxValue;
            }

            float dX = 0, dY = 0, d2X = 0, d2Y = 0;
            for (int i = 0; i < BcConstants.NumPixelsPerBlock; i++)
            {
                float fDot = (points[i] - fX) * fScale;
                int step;
                if (fDot <= 0.0f)
                    step = (steps == 6 && points[i] <= (fX + minValue) * 0.5f) ? 6 : 0;
                else if (fDot >= fSteps)
                    step = (steps == 6 && points[i] >= (fY + maxValue) * 0.5f) ? 7 : steps - 1;
                else
                    step = (int)(fDot + 0.5f);

                if (step < steps)
                {
                    float diff = pSteps[step] - points[i];
                    dX += pC[step] * diff;
                    d2X += pC[step] * pC[step];
                    dY += pD[step] * diff;
                    d2Y += pD[step] * pD[step];
                }
            }

            if (d2X > 0.0f) fX -= dX / d2X;
            if (d2Y > 0.0f) fY -= dY / d2Y;
            if (fX > fY) (fX, fY) = (fY, fX);

            if (dX * dX < (1.0f / 64.0f) && dY * dY < (1.0f / 64.0f)) break;
        }

        pX = fX < minValue ? minValue : fX > maxValue ? maxValue : fX;
        pY = fY < minValue ? minValue : fY > maxValue ? maxValue : fY;
    }
}
