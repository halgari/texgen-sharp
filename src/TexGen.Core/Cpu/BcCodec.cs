//-------------------------------------------------------------------------------------
// Cpu/BcCodec.cs
//
// DirectXTex CPU block encoders for BC1/BC2/BC3 (port of BC.cpp) and BC4/BC5
// (port of BC4BC5.cpp). These are texconv's reference-quality encoders: BC1-3 do a
// Newton-iteration endpoint fit along the best color diagonal (OptimizeRGB) with
// perceptual luminance weighting, optional Floyd-Steinberg dithering of color and
// alpha, and BC1's 3-color + transparent mode below the alpha threshold; BC4/BC5 fit
// 6- or 8-step palettes with OptimizeAlpha and pick nearest indices.
//
// Pixels are 16 Vector4s (row-major), [0,1] for UNORM and [-1,1] for the SNORM
// encoders. Output layout matches D3DX_BC1/BC2/BC3 (little-endian). Everything runs
// on the stack: no per-block allocations.
//-------------------------------------------------------------------------------------

using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace TexGen.Cpu;

internal static class BcCodec
{
    private const int NumPixels = BcConstants.NumPixelsPerBlock;

    // Perceptual weightings for the importance of each channel.
    private static readonly HdrColorA Luminance = new(0.2125f / 0.7154f, 1.0f, 0.0721f / 0.7154f, 1.0f);
    private static readonly HdrColorA LuminanceInv = new(0.7154f / 0.2125f, 1.0f, 0.7154f / 0.0721f, 1.0f);

    private static readonly float[] C3 = [2f / 2f, 1f / 2f, 0f / 2f];
    private static readonly float[] D3 = [0f / 2f, 1f / 2f, 2f / 2f];
    private static readonly float[] C4 = [3f / 3f, 2f / 3f, 1f / 3f, 0f / 3f];
    private static readonly float[] D4 = [0f / 3f, 1f / 3f, 2f / 3f, 3f / 3f];

    private static readonly int[] Steps3 = [0, 2, 1];
    private static readonly int[] Steps4 = [0, 2, 3, 1];
    private static readonly int[] AlphaSteps6 = [0, 2, 3, 4, 5, 1];
    private static readonly int[] AlphaSteps8 = [0, 2, 3, 4, 5, 6, 7, 1];

    //---------------------------------------------------------------------------------
    // Decode/Encode RGB 5/6/5 colors
    //---------------------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static HdrColorA Decode565(ushort w565) => new(
        ((w565 >> 11) & 31) * (1.0f / 31.0f),
        ((w565 >> 5) & 63) * (1.0f / 63.0f),
        (w565 & 31) * (1.0f / 31.0f),
        1.0f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ushort Encode565(in HdrColorA c)
    {
        float r = c.R < 0.0f ? 0.0f : c.R > 1.0f ? 1.0f : c.R;
        float g = c.G < 0.0f ? 0.0f : c.G > 1.0f ? 1.0f : c.G;
        float b = c.B < 0.0f ? 0.0f : c.B > 1.0f ? 1.0f : c.B;
        return (ushort)(((int)(r * 31.0f + 0.5f) << 11) | ((int)(g * 63.0f + 0.5f) << 5) | (int)(b * 31.0f + 0.5f));
    }

    //---------------------------------------------------------------------------------
    // Floyd-Steinberg error diffusion within a 4x4 block (shared by all dither paths).
    //---------------------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Diffuse(Span<float> error, int i, float diff)
    {
        if ((i & 3) != 3) error[i + 1] += diff * (7.0f / 16.0f);
        if (i < 12)
        {
            if ((i & 3) != 0) error[i + 3] += diff * (3.0f / 16.0f);
            error[i + 4] += diff * (5.0f / 16.0f);
            if ((i & 3) != 3) error[i + 5] += diff * (1.0f / 16.0f);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DiffuseRgb(Span<HdrColorA> error, int i, float dr, float dg, float db)
    {
        if ((i & 3) != 3)
        {
            error[i + 1].R += dr * (7.0f / 16.0f);
            error[i + 1].G += dg * (7.0f / 16.0f);
            error[i + 1].B += db * (7.0f / 16.0f);
        }
        if (i < 12)
        {
            if ((i & 3) != 0)
            {
                error[i + 3].R += dr * (3.0f / 16.0f);
                error[i + 3].G += dg * (3.0f / 16.0f);
                error[i + 3].B += db * (3.0f / 16.0f);
            }
            error[i + 4].R += dr * (5.0f / 16.0f);
            error[i + 4].G += dg * (5.0f / 16.0f);
            error[i + 4].B += db * (5.0f / 16.0f);
            if ((i & 3) != 3)
            {
                error[i + 5].R += dr * (1.0f / 16.0f);
                error[i + 5].G += dg * (1.0f / 16.0f);
                error[i + 5].B += db * (1.0f / 16.0f);
            }
        }
    }

    //---------------------------------------------------------------------------------
    // OptimizeRGB: 6D root finding for the two endpoints of the color axis.
    //---------------------------------------------------------------------------------

    private static void OptimizeRGB(out HdrColorA pX, out HdrColorA pY, ReadOnlySpan<HdrColorA> points, int steps, BcFlags flags)
    {
        const float epsilon = (0.25f / 64.0f) * (0.25f / 64.0f);
        var pC = steps == 3 ? C3 : C4;
        var pD = steps == 3 ? D3 : D4;

        // Find Min and Max points, as starting point
        HdrColorA x = (flags & BcFlags.Uniform) != 0 ? new HdrColorA(1f, 1f, 1f, 1f) : Luminance;
        var y = new HdrColorA(0f, 0f, 0f, 1f);

        for (int i = 0; i < NumPixels; i++)
        {
            ref readonly var p = ref points[i];
            if (p.R < x.R) x.R = p.R;
            if (p.G < x.G) x.G = p.G;
            if (p.B < x.B) x.B = p.B;
            if (p.R > y.R) y.R = p.R;
            if (p.G > y.G) y.G = p.G;
            if (p.B > y.B) y.B = p.B;
        }

        // Diagonal axis
        float abR = y.R - x.R, abG = y.G - x.G, abB = y.B - x.B;
        float fAB = abR * abR + abG * abG + abB * abB;

        // Single color block.. no need to root-find
        if (fAB < 1.17549435e-38f) // FLT_MIN
        {
            pX = new HdrColorA(x.R, x.G, x.B, 1.0f);
            pY = new HdrColorA(y.R, y.G, y.B, 1.0f);
            return;
        }

        // Try all four axis directions, to determine which diagonal best fits data
        float fABInv = 1.0f / fAB;
        float dirR = abR * fABInv, dirG = abG * fABInv, dirB = abB * fABInv;
        float midR = (x.R + y.R) * 0.5f, midG = (x.G + y.G) * 0.5f, midB = (x.B + y.B) * 0.5f;

        float dir0 = 0, dir1 = 0, dir2 = 0, dir3 = 0;
        for (int i = 0; i < NumPixels; i++)
        {
            float ptR = (points[i].R - midR) * dirR;
            float ptG = (points[i].G - midG) * dirG;
            float ptB = (points[i].B - midB) * dirB;

            float f = ptR + ptG + ptB;
            dir0 += f * f;
            f = ptR + ptG - ptB;
            dir1 += f * f;
            f = ptR - ptG + ptB;
            dir2 += f * f;
            f = ptR - ptG - ptB;
            dir3 += f * f;
        }

        float dirMax = dir0;
        int iDirMax = 0;
        if (dir1 > dirMax) { dirMax = dir1; iDirMax = 1; }
        if (dir2 > dirMax) { dirMax = dir2; iDirMax = 2; }
        if (dir3 > dirMax) { iDirMax = 3; }

        if ((iDirMax & 2) != 0) (x.G, y.G) = (y.G, x.G);
        if ((iDirMax & 1) != 0) (x.B, y.B) = (y.B, x.B);

        // Two color block.. no need to root-find
        if (fAB < 1.0f / 4096.0f)
        {
            pX = new HdrColorA(x.R, x.G, x.B, 1.0f);
            pY = new HdrColorA(y.R, y.G, y.B, 1.0f);
            return;
        }

        // Use Newton's Method to find local minima of sum-of-squares error.
        float fSteps = steps - 1;
        Span<HdrColorA> pSteps = stackalloc HdrColorA[4];

        for (int iteration = 0; iteration < 8; iteration++)
        {
            // Calculate new steps
            for (int s = 0; s < steps; s++)
            {
                pSteps[s].R = x.R * pC[s] + y.R * pD[s];
                pSteps[s].G = x.G * pC[s] + y.G * pD[s];
                pSteps[s].B = x.B * pC[s] + y.B * pD[s];
                pSteps[s].A = 1.0f;
            }

            // Calculate color direction
            dirR = y.R - x.R;
            dirG = y.G - x.G;
            dirB = y.B - x.B;

            float fLen = dirR * dirR + dirG * dirG + dirB * dirB;
            if (fLen < (1.0f / 4096.0f)) break;

            float fScale = fSteps / fLen;
            dirR *= fScale;
            dirG *= fScale;
            dirB *= fScale;

            // Evaluate function, and derivatives
            float d2X = 0f, d2Y = 0f;
            float dXr = 0f, dXg = 0f, dXb = 0f;
            float dYr = 0f, dYg = 0f, dYb = 0f;

            for (int i = 0; i < NumPixels; i++)
            {
                ref readonly var p = ref points[i];
                float fDot = (p.R - x.R) * dirR + (p.G - x.G) * dirG + (p.B - x.B) * dirB;

                int iStep;
                if (fDot <= 0.0f) iStep = 0;
                else if (fDot >= fSteps) iStep = steps - 1;
                else iStep = (int)(uint)(fDot + 0.5f);

                float diffR = pSteps[iStep].R - p.R;
                float diffG = pSteps[iStep].G - p.G;
                float diffB = pSteps[iStep].B - p.B;

                float fC = pC[iStep] * (1.0f / 8.0f);
                float fD = pD[iStep] * (1.0f / 8.0f);

                d2X += fC * pC[iStep];
                dXr += fC * diffR;
                dXg += fC * diffG;
                dXb += fC * diffB;

                d2Y += fD * pD[iStep];
                dYr += fD * diffR;
                dYg += fD * diffG;
                dYb += fD * diffB;
            }

            // Move endpoints
            if (d2X > 0.0f)
            {
                float f = -1.0f / d2X;
                x.R += dXr * f;
                x.G += dXg * f;
                x.B += dXb * f;
            }

            if (d2Y > 0.0f)
            {
                float f = -1.0f / d2Y;
                y.R += dYr * f;
                y.G += dYg * f;
                y.B += dYb * f;
            }

            if (dXr * dXr < epsilon && dXg * dXg < epsilon && dXb * dXb < epsilon
                && dYr * dYr < epsilon && dYg * dYg < epsilon && dYb * dYb < epsilon)
            {
                break;
            }
        }

        pX = new HdrColorA(x.R, x.G, x.B, 1.0f);
        pY = new HdrColorA(y.R, y.G, y.B, 1.0f);
    }

    //---------------------------------------------------------------------------------
    // EncodeBC1 (internal): writes the 8-byte D3DX_BC1 color block.
    //---------------------------------------------------------------------------------

    private static void WriteBC1(Span<byte> bc, ushort rgb0, ushort rgb1, uint bitmap)
    {
        BinaryPrimitives.WriteUInt16LittleEndian(bc, rgb0);
        BinaryPrimitives.WriteUInt16LittleEndian(bc[2..], rgb1);
        BinaryPrimitives.WriteUInt32LittleEndian(bc[4..], bitmap);
    }

    private static void EncodeBC1Block(Span<byte> bc, ReadOnlySpan<HdrColorA> pColor, bool colorKey, float threshold, BcFlags flags)
    {
        bool dither = (flags & BcFlags.DitherRgb) != 0;
        bool uniform = (flags & BcFlags.Uniform) != 0;

        // Determine if we need to colorkey this block
        int uSteps;
        if (colorKey)
        {
            int uColorKey = 0;
            for (int i = 0; i < NumPixels; ++i)
            {
                if (pColor[i].A < threshold) uColorKey++;
            }

            if (uColorKey == NumPixels)
            {
                WriteBC1(bc, 0x0000, 0xffff, 0xffffffff);
                return;
            }

            uSteps = uColorKey > 0 ? 3 : 4;
        }
        else
        {
            uSteps = 4;
        }

        // Quantize block to R56B5, using Floyd Stienberg error diffusion. This
        // increases the chance that colors will map directly to the quantized
        // axis endpoints.
        Span<HdrColorA> color = stackalloc HdrColorA[NumPixels];
        Span<HdrColorA> error = stackalloc HdrColorA[NumPixels];
        if (dither) error.Clear();

        for (int i = 0; i < NumPixels; ++i)
        {
            float clrR = pColor[i].R, clrG = pColor[i].G, clrB = pColor[i].B;
            if (dither)
            {
                clrR += error[i].R;
                clrG += error[i].G;
                clrB += error[i].B;
            }

            color[i].R = (int)(clrR * 31.0f + 0.5f) * (1.0f / 31.0f);
            color[i].G = (int)(clrG * 63.0f + 0.5f) * (1.0f / 63.0f);
            color[i].B = (int)(clrB * 31.0f + 0.5f) * (1.0f / 31.0f);
            color[i].A = 1.0f;

            if (dither)
            {
                float a = color[i].A;
                DiffuseRgb(error, i, a * (clrR - color[i].R), a * (clrG - color[i].G), a * (clrB - color[i].B));
            }

            if (!uniform)
            {
                color[i].R *= Luminance.R;
                color[i].G *= Luminance.G;
                color[i].B *= Luminance.B;
            }
        }

        // Perform 6D root finding function to find two endpoints of color axis.
        // Then quantize and sort the endpoints depending on mode.
        OptimizeRGB(out var colorA, out var colorB, color, uSteps, flags);

        HdrColorA colorC, colorD;
        if (uniform)
        {
            colorC = colorA;
            colorD = colorB;
        }
        else
        {
            colorC = new HdrColorA(colorA.R * LuminanceInv.R, colorA.G * LuminanceInv.G, colorA.B * LuminanceInv.B, colorA.A);
            colorD = new HdrColorA(colorB.R * LuminanceInv.R, colorB.G * LuminanceInv.G, colorB.B * LuminanceInv.B, colorB.A);
        }

        ushort wColorA = Encode565(colorC);
        ushort wColorB = Encode565(colorD);

        if (uSteps == 4 && wColorA == wColorB)
        {
            WriteBC1(bc, wColorA, wColorB, 0x00000000);
            return;
        }

        colorC = Decode565(wColorA);
        colorD = Decode565(wColorB);

        if (uniform)
        {
            colorA = colorC;
            colorB = colorD;
        }
        else
        {
            // Alpha keeps OptimizeRGB's value (1.0), as in the original.
            colorA.R = colorC.R * Luminance.R;
            colorA.G = colorC.G * Luminance.G;
            colorA.B = colorC.B * Luminance.B;
            colorB.R = colorD.R * Luminance.R;
            colorB.G = colorD.G * Luminance.G;
            colorB.B = colorD.B * Luminance.B;
        }

        // Calculate color steps
        Span<HdrColorA> step = stackalloc HdrColorA[4];
        ushort rgb0, rgb1;
        if ((uSteps == 3) == (wColorA <= wColorB))
        {
            rgb0 = wColorA;
            rgb1 = wColorB;
            step[0] = colorA;
            step[1] = colorB;
        }
        else
        {
            rgb0 = wColorB;
            rgb1 = wColorA;
            step[0] = colorB;
            step[1] = colorA;
        }

        int[] pSteps;
        if (uSteps == 3)
        {
            pSteps = Steps3;
            step[2] = HdrColorA.Lerp(step[0], step[1], 0.5f);
        }
        else
        {
            pSteps = Steps4;
            step[2] = HdrColorA.Lerp(step[0], step[1], 1.0f / 3.0f);
            step[3] = HdrColorA.Lerp(step[0], step[1], 2.0f / 3.0f);
        }

        // Calculate color direction
        float dirR = step[1].R - step[0].R;
        float dirG = step[1].G - step[0].G;
        float dirB = step[1].B - step[0].B;

        float fSteps = uSteps - 1;
        float fScale = wColorA != wColorB ? fSteps / (dirR * dirR + dirG * dirG + dirB * dirB) : 0.0f;
        dirR *= fScale;
        dirG *= fScale;
        dirB *= fScale;

        // Encode colors
        uint dw = 0;
        if (dither) error.Clear();

        for (int i = 0; i < NumPixels; ++i)
        {
            if (uSteps == 3 && pColor[i].A < threshold)
            {
                dw = (3u << 30) | (dw >> 2);
                continue;
            }

            float clrR, clrG, clrB;
            if (uniform)
            {
                clrR = pColor[i].R;
                clrG = pColor[i].G;
                clrB = pColor[i].B;
            }
            else
            {
                clrR = pColor[i].R * Luminance.R;
                clrG = pColor[i].G * Luminance.G;
                clrB = pColor[i].B * Luminance.B;
            }

            if (dither)
            {
                clrR += error[i].R;
                clrG += error[i].G;
                clrB += error[i].B;
            }

            float fDot = (clrR - step[0].R) * dirR + (clrG - step[0].G) * dirG + (clrB - step[0].B) * dirB;

            int iStep;
            if (fDot <= 0.0f) iStep = 0;
            else if (fDot >= fSteps) iStep = 1;
            else iStep = pSteps[(int)(uint)(fDot + 0.5f)];

            dw = ((uint)iStep << 30) | (dw >> 2);

            if (dither)
            {
                float a = color[i].A;
                DiffuseRgb(error, i, a * (clrR - step[iStep].R), a * (clrG - step[iStep].G), a * (clrB - step[iStep].B));
            }
        }

        WriteBC1(bc, rgb0, rgb1, dw);
    }

    private static void LoadColors(ReadOnlySpan<Vector4> src, Span<HdrColorA> dst)
    {
        for (int i = 0; i < NumPixels; ++i) dst[i] = HdrColorA.FromVector(src[i]);
    }

    //=================================================================================
    // Entry points
    //=================================================================================

    /// <summary>D3DXEncodeBC1: 8-byte block; pixels with alpha below <paramref name="threshold"/> become transparent.</summary>
    public static void EncodeBC1(Span<byte> bc, ReadOnlySpan<Vector4> color, float threshold, BcFlags flags)
    {
        Span<HdrColorA> c = stackalloc HdrColorA[NumPixels];

        if ((flags & BcFlags.DitherA) != 0)
        {
            Span<float> fError = stackalloc float[NumPixels];
            fError.Clear();
            for (int i = 0; i < NumPixels; ++i)
            {
                var clr = color[i];
                float fAlph = clr.W + fError[i];
                c[i].R = clr.X;
                c[i].G = clr.Y;
                c[i].B = clr.Z;
                c[i].A = (int)(clr.W + fError[i] + 0.5f);
                Diffuse(fError, i, fAlph - c[i].A);
            }
        }
        else
        {
            LoadColors(color, c);
        }

        EncodeBC1Block(bc[..8], c, true, threshold, flags);
    }

    /// <summary>D3DXEncodeBC2: explicit 4-bit alpha (optionally dithered) + BC1 color.</summary>
    public static void EncodeBC2(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags)
    {
        Span<HdrColorA> c = stackalloc HdrColorA[NumPixels];
        LoadColors(color, c);

        // 4-bit alpha part. Dithered using Floyd Stienberg error diffusion.
        bool dither = (flags & BcFlags.DitherA) != 0;
        uint bitmap0 = 0, bitmap1 = 0;
        Span<float> fError = stackalloc float[NumPixels];
        fError.Clear();
        for (int i = 0; i < NumPixels; ++i)
        {
            float fAlph = c[i].A;
            if (dither) fAlph += fError[i];

            uint u = (uint)(fAlph * 15.0f + 0.5f);
            if (i < 8) bitmap0 = (bitmap0 >> 4) | (u << 28);
            else bitmap1 = (bitmap1 >> 4) | (u << 28);

            if (dither) Diffuse(fError, i, fAlph - u * (1.0f / 15.0f));
        }
        BinaryPrimitives.WriteUInt32LittleEndian(bc, bitmap0);
        BinaryPrimitives.WriteUInt32LittleEndian(bc[4..], bitmap1);

        // RGB part
        EncodeBC1Block(bc.Slice(8, 8), c, false, 0f, flags);
    }

    /// <summary>D3DXEncodeBC3: interpolated 3-bit alpha (OptimizeAlpha) + BC1 color.</summary>
    public static void EncodeBC3(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags)
    {
        Span<HdrColorA> c = stackalloc HdrColorA[NumPixels];
        LoadColors(color, c);

        // Quantize block to A8, using Floyd Stienberg error diffusion. This
        // increases the chance that colors will map directly to the quantized
        // axis endpoints.
        bool dither = (flags & BcFlags.DitherA) != 0;
        Span<float> fAlpha = stackalloc float[NumPixels];
        Span<float> fError = stackalloc float[NumPixels];
        fError.Clear();

        float fMinAlpha = c[0].A;
        float fMaxAlpha = c[0].A;

        for (int i = 0; i < NumPixels; ++i)
        {
            float fAlph = c[i].A;
            if (dither) fAlph += fError[i];

            fAlpha[i] = (int)(fAlph * 255.0f + 0.5f) * (1.0f / 255.0f);

            if (fAlpha[i] < fMinAlpha) fMinAlpha = fAlpha[i];
            else if (fAlpha[i] > fMaxAlpha) fMaxAlpha = fAlpha[i];

            if (dither) Diffuse(fError, i, fAlph - fAlpha[i]);
        }

        // RGB part
        EncodeBC1Block(bc.Slice(8, 8), c, false, 0f, flags);

        // Alpha part
        var alpha = bc[..8];
        if (fMinAlpha == 1.0f)
        {
            alpha[0] = 0xff;
            alpha[1] = 0xff;
            alpha[2..].Clear();
            return;
        }

        // Optimize and Quantize Min and Max values
        int uSteps = (fMinAlpha == 0.0f || fMaxAlpha == 1.0f) ? 6 : 8;

        BcAlpha.OptimizeAlpha(false, out float fAlphaA, out float fAlphaB, fAlpha, uSteps);

        byte bAlphaA = (byte)(int)(fAlphaA * 255.0f + 0.5f);
        byte bAlphaB = (byte)(int)(fAlphaB * 255.0f + 0.5f);

        fAlphaA = bAlphaA * (1.0f / 255.0f);
        fAlphaB = bAlphaB * (1.0f / 255.0f);

        // Setup block
        if (uSteps == 8 && bAlphaA == bAlphaB)
        {
            alpha[0] = bAlphaA;
            alpha[1] = bAlphaB;
            alpha[2..].Clear();
            return;
        }

        Span<float> fStep = stackalloc float[8];
        fStep.Clear();
        int[] pSteps;

        if (uSteps == 6)
        {
            alpha[0] = bAlphaA;
            alpha[1] = bAlphaB;
            fStep[0] = fAlphaA;
            fStep[1] = fAlphaB;
            for (int i = 1; i < 5; ++i) fStep[i + 1] = (fStep[0] * (5 - i) + fStep[1] * i) * (1.0f / 5.0f);
            fStep[6] = 0.0f;
            fStep[7] = 1.0f;
            pSteps = AlphaSteps6;
        }
        else
        {
            alpha[0] = bAlphaB;
            alpha[1] = bAlphaA;
            fStep[0] = fAlphaB;
            fStep[1] = fAlphaA;
            for (int i = 1; i < 7; ++i) fStep[i + 1] = (fStep[0] * (7 - i) + fStep[1] * i) * (1.0f / 7.0f);
            pSteps = AlphaSteps8;
        }

        // Encode alpha bitmap
        float fSteps = uSteps - 1;
        float fScale = fStep[0] != fStep[1] ? fSteps / (fStep[1] - fStep[0]) : 0.0f;

        if (dither) fError.Clear();

        for (int iSet = 0; iSet < 2; iSet++)
        {
            uint dw = 0;
            int iMin = iSet * 8;
            int iLim = iMin + 8;

            for (int i = iMin; i < iLim; ++i)
            {
                float fAlph = c[i].A;
                if (dither) fAlph += fError[i];
                float fDot = (fAlph - fStep[0]) * fScale;

                int iStep;
                if (fDot <= 0.0f)
                    iStep = (uSteps == 6 && fAlph <= fStep[0] * 0.5f) ? 6 : 0;
                else if (fDot >= fSteps)
                    iStep = (uSteps == 6 && fAlph >= (fStep[1] + 1.0f) * 0.5f) ? 7 : 1;
                else
                    iStep = pSteps[(int)(uint)(fDot + 0.5f)];

                dw = ((uint)iStep << 21) | (dw >> 3);

                if (dither) Diffuse(fError, i, fAlph - fStep[iStep]);
            }

            alpha[2 + iSet * 3] = (byte)dw;
            alpha[3 + iSet * 3] = (byte)(dw >> 8);
            alpha[4 + iSet * 3] = (byte)(dw >> 16);
        }
    }

    //---------------------------------------------------------------------------------
    // BC4 / BC5 (port of BC4BC5.cpp)
    //---------------------------------------------------------------------------------

    /// <summary>Convert a floating point value to an 8-bit SNORM (FloatToSNorm).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static sbyte FloatToSNorm(float v)
    {
        if (float.IsNaN(v)) v = 0;
        else if (v > 1) v = 1;
        else if (v < -1) v = -1;
        v *= 127;
        v += v >= 0 ? .5f : -.5f;
        return (sbyte)(int)v;
    }

    private static void FindEndPointsBC4U(ReadOnlySpan<float> texels, out byte endpoint0, out byte endpoint1)
    {
        // Find max/min of input texels
        float fBlockMax = texels[0];
        float fBlockMin = texels[0];
        for (int i = 0; i < NumPixels; ++i)
        {
            if (texels[i] < fBlockMin) fBlockMin = texels[i];
            else if (texels[i] > fBlockMax) fBlockMax = texels[i];
        }

        // If there are boundary values in input texels, should use 4 interpolated color
        // values to guarantee the exact code of the boundary values.
        bool using4BlockCodec = fBlockMin == 0f || fBlockMax == 1f;

        if (!using4BlockCodec)
        {
            // 6 interpolated color values
            BcAlpha.OptimizeAlpha(false, out float fStart, out float fEnd, texels, 8);
            endpoint0 = (byte)(fEnd * 255.0f);
            endpoint1 = (byte)(fStart * 255.0f);
        }
        else
        {
            // 4 interpolated color values
            BcAlpha.OptimizeAlpha(false, out float fStart, out float fEnd, texels, 6);
            endpoint1 = (byte)(fEnd * 255.0f);
            endpoint0 = (byte)(fStart * 255.0f);
        }
    }

    private static void FindEndPointsBC4S(ReadOnlySpan<float> texels, out sbyte endpoint0, out sbyte endpoint1)
    {
        // Find max/min of input texels
        float fBlockMax = texels[0];
        float fBlockMin = texels[0];
        for (int i = 0; i < NumPixels; ++i)
        {
            if (texels[i] < fBlockMin) fBlockMin = texels[i];
            else if (texels[i] > fBlockMax) fBlockMax = texels[i];
        }

        bool using4BlockCodec = fBlockMin == -1f || fBlockMax == 1f;

        if (!using4BlockCodec)
        {
            BcAlpha.OptimizeAlpha(true, out float fStart, out float fEnd, texels, 8);
            endpoint0 = FloatToSNorm(fEnd);
            endpoint1 = FloatToSNorm(fStart);
        }
        else
        {
            BcAlpha.OptimizeAlpha(true, out float fStart, out float fEnd, texels, 6);
            endpoint1 = FloatToSNorm(fEnd);
            endpoint0 = FloatToSNorm(fStart);
        }
    }

    /// <summary>BC4_UNORM::DecodeFromIndex.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DecodeUnorm(byte red0, byte red1, int index)
    {
        if (index == 0) return red0 / 255.0f;
        if (index == 1) return red1 / 255.0f;
        float f0 = red0 / 255.0f;
        float f1 = red1 / 255.0f;
        if (red0 > red1)
        {
            index -= 1;
            return (f0 * (7 - index) + f1 * index) / 7.0f;
        }
        if (index == 6) return 0.0f;
        if (index == 7) return 1.0f;
        index -= 1;
        return (f0 * (5 - index) + f1 * index) / 5.0f;
    }

    /// <summary>BC4_SNORM::DecodeFromIndex.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DecodeSnorm(sbyte red0, sbyte red1, int index)
    {
        sbyte s0 = red0 == -128 ? (sbyte)-127 : red0;
        sbyte s1 = red1 == -128 ? (sbyte)-127 : red1;
        if (index == 0) return s0 / 127.0f;
        if (index == 1) return s1 / 127.0f;
        float f0 = s0 / 127.0f;
        float f1 = s1 / 127.0f;
        if (red0 > red1)
        {
            index -= 1;
            return (f0 * (7 - index) + f1 * index) / 7.0f;
        }
        if (index == 6) return -1.0f;
        if (index == 7) return 1.0f;
        index -= 1;
        return (f0 * (5 - index) + f1 * index) / 5.0f;
    }

    /// <summary>FindClosestUNORM/SNORM: nearest palette entry per texel; writes the 8-byte block.</summary>
    private static void WriteBC4(Span<byte> bc, byte red0, byte red1, ReadOnlySpan<float> gradient, ReadOnlySpan<float> texels)
    {
        ulong data = red0 | ((ulong)red1 << 8);
        for (int i = 0; i < NumPixels; ++i)
        {
            int bestIndex = 0;
            float bestDelta = 100000;
            for (int index = 0; index < 8; index++)
            {
                float delta = MathF.Abs(gradient[index] - texels[i]);
                if (delta < bestDelta)
                {
                    bestIndex = index;
                    bestDelta = delta;
                }
            }
            data |= (ulong)bestIndex << (3 * i + 16);
        }
        BinaryPrimitives.WriteUInt64LittleEndian(bc, data);
    }

    private static void EncodeBC4UChannel(Span<byte> bc, ReadOnlySpan<float> texels)
    {
        FindEndPointsBC4U(texels, out byte red0, out byte red1);
        Span<float> gradient = stackalloc float[8];
        for (int i = 0; i < 8; ++i) gradient[i] = DecodeUnorm(red0, red1, i);
        WriteBC4(bc, red0, red1, gradient, texels);
    }

    private static void EncodeBC4SChannel(Span<byte> bc, ReadOnlySpan<float> texels)
    {
        FindEndPointsBC4S(texels, out sbyte red0, out sbyte red1);
        Span<float> gradient = stackalloc float[8];
        for (int i = 0; i < 8; ++i) gradient[i] = DecodeSnorm(red0, red1, i);
        WriteBC4(bc, (byte)red0, (byte)red1, gradient, texels);
    }

    /// <summary>D3DXEncodeBC4U: red channel, [0,1].</summary>
    public static void EncodeBC4U(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags)
    {
        Span<float> u = stackalloc float[NumPixels];
        for (int i = 0; i < NumPixels; ++i) u[i] = color[i].X;
        EncodeBC4UChannel(bc[..8], u);
    }

    /// <summary>D3DXEncodeBC4S: red channel, [-1,1].</summary>
    public static void EncodeBC4S(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags)
    {
        Span<float> u = stackalloc float[NumPixels];
        for (int i = 0; i < NumPixels; ++i) u[i] = color[i].X;
        EncodeBC4SChannel(bc[..8], u);
    }

    /// <summary>D3DXEncodeBC5U: red + green channels, [0,1].</summary>
    public static void EncodeBC5U(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags)
    {
        Span<float> u = stackalloc float[NumPixels];
        Span<float> v = stackalloc float[NumPixels];
        for (int i = 0; i < NumPixels; ++i)
        {
            u[i] = color[i].X;
            v[i] = color[i].Y;
        }
        EncodeBC4UChannel(bc[..8], u);
        EncodeBC4UChannel(bc.Slice(8, 8), v);
    }

    /// <summary>D3DXEncodeBC5S: red + green channels, [-1,1].</summary>
    public static void EncodeBC5S(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags)
    {
        Span<float> u = stackalloc float[NumPixels];
        Span<float> v = stackalloc float[NumPixels];
        for (int i = 0; i < NumPixels; ++i)
        {
            u[i] = color[i].X;
            v[i] = color[i].Y;
        }
        EncodeBC4SChannel(bc[..8], u);
        EncodeBC4SChannel(bc.Slice(8, 8), v);
    }
}
