//-------------------------------------------------------------------------------------
// Gpu/ImageKernels.cs
//
// GPU box-filter resize / mip downsample (port of texconv-js resize.wgsl) plus a
// premultiply-alpha kernel for texconv -pmalpha, which runs after mip generation.
// One thread per destination texel; each averages its source footprint, optionally
// linearizing sRGB before averaging (and re-encoding after) so mip colors are correct.
//-------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Runtime;

namespace TexGen.Gpu;

public struct ResizeParams
{
    public int SrcW, SrcH, DstW, DstH;
    /// <summary>1 = treat data as sRGB during averaging (RGBA8 path only).</summary>
    public int Srgb;
}

internal sealed class ImageKernels
{
    private readonly Accelerator _acc;
    private readonly Action<Index2D, ResizeParams, ArrayView<uint>, ArrayView<uint>> _resize;
    private readonly Action<Index2D, ResizeParams, ArrayView<Float4>, ArrayView<Float4>> _resizeF32;
    private readonly Action<Index1D, ArrayView<uint>> _premultiply;

    public ImageKernels(Accelerator acc)
    {
        _acc = acc;
        _resize = acc.LoadAutoGroupedStreamKernel<Index2D, ResizeParams, ArrayView<uint>, ArrayView<uint>>(ResizeKernel);
        _resizeF32 = acc.LoadAutoGroupedStreamKernel<Index2D, ResizeParams, ArrayView<Float4>, ArrayView<Float4>>(ResizeF32Kernel);
        _premultiply = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>>(PremultiplyKernel);
    }

    /// <summary>Box-filter resize to (dstW, dstH). HDR float data is always filtered linearly.</summary>
    public GpuTexture Resize(GpuTexture src, int dstW, int dstH, bool srgb)
    {
        var dst = GpuTexture.Allocate(_acc, dstW, dstH, src.Format);
        var p = new ResizeParams { SrcW = src.Width, SrcH = src.Height, DstW = dstW, DstH = dstH, Srgb = srgb && !src.IsFloat ? 1 : 0 };
        if (src.IsFloat) _resizeF32(new Index2D(dstW, dstH), p, src.Rgba32FView, dst.Rgba32FView);
        else _resize(new Index2D(dstW, dstH), p, src.Rgba8View, dst.Rgba8View);
        return dst;
    }

    /// <summary>Premultiply color by alpha in place (RGBA8 only).</summary>
    public void Premultiply(GpuTexture tex)
    {
        if (tex.IsFloat) throw new InvalidOperationException("Premultiply: RGBA8 textures only");
        _premultiply((int)tex.Rgba8View.Length, tex.Rgba8View);
    }

    //---------------------------------------------------------------------------------
    // Kernels
    //---------------------------------------------------------------------------------

    private static float SrgbToLinear(float c) =>
        c <= 0.04045f ? c / 12.92f : MathF.Pow((c + 0.055f) / 1.055f, 2.4f);

    private static float LinearToSrgb(float c) =>
        c <= 0.0031308f ? c * 12.92f : 1.055f * MathF.Pow(c, 1f / 2.4f) - 0.055f;

    /// <summary>Source footprint [x0,x1) x [y0,y1) of destination texel (x, y).</summary>
    private static void Footprint(ResizeParams p, int x, int y, out int x0, out int x1, out int y0, out int y1)
    {
        x0 = (int)((long)x * p.SrcW / p.DstW);
        y0 = (int)((long)y * p.SrcH / p.DstH);
        x1 = (int)((long)(x + 1) * p.SrcW / p.DstW);
        y1 = (int)((long)(y + 1) * p.SrcH / p.DstH);
        x1 = Math.Min(Math.Max(x1, x0 + 1), p.SrcW);
        y1 = Math.Min(Math.Max(y1, y0 + 1), p.SrcH);
    }

    private static void ResizeKernel(Index2D idx, ResizeParams p, ArrayView<uint> src, ArrayView<uint> dst)
    {
        int x = idx.X, y = idx.Y;
        Footprint(p, x, y, out int x0, out int x1, out int y0, out int y1);
        float r = 0, g = 0, b = 0, a = 0;
        for (int sy = y0; sy < y1; sy++)
        {
            for (int sx = x0; sx < x1; sx++)
            {
                var t = Texel.UnpackUnorm(src[sy * p.SrcW + sx]);
                if (p.Srgb != 0)
                {
                    t.X = SrgbToLinear(t.X);
                    t.Y = SrgbToLinear(t.Y);
                    t.Z = SrgbToLinear(t.Z);
                }
                r += t.X;
                g += t.Y;
                b += t.Z;
                a += t.W;
            }
        }
        float n = (x1 - x0) * (y1 - y0);
        var c = new Float4(r / n, g / n, b / n, a / n);
        if (p.Srgb != 0)
        {
            c.X = LinearToSrgb(c.X);
            c.Y = LinearToSrgb(c.Y);
            c.Z = LinearToSrgb(c.Z);
        }
        dst[y * p.DstW + x] = Texel.PackUnorm(c);
    }

    private static void ResizeF32Kernel(Index2D idx, ResizeParams p, ArrayView<Float4> src, ArrayView<Float4> dst)
    {
        int x = idx.X, y = idx.Y;
        Footprint(p, x, y, out int x0, out int x1, out int y0, out int y1);
        var acc = new Float4(0, 0, 0, 0);
        for (int sy = y0; sy < y1; sy++)
            for (int sx = x0; sx < x1; sx++)
                acc += src[sy * p.SrcW + sx];
        dst[y * p.DstW + x] = acc / ((x1 - x0) * (y1 - y0));
    }

    private static void PremultiplyKernel(Index1D i, ArrayView<uint> tex)
    {
        uint p = tex[i];
        uint a = p >> 24;
        float scale = a / 255f;
        uint r = Math.Min(255u, (uint)MathF.Floor((p & 0xFF) * scale + 0.5f));
        uint g = Math.Min(255u, (uint)MathF.Floor(((p >> 8) & 0xFF) * scale + 0.5f));
        uint b = Math.Min(255u, (uint)MathF.Floor(((p >> 16) & 0xFF) * scale + 0.5f));
        tex[i] = r | (g << 8) | (b << 16) | (a << 24);
    }
}
