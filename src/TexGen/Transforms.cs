//-------------------------------------------------------------------------------------
// Transforms.cs
//
// CPU image transforms mirroring texconv's pre/post-processing stages: flip
// (-hflip/-vflip), channel swizzle (-swizzle), colorkey (-c), and premultiply/
// straighten alpha (-pmalpha/-alpha). All operate on R8G8B8A8 images except flip,
// which is generic over pixel size (works on HDR float images too). Rows are
// processed in parallel.
//-------------------------------------------------------------------------------------

namespace TexGen;

public static class Transforms
{
    private static void AssertRgba8(Image img, string op)
    {
        if (!img.IsRgba8) throw new ArgumentException($"{op}: source must be R8G8B8A8_UNORM(_SRGB); got {img.Format}");
    }

    // JS Math.round semantics for non-negative values (texconv-js parity).
    private static int Round(float v) => (int)MathF.Floor(v + 0.5f);

    /// <summary>Mirror an image horizontally and/or vertically (texconv -hflip / -vflip).</summary>
    public static Image Flip(Image src, bool horizontal, bool vertical)
    {
        if (!horizontal && !vertical) return src;
        int bpp = Dxgi.BitsPerPixel(src.Format) / 8;
        if (bpp == 0) throw new ArgumentException($"Flip: unsupported format {src.Format}");
        var dst = Image.Create(src.Format, src.Width, src.Height);
        Parallel.For(0, src.Height, y =>
        {
            int sy = vertical ? src.Height - 1 - y : y;
            var srcRow = src.Pixels.AsSpan(sy * src.RowPitch, src.Width * bpp);
            var dstRow = dst.Pixels.AsSpan(y * dst.RowPitch, src.Width * bpp);
            if (!horizontal)
            {
                srcRow.CopyTo(dstRow);
                return;
            }
            for (int x = 0; x < src.Width; x++)
                srcRow.Slice((src.Width - 1 - x) * bpp, bpp).CopyTo(dstRow.Slice(x * bpp, bpp));
        });
        return dst;
    }

    /// <summary>
    /// Parse a texconv -swizzle mask into per-channel sources. Accepts 1-4 chars from
    /// rgba / xyzw / 0 / 1 (case-insensitive); like upstream, a short mask replicates its
    /// last element into the remaining positions. Source codes: 0-3 = channel index,
    /// 4 = constant 0, 5 = constant 1.
    /// </summary>
    public static int[] ParseSwizzle(string mask)
    {
        if (string.IsNullOrEmpty(mask) || mask.Length > 4)
            throw new FormatException($"swizzle: mask must be 1-4 chars; got '{mask}'");
        var result = new int[4];
        int last = 0;
        for (int j = 0; j < 4; j++)
        {
            if (j < mask.Length)
            {
                last = char.ToLowerInvariant(mask[j]) switch
                {
                    'r' or 'x' => 0,
                    'g' or 'y' => 1,
                    'b' or 'z' => 2,
                    'a' or 'w' => 3,
                    '0' => 4,
                    '1' => 5,
                    _ => throw new FormatException($"swizzle: invalid mask char '{mask[j]}'"),
                };
            }
            result[j] = last;
        }
        return result;
    }

    /// <summary>Rearrange channels per a parsed swizzle (texconv -swizzle).</summary>
    public static Image Swizzle(Image src, string mask)
    {
        AssertRgba8(src, "Swizzle");
        var sel = ParseSwizzle(mask);
        var dst = Image.Create(src.Format, src.Width, src.Height);
        Parallel.For(0, src.Height, y =>
        {
            int s = y * src.RowPitch, d = y * dst.RowPitch;
            for (int x = 0; x < src.Width; x++)
            {
                for (int c = 0; c < 4; c++)
                {
                    int code = sel[c];
                    dst.Pixels[d + x * 4 + c] = code switch
                    {
                        4 => 0,
                        5 => 255,
                        _ => src.Pixels[s + x * 4 + code],
                    };
                }
            }
        });
        return dst;
    }

    /// <summary>
    /// Colorkey (texconv -c): pixels whose RGB is within upstream's 0.2 tolerance of
    /// <paramref name="key"/> (0xRRGGBB) become transparent black; all other pixels get
    /// opaque alpha, exactly as texconv's TransformImage lambda does.
    /// </summary>
    public static Image ColorKey(Image src, uint key)
    {
        AssertRgba8(src, "ColorKey");
        int kr = (int)((key >> 16) & 0xff), kg = (int)((key >> 8) & 0xff), kb = (int)(key & 0xff);
        const float tol = 0.2f * 255;
        var dst = Image.Create(src.Format, src.Width, src.Height);
        Parallel.For(0, src.Height, y =>
        {
            int s = y * src.RowPitch, d = y * dst.RowPitch;
            for (int x = 0; x < src.Width; x++)
            {
                int i = s + x * 4, o = d + x * 4;
                if (Math.Abs(src.Pixels[i] - kr) <= tol && Math.Abs(src.Pixels[i + 1] - kg) <= tol &&
                    Math.Abs(src.Pixels[i + 2] - kb) <= tol)
                {
                    dst.Pixels.AsSpan(o, 4).Clear();
                }
                else
                {
                    dst.Pixels[o] = src.Pixels[i];
                    dst.Pixels[o + 1] = src.Pixels[i + 1];
                    dst.Pixels[o + 2] = src.Pixels[i + 2];
                    dst.Pixels[o + 3] = 255;
                }
            }
        });
        return dst;
    }

    /// <summary>Premultiply color by alpha (texconv -pmalpha) or undo it (texconv -alpha).</summary>
    public static Image PremultiplyAlpha(Image src, bool undo = false)
    {
        AssertRgba8(src, "PremultiplyAlpha");
        var dst = Image.Create(src.Format, src.Width, src.Height);
        Parallel.For(0, src.Height, y =>
        {
            int s = y * src.RowPitch, d = y * dst.RowPitch;
            for (int x = 0; x < src.Width; x++)
            {
                int i = s + x * 4, o = d + x * 4;
                byte a = src.Pixels[i + 3];
                float scale = undo ? (a == 0 ? 0 : 255f / a) : a / 255f;
                for (int c = 0; c < 3; c++) dst.Pixels[o + c] = (byte)Math.Min(255, Round(src.Pixels[i + c] * scale));
                dst.Pixels[o + 3] = a;
            }
        });
        return dst;
    }
}
