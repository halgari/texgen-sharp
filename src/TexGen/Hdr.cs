//-------------------------------------------------------------------------------------
// Hdr.cs
//
// Radiance RGBE (.hdr) reader — the standard HDR input for BC6H workflows. Produces
// a linear-light R32G32B32A32_FLOAT Image. Mirrors DirectXTexHDR.cpp behavior for the
// common case: "-Y h +X w" orientation, 32-bit_rle_rgbe format, both new-style
// (scanline RLE) and flat/old-style data. Also LDR->linear-float and tonemap helpers.
//-------------------------------------------------------------------------------------

using System.Text;

namespace TexGen;

public static class Hdr
{
    /// <summary>True if a filename looks like a Radiance HDR file.</summary>
    public static bool IsHdrFile(string name) => name.EndsWith(".hdr", StringComparison.OrdinalIgnoreCase);

    /// <summary>Parse a Radiance .hdr file into a linear R32G32B32A32_FLOAT Image.</summary>
    public static Image Load(ReadOnlySpan<byte> data)
    {
        int pos = 0;
        string ReadLine(ReadOnlySpan<byte> d)
        {
            int end = pos;
            while (end < d.Length && d[end] != 0x0a) end++;
            var line = Encoding.UTF8.GetString(d[pos..end]);
            pos = end + 1;
            return line;
        }

        var magic = ReadLine(data);
        if (!magic.StartsWith("#?RADIANCE", StringComparison.Ordinal) && !magic.StartsWith("#?RGBE", StringComparison.Ordinal))
            throw new InvalidDataException("HDR: missing #?RADIANCE signature");

        double exposure = 1;
        while (true)
        {
            if (pos >= data.Length) throw new InvalidDataException("HDR: truncated header");
            var line = ReadLine(data);
            if (line.Length == 0) break; // blank line ends the header
            if (line.StartsWith('#')) continue;
            int eq = line.IndexOf('=');
            if (eq < 0) continue;
            var key = line[..eq];
            var value = line[(eq + 1)..];
            if (key == "FORMAT" && value != "32-bit_rle_rgbe")
                throw new InvalidDataException($"HDR: unsupported FORMAT '{value}'");
            if (key == "EXPOSURE" && double.TryParse(value, System.Globalization.CultureInfo.InvariantCulture, out var e) && e > 0)
                exposure *= e;
        }

        var res = ReadLine(data).Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        // Standard orientation only: "-Y height +X width".
        if (res.Length != 4 || res[0] != "-Y" || res[2] != "+X")
            throw new InvalidDataException($"HDR: unsupported resolution line '{string.Join(' ', res)}'");
        if (!int.TryParse(res[1], out int height) || !int.TryParse(res[3], out int width) || width <= 0 || height <= 0)
            throw new InvalidDataException("HDR: bad dimensions");

        var img = Image.Create(DxgiFormat.R32G32B32A32_FLOAT, width, height);
        var outF = img.Floats;
        float invExposure = (float)(1 / exposure);
        var scanline = new byte[width * 4]; // RGBE per pixel

        for (int y = 0; y < height; y++)
        {
            pos = ReadScanline(data, pos, scanline, width);
            int o = y * width * 4;
            for (int x = 0; x < width; x++)
            {
                int e = scanline[x * 4 + 3];
                // rgbe -> float: value = mantissa * 2^(e-136); zero exponent = black.
                float scale = e != 0 ? MathF.ScaleB(1f, e - 136) * invExposure : 0;
                outF[o++] = scanline[x * 4] * scale;
                outF[o++] = scanline[x * 4 + 1] * scale;
                outF[o++] = scanline[x * 4 + 2] * scale;
                outF[o++] = 1;
            }
        }
        return img;
    }

    /// <summary>Decode one scanline (RGBE interleaved into <paramref name="dst"/>), returning the new read offset.</summary>
    private static int ReadScanline(ReadOnlySpan<byte> data, int pos, byte[] dst, int width)
    {
        if (pos + 4 > data.Length) throw new InvalidDataException("HDR: truncated file");

        // New-style RLE scanline: 0x02 0x02 then 16-bit width, then 4 component planes.
        if (data[pos] == 2 && data[pos + 1] == 2 && ((data[pos + 2] << 8) | data[pos + 3]) == width && width >= 8 && width < 32768)
        {
            pos += 4;
            for (int c = 0; c < 4; c++)
            {
                int x = 0;
                while (x < width)
                {
                    if (pos >= data.Length) throw new InvalidDataException("HDR: truncated RLE scanline");
                    int count = data[pos++];
                    if (count > 128)
                    {
                        // Run of a single value.
                        count -= 128;
                        if (x + count > width) throw new InvalidDataException("HDR: RLE run overflow");
                        if (pos >= data.Length) throw new InvalidDataException("HDR: truncated RLE scanline");
                        byte v = data[pos++];
                        for (int i = 0; i < count; i++) dst[(x + i) * 4 + c] = v;
                    }
                    else
                    {
                        // Literal span.
                        if (count == 0 || x + count > width) throw new InvalidDataException("HDR: bad RLE literal");
                        if (pos + count > data.Length) throw new InvalidDataException("HDR: truncated RLE scanline");
                        for (int i = 0; i < count; i++) dst[(x + i) * 4 + c] = data[pos++];
                    }
                    x += count;
                }
            }
            return pos;
        }

        // Flat / old-style RLE: pixels in RGBE order; (1,1,1,n) repeats the previous pixel.
        int px = 0, shift = 0;
        while (px < width)
        {
            if (pos + 4 > data.Length) throw new InvalidDataException("HDR: truncated scanline");
            byte r = data[pos], g = data[pos + 1], b = data[pos + 2], e = data[pos + 3];
            pos += 4;
            if (r == 1 && g == 1 && b == 1)
            {
                // Old-style run: repeat previous pixel e<<shift times.
                if (px == 0) throw new InvalidDataException("HDR: old-style run with no previous pixel");
                int count = e << shift;
                if (px + count > width) throw new InvalidDataException("HDR: old-style run overflow");
                int prev = (px - 1) * 4;
                for (int i = 0; i < count; i++) Array.Copy(dst, prev, dst, (px + i) * 4, 4);
                px += count;
                shift += 8;
            }
            else
            {
                dst[px * 4] = r;
                dst[px * 4 + 1] = g;
                dst[px * 4 + 2] = b;
                dst[px * 4 + 3] = e;
                px++;
                shift = 0;
            }
        }
        return pos;
    }

    private static float SrgbToLinear(float v) => v <= 0.04045f ? v / 12.92f : MathF.Pow((v + 0.055f) / 1.055f, 2.4f);

    private static float LinearToSrgb(float v) => v <= 0.0031308f ? v * 12.92f : 1.055f * MathF.Pow(v, 1 / 2.4f) - 0.055f;

    /// <summary>
    /// Expand an R8G8B8A8 image to linear R32G32B32A32_FLOAT (sRGB-decoding the color
    /// channels when <paramref name="srgb"/>), e.g. to feed an LDR source to the BC6H encoder.
    /// </summary>
    public static Image Rgba8ToLinearFloat(Image src, bool srgb = true)
    {
        if (!src.IsRgba8) throw new ArgumentException("Rgba8ToLinearFloat: source must be R8G8B8A8_UNORM(_SRGB)");
        Span<float> lut = stackalloc float[256];
        for (int i = 0; i < 256; i++) lut[i] = srgb ? SrgbToLinear(i / 255f) : i / 255f;
        var lutArr = lut.ToArray();

        var img = Image.Create(DxgiFormat.R32G32B32A32_FLOAT, src.Width, src.Height);
        Parallel.For(0, src.Height, y =>
        {
            var outF = img.Floats;
            int row = y * src.RowPitch;
            int o = y * src.Width * 4;
            for (int x = 0; x < src.Width; x++)
            {
                int i = row + x * 4;
                outF[o++] = lutArr[src.Pixels[i]];
                outF[o++] = lutArr[src.Pixels[i + 1]];
                outF[o++] = lutArr[src.Pixels[i + 2]];
                outF[o++] = src.Pixels[i + 3] / 255f;
            }
        });
        return img;
    }

    /// <summary>
    /// Tonemap a linear float image to displayable R8G8B8A8 (Reinhard + sRGB encode) —
    /// used for previews of HDR/BC6H content. Not part of the conversion pipeline.
    /// </summary>
    public static Image TonemapToRgba8(Image src)
    {
        if (src.Format != DxgiFormat.R32G32B32A32_FLOAT)
            throw new ArgumentException("TonemapToRgba8: source must be R32G32B32A32_FLOAT");
        var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, src.Width, src.Height);
        var f = src.Floats;
        for (int i = 0; i < f.Length; i += 4)
        {
            for (int c = 0; c < 3; c++)
            {
                float lin = MathF.Max(0, f[i + c]);
                float s = LinearToSrgb(lin / (1 + lin)); // Reinhard
                img.Pixels[i + c] = (byte)MathF.Floor(Math.Clamp(s, 0, 1) * 255 + 0.5f);
            }
            img.Pixels[i + 3] = (byte)MathF.Floor(Math.Clamp(f[i + 3], 0, 1) * 255 + 0.5f);
        }
        return img;
    }
}
