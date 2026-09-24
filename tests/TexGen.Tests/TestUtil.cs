//-------------------------------------------------------------------------------------
// Shared test helpers: synthetic images, PSNR metrics, and the GPU device fixture.
//-------------------------------------------------------------------------------------

using TexGen.Gpu;

namespace TexGen.Tests;

public static class TestImages
{
    /// <summary>RGBA gradient with a smooth alpha ramp (the texconv-js BC test image).</summary>
    public static Image Gradient(int width, int height, bool alpha = true)
    {
        var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, width, height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int o = y * img.RowPitch + x * 4;
                img.Pixels[o] = (byte)Math.Floor(x / (double)(width - 1) * 255);
                img.Pixels[o + 1] = (byte)Math.Floor(y / (double)(height - 1) * 255);
                img.Pixels[o + 2] = alpha
                    ? (byte)Math.Floor((x + y) / (double)(width + height - 2) * 255)
                    : (byte)128;
                img.Pixels[o + 3] = alpha ? (byte)Math.Floor((width - 1 - x) / (double)(width - 1) * 255) : (byte)255;
            }
        }
        return img;
    }

    public static Image Flat(int width, int height, byte r, byte g, byte b, byte a)
    {
        var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, width, height);
        for (int i = 0; i < width * height; i++)
        {
            img.Pixels[i * 4] = r;
            img.Pixels[i * 4 + 1] = g;
            img.Pixels[i * 4 + 2] = b;
            img.Pixels[i * 4 + 3] = a;
        }
        return img;
    }

    /// <summary>Smooth HDR gradient with values well above 1.0 (and below 0 when signed).</summary>
    public static Image HdrGradient(int width, int height, bool signed = false)
    {
        var img = Image.Create(DxgiFormat.R32G32B32A32_FLOAT, width, height);
        var f = img.Floats;
        int o = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float u = x / (float)(width - 1);
                float v = y / (float)(height - 1);
                f[o++] = u * 4.0f;
                f[o++] = 0.05f + v * 1.5f;
                f[o++] = signed ? (u - 0.5f) * 2.0f : (u + v) * 0.5f;
                f[o++] = 1;
            }
        }
        return img;
    }

    /// <summary>PSNR over the given RGBA8 channels.</summary>
    public static double Psnr(Image a, Image b, params int[] channels)
    {
        if (channels.Length == 0) channels = [0, 1, 2, 3];
        double sse = 0;
        long n = 0;
        for (int y = 0; y < a.Height; y++)
        {
            for (int x = 0; x < a.Width; x++)
            {
                int oa = y * a.RowPitch + x * 4, ob = y * b.RowPitch + x * 4;
                foreach (int c in channels)
                {
                    double d = a.Pixels[oa + c] - b.Pixels[ob + c];
                    sse += d * d;
                    n++;
                }
            }
        }
        return sse == 0 ? double.PositiveInfinity : 10 * Math.Log10(255.0 * 255.0 / (sse / n));
    }

    /// <summary>PSNR over RGB in linear float space, peak = max source component.</summary>
    public static double PsnrHdr(Image a, Image b)
    {
        var fa = a.Floats;
        var fb = b.Floats;
        double sse = 0, peak = 0;
        long n = 0;
        for (int i = 0; i < fa.Length; i += 4)
        {
            for (int c = 0; c < 3; c++)
            {
                double d = fa[i + c] - fb[i + c];
                sse += d * d;
                n++;
                peak = Math.Max(peak, Math.Abs(fa[i + c]));
            }
        }
        return sse == 0 ? double.PositiveInfinity : 10 * Math.Log10(peak * peak / (sse / n));
    }
}

/// <summary>
/// One GPU device shared by all GPU tests. Uses the best available backend; set
/// TEXGEN_BACKEND=cpu to run the same tests on the ILGPU CPU accelerator.
/// </summary>
public sealed class GpuFixture : IDisposable
{
    public GpuDevice Device { get; }

    public GpuFixture()
    {
        var backend = Environment.GetEnvironmentVariable("TEXGEN_BACKEND")?.ToLowerInvariant() switch
        {
            "cpu" => GpuBackend.Cpu,
            "cuda" => GpuBackend.Cuda,
            "opencl" => GpuBackend.OpenCL,
            _ => GpuBackend.Auto,
        };
        Device = GpuDevice.Create(backend);
    }

    public void Dispose() => Device.Dispose();
}

[CollectionDefinition("gpu")]
public sealed class GpuCollection : ICollectionFixture<GpuFixture>;
