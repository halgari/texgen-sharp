//-------------------------------------------------------------------------------------
// TexGen.Bench — end-to-end conversion throughput.
//
//   dotnet run -c Release --project bench/TexGen.Bench -- [--sizes 1024,4096]
//       [--formats BC1_UNORM,BC7_UNORM] [--iters 5] [--mips] [--cpu] [--image path]
//
// For every format: one cold Convert (includes ILGPU kernel compilation), then
// --iters warm Converts timed end-to-end (upload -> resize/mips -> encode -> download).
// Reports median milliseconds and megapixels/second of source image.
//-------------------------------------------------------------------------------------

using System.Diagnostics;
using TexGen;
using TexGen.Gpu;

var sizes = new List<int> { 1024, 2048, 4096 };
var formats = new List<DxgiFormat>
{
    DxgiFormat.BC1_UNORM, DxgiFormat.BC3_UNORM, DxgiFormat.BC4_UNORM, DxgiFormat.BC5_UNORM,
    DxgiFormat.BC7_UNORM, DxgiFormat.BC6H_UF16,
};
int iters = 5;
bool mips = false, cpu = false, quick = false, threeSubsets = false;
string? imagePath = null;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--sizes": sizes = args[++i].Split(',').Select(int.Parse).ToList(); break;
        case "--formats": formats = args[++i].Split(',').Select(f => Dxgi.FormatFromName(f) ?? throw new ArgumentException(f)).ToList(); break;
        case "--iters": iters = int.Parse(args[++i]); break;
        case "--mips": mips = true; break;
        case "--cpu": cpu = true; break;
        case "--quick": quick = true; break;
        case "--3subsets": threeSubsets = true; break;
        case "--image": imagePath = args[++i]; break;
        default: throw new ArgumentException($"unknown argument {args[i]}");
    }
}

using var device = GpuDevice.Create(cpu ? GpuBackend.Cpu : GpuBackend.Auto);
Console.WriteLine($"Device: {device.Description}");
Console.WriteLine($"Mips: {(mips ? "full chain" : "base only")}, iterations: {iters}{(quick ? ", BC7 quick" : "")}{(threeSubsets ? ", BC7 3-subsets" : "")}");
Console.WriteLine();
Console.WriteLine($"{"format",-12} {"size",-11} {"cold ms",9} {"warm ms",9} {"MPix/s",9}");

Image? loaded = imagePath is null ? null : ImageIO.Load(imagePath);

foreach (var format in formats)
{
    bool cold = true;
    foreach (int size in sizes)
    {
        var src = loaded ?? MakeTestImage(size, size);
        if (loaded is not null && size != sizes[0]) break; // a real image has one size
        int w = src.Width, h = src.Height;
        var options = new ConvertOptions
        {
            Format = format,
            MipLevels = mips ? 0 : 1,
            Bc7Quick = quick,
            Bc7Use3Subsets = threeSubsets,
        };

        double coldMs = double.NaN;
        if (cold)
        {
            var sw = Stopwatch.StartNew();
            Converter.Convert(src, options, device);
            coldMs = sw.Elapsed.TotalMilliseconds;
            cold = false;
        }
        else
        {
            Converter.Convert(src, options, device); // warm-up for this size
        }

        var samples = new double[iters];
        for (int it = 0; it < iters; it++)
        {
            var sw = Stopwatch.StartNew();
            Converter.Convert(src, options, device);
            samples[it] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        double median = samples[iters / 2];
        double mpix = w * (double)h / 1e6 / (median / 1000);
        Console.WriteLine($"{format,-12} {$"{w}x{h}",-11} {(double.IsNaN(coldMs) ? "" : coldMs.ToString("F0")),9} {median,9:F1} {mpix,9:F0}");
    }
}

// Photo-like content: smooth gradients plus hard edges and noise, so every encoder
// mode gets exercised (a pure gradient is unrealistically easy for BC7/BC6H).
static Image MakeTestImage(int w, int h)
{
    var img = Image.Create(DxgiFormat.R8G8B8A8_UNORM, w, h);
    var rng = new Random(1234);
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int o = y * img.RowPitch + x * 4;
            bool cell = ((x / 37) + (y / 53)) % 3 == 0;
            int noise = rng.Next(-12, 13);
            img.Pixels[o] = (byte)Math.Clamp(x * 255 / w + noise, 0, 255);
            img.Pixels[o + 1] = (byte)Math.Clamp((cell ? 220 : y * 255 / h) + noise, 0, 255);
            img.Pixels[o + 2] = (byte)Math.Clamp(((x ^ y) & 0xff) / 2 + noise, 0, 255);
            img.Pixels[o + 3] = (byte)(cell ? 128 : 255);
        }
    }
    return img;
}
