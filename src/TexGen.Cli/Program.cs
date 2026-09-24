//-------------------------------------------------------------------------------------
// texgen — texconv-compatible command-line front end.
//
//   texgen [texconv options] <files...>
//
// Accepts DirectXTex texconv's flags (see TexconvArgs); unsupported ones are reported
// as warnings. Inputs may use wildcards (with -r to recurse) or come from -flist.
// Like texconv, outputs go to the current directory unless -o is given, and existing
// files are only replaced with -y.
//-------------------------------------------------------------------------------------

using System.Diagnostics;
using TexGen;
using TexGen.Gpu;

return Cli.Run(args);

internal static class Cli
{
    private const string Usage = """
        Usage: texgen <options> <files>

           -r                  wildcard filename search is recursive
           -flist <filename>   use text file with a list of input files (one per line)
           -w <n>              width
           -h <n>              height
           -m <n>              miplevels (0 = full chain, default 1)
           -f <format>         format (e.g. BC7_UNORM, BC1_UNORM, DXT5, BC6H_UF16, RGBA, BGRA)
           -if <filter>        image filtering (box-filtered, sRGB-aware)
           -srgb{i|o}          sRGB {input, output}
           -px <string>        name prefix
           -sx <string>        name suffix
           -o <directory>      output directory
           -l                  force output filename to lower case
           -y                  overwrite existing output file
           -ft <filetype>      output file type (dds, png, jpg, webp)
           -hflip              horizonal flip of source image
           -vflip              vertical flip of source image
           -c <hex-RGB>        colorkey (a.k.a. chromakey) transparency
           -pmalpha            convert final texture to use premultiplied alpha
           -alpha              convert premultiplied alpha to straight alpha
           -swizzle <rgba>     swizzle image channels using HLSL-style mask
           -aw <weight>        BC7 GPU compressor weighting for alpha error metric
           -bc <q|x|d|u>       BC7 q = quick, x = 3-subset modes; CPU BC1-3 d = dither, u = uniform
           -at <threshold>     BC1 alpha threshold (CPU codec, default 0.5)
           -nologo             suppress copyright message
           -timing             display elapsed processing time
           -gpu <adapter>      select GPU adapter index
           -nogpu              do not use the GPU (DirectXTex CPU codec)

        Input formats: png jpg bmp gif webp ico wbmp hdr dds
        """;

    public static int Run(string[] args)
    {
        var cmd = TexconvArgs.Parse(args);
        if (!cmd.NoLogo)
        {
            Console.WriteLine("texgen — GPU texture converter (texconv-compatible), .NET port of texconv-js");
            Console.WriteLine();
        }

        foreach (var w in cmd.Warnings) Console.Error.WriteLine($"WARNING: {w}");

        List<string> inputs;
        try
        {
            inputs = ExpandInputs(cmd);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"ERROR: {e.Message}");
            return 1;
        }

        if (inputs.Count == 0)
        {
            if (cmd.Files.Count == 0 && cmd.FileList is null)
            {
                Console.WriteLine(Usage);
                return 0;
            }
            Console.Error.WriteLine("ERROR: No matching input files found");
            return 1;
        }

        if (cmd.Output.OutputDir is { } outDir && !Directory.Exists(outDir))
        {
            Console.Error.WriteLine($"ERROR: Output directory '{outDir}' does not exist");
            return 1;
        }

        GpuDevice? device = null;
        int failures = 0;
        var total = Stopwatch.StartNew();
        try
        {
            foreach (var input in inputs)
            {
                if (!ConvertFile(input, cmd, ref device)) failures++;
            }
        }
        finally
        {
            device?.Dispose();
        }

        if (cmd.Timing) Console.WriteLine($"Processing time: {total.Elapsed.TotalSeconds:F3} seconds");
        return failures == 0 ? 0 : 1;
    }

    private static GpuDevice GetDevice(ParsedCommand cmd, ref GpuDevice? device)
    {
        if (device is not null) return device;
        device = GpuDevice.Create(cmd.NoGpu ? GpuBackend.Cpu : GpuBackend.Auto, cmd.GpuAdapter);
        if (!cmd.NoLogo) Console.WriteLine($"Using {(device.IsHardware ? "GPU" : "CPU")} accelerator: {device.Description}");
        return device;
    }

    private static bool ConvertFile(string input, ParsedCommand cmd, ref GpuDevice? device)
    {
        var sw = Stopwatch.StartNew();
        bool lineOpen = false;
        try
        {
            var image = ImageIO.Load(input);
            var options = cmd.Options;
            var gpu = NeedsGpu(image, options) ? GetDevice(cmd, ref device) : null;
            Console.Write($"reading {input} ({image.Width}x{image.Height} {image.Format})");
            lineOpen = true;

            var scratch = Converter.Convert(image, options, gpu);
            var meta = scratch.Metadata;
            Console.WriteLine($" as ({meta.Width}x{meta.Height},{meta.MipLevels} {meta.Format})");
            lineOpen = false;

            var name = TexconvArgs.OutputFilename(input, cmd.Output);
            var outPath = cmd.Output.OutputDir is { } dir ? Path.Combine(dir, name) : name;
            if (File.Exists(outPath) && !cmd.Overwrite)
            {
                Console.Error.WriteLine($"ERROR: Output file already exists, use -y to overwrite: '{outPath}'");
                return false;
            }

            var ext = Path.GetExtension(outPath).ToLowerInvariant();
            if (ext == ".dds")
            {
                var alphaMode = options.PremultiplyAlpha ? DdsAlphaMode.Premultiplied : DdsAlphaMode.Unknown;
                File.WriteAllBytes(outPath, Dds.Write(scratch, alphaMode));
            }
            else
            {
                ImageIO.SaveImage(scratch.BaseImage, outPath);
            }

            Console.Write($"writing {outPath}");
            if (cmd.Timing) Console.Write($" ({sw.Elapsed.TotalMilliseconds:F1} ms)");
            Console.WriteLine();
            return true;
        }
        catch (Exception e)
        {
            if (lineOpen) Console.WriteLine();
            Console.Error.WriteLine($"ERROR: {input}: {e.Message}");
            return false;
        }
    }

    private static bool NeedsGpu(Image image, ConvertOptions o) =>
        (Dxgi.IsCompressed(o.Format) && o.Codec != CodecPreference.Cpu) || o.MipLevels is not 1 and not null
        || (o.Width is > 0 && o.Width != image.Width) || (o.Height is > 0 && o.Height != image.Height);

    /// <summary>Resolve positional arguments (with wildcards / -r) and -flist into input paths.</summary>
    internal static List<string> ExpandInputs(ParsedCommand cmd)
    {
        var result = new List<string>();
        var patterns = new List<string>(cmd.Files);
        if (cmd.FileList is { } list)
        {
            foreach (var raw in File.ReadAllLines(list))
            {
                var line = raw.Trim();
                // texconv -flist skips blank lines and '#' comments.
                if (line.Length > 0 && line[0] != '#') patterns.Add(line);
            }
        }

        foreach (var pattern in patterns)
        {
            var fileName = Path.GetFileName(pattern);
            if (fileName.IndexOfAny(['*', '?']) < 0)
            {
                if (!File.Exists(pattern)) throw new FileNotFoundException($"Input file not found: '{pattern}'");
                result.Add(pattern);
                continue;
            }
            var dir = Path.GetDirectoryName(pattern);
            if (string.IsNullOrEmpty(dir)) dir = ".";
            var opts = new EnumerationOptions
            {
                RecurseSubdirectories = cmd.Recursive,
                MatchCasing = MatchCasing.CaseInsensitive,
            };
            var matches = Directory.EnumerateFiles(dir, fileName, opts)
                .Where(ImageIO.IsSupported)
                .Select(p => dir == "." && p.StartsWith("./", StringComparison.Ordinal) ? p[2..] : p)
                .OrderBy(p => p, StringComparer.Ordinal);
            result.AddRange(matches);
        }
        return result;
    }
}
