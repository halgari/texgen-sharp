//-------------------------------------------------------------------------------------
// TexconvArgs.cs
//
// Parses the texconv command-line option surface into ConvertOptions plus the
// file-system/output settings the texgen CLI acts on, so the flags users know from
// texconv drive texgen unchanged. Unsupported texconv options are accepted (with
// their values consumed) and reported as warnings rather than errors.
//-------------------------------------------------------------------------------------

using System.Globalization;

namespace TexGen;

/// <summary>Output filename pieces (texconv -px/-sx/-o/-l/-ft).</summary>
public sealed record OutputNaming
{
    public string Prefix { get; init; } = "";
    public string Suffix { get; init; } = "";
    public string? OutputDir { get; init; }
    public bool ToLower { get; init; }
    public string? FileType { get; init; }
}

public sealed record ParsedCommand
{
    public required ConvertOptions Options { get; init; }
    public required OutputNaming Output { get; init; }
    /// <summary>Positional input file arguments (may contain wildcards).</summary>
    public required IReadOnlyList<string> Files { get; init; }
    /// <summary>Non-fatal notes about ignored/unsupported flags.</summary>
    public required IReadOnlyList<string> Warnings { get; init; }

    /// <summary>-r: search subdirectories for wildcard inputs.</summary>
    public bool Recursive { get; init; }
    /// <summary>-flist: text file listing input files, one per line.</summary>
    public string? FileList { get; init; }
    /// <summary>-y: overwrite existing output files.</summary>
    public bool Overwrite { get; init; }
    public bool NoLogo { get; init; }
    /// <summary>-timing: report per-file processing time.</summary>
    public bool Timing { get; init; }
    /// <summary>-gpu N: GPU adapter index.</summary>
    public int GpuAdapter { get; init; }
    /// <summary>-nogpu: run all kernels on the ILGPU CPU accelerator.</summary>
    public bool NoGpu { get; init; }
    /// <summary>-dx10 / -dx9: DDS header preference (texgen always writes DX10).</summary>
    public bool Dx9 { get; init; }
}

public static class TexconvArgs
{
    // Flags that consume the following token as their value.
    private static readonly HashSet<string> ValueFlags =
    [
        "f", "w", "h", "m", "aw", "bc", "if", "px", "sx", "o", "ft", "gpu", "flist", "c", "swizzle",
        "fl", "wicq", "at", "nmapamp", "nits", "d", "rotatecolor", "nmap", "keepcoverage",
    ];

    // Value-consuming texconv flags accepted (so their argument isn't mistaken for an
    // input file) but with no effect in texgen.
    private static readonly HashSet<string> IgnoredValueFlags =
        ["fl", "wicq", "nmapamp", "nits", "d", "rotatecolor", "nmap", "keepcoverage"];

    // Boolean texconv flags that change nothing in texgen.
    private static readonly HashSet<string> IgnoredBoolFlags =
    [
        "singleproc", "nowic", "wiclossless", "wicmulti", "sepalpha", "fixbc4x4", "permissive", "ignoremips",
        "dword", "badtails", "xlum", "tonemap", "inverty", "reconstructz", "x2bias", "tu", "tf",
    ];

    /// <summary>Parse texconv-style argv (without the program name).</summary>
    public static ParsedCommand Parse(IReadOnlyList<string> argv)
    {
        var o = new ConvertOptions();
        var output = new OutputNaming();
        var files = new List<string>();
        var warnings = new List<string>();
        bool srgbOut = false, recursive = false, overwrite = false, noLogo = false, timing = false, noGpu = false, dx9 = false;
        string? fileList = null;
        int gpuAdapter = 0;

        // Parse a numeric flag value, warning (and returning null) on a missing or
        // non-numeric argument so it doesn't silently become garbage.
        double? Num(string? raw, string label)
        {
            if (!string.IsNullOrEmpty(raw) && double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var n))
                return n;
            warnings.Add($"Invalid {label} '{raw ?? ""}' (expected a number)");
            return null;
        }

        for (int i = 0; i < argv.Count; i++)
        {
            var tok = argv[i];
            if (tok.Length == 0) continue;
            if (tok[0] != '-' && tok[0] != '/' || tok.Length == 1)
            {
                files.Add(tok);
                continue;
            }
            // A leading '/' could be an absolute Unix path rather than a /flag.
            var rawName = tok[1..];
            var name = rawName.ToLowerInvariant();
            if (tok[0] == '/' && (rawName.Contains('/') || File.Exists(tok)))
            {
                files.Add(tok);
                continue;
            }
            // texconv also accepts "-name:value" for value flags.
            string? inlineValue = null;
            int colon = name.IndexOf(':');
            if (colon > 0 && ValueFlags.Contains(name[..colon]))
            {
                inlineValue = rawName[(colon + 1)..];
                name = name[..colon];
            }
            bool takesValue = ValueFlags.Contains(name);
            string? value = inlineValue ?? (takesValue && i + 1 < argv.Count ? argv[++i] : null);

            switch (name)
            {
                case "f":
                {
                    var fmt = value is null ? null : Dxgi.FormatFromName(value);
                    if (fmt is null) warnings.Add($"Unknown format '{value}'");
                    else o = o with { Format = fmt.Value };
                    break;
                }
                case "w": if (Num(value, "-w width") is { } w) o = o with { Width = (int)w }; break;
                case "h": if (Num(value, "-h height") is { } h) o = o with { Height = (int)h }; break;
                case "m": if (Num(value, "-m mip levels") is { } m) o = o with { MipLevels = (int)m }; break;
                case "at":
                    if (Num(value, "-at alpha threshold") is { } at) o = o with { AlphaThreshold = (float)at };
                    break;
                case "aw": if (Num(value, "-aw alpha weight") is { } aw) o = o with { AlphaWeight = (float)aw }; break;
                // -srgbi: treat INPUT as sRGB (filtering); -srgbo: tag OUTPUT format as
                // sRGB; -srgb: both — mirroring upstream texconv.
                case "srgb": o = o with { SrgbFilter = true }; srgbOut = true; break;
                case "srgbi": o = o with { SrgbFilter = true }; break;
                case "srgbo": srgbOut = true; break;
                case "hflip": o = o with { HFlip = true }; break;
                case "vflip": o = o with { VFlip = true }; break;
                case "swizzle":
                    try
                    {
                        if (value is not null) Transforms.ParseSwizzle(value);
                        o = o with { Swizzle = value };
                    }
                    catch (FormatException e)
                    {
                        warnings.Add(e.Message);
                    }
                    break;
                case "c":
                {
                    var hex = value?.StartsWith("0x", StringComparison.OrdinalIgnoreCase) == true ? value[2..] : value;
                    if (hex is not null && uint.TryParse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var key) && key <= 0xffffff)
                        o = o with { ColorKey = key };
                    else
                        warnings.Add($"Invalid colorkey '{value}' (expected hex RRGGBB)");
                    break;
                }
                case "pmalpha": o = o with { PremultiplyAlpha = true }; break;
                case "alpha": o = o with { StraightAlpha = true }; break;
                case "bc":
                    foreach (var ch in value ?? "")
                    {
                        switch (char.ToLowerInvariant(ch))
                        {
                            case 'q': o = o with { Bc7Quick = true }; break;
                            case 'x': o = o with { Bc7Use3Subsets = true }; break;
                            case 'd': o = o with { Dither = true }; break;
                            case 'u': o = o with { UniformWeighting = true }; break;
                            default: warnings.Add($"-bc '{ch}' not supported (ignored)"); break;
                        }
                    }
                    break;
                case "if":
                    if (value is not null && !IsBoxLikeFilter(value)) warnings.Add($"Filter '{value}' approximated by box");
                    break;
                case "px": output = output with { Prefix = value ?? "" }; break;
                case "sx": output = output with { Suffix = value ?? "" }; break;
                case "o": output = output with { OutputDir = value }; break;
                case "l": output = output with { ToLower = true }; break;
                case "ft": output = output with { FileType = value }; break;
                case "y": overwrite = true; break;
                case "r": recursive = true; break;
                case "flist": fileList = value; break;
                case "nologo": noLogo = true; break;
                case "timing": timing = true; break;
                case "nogpu":
                    noGpu = true;
                    o = o with { Codec = CodecPreference.Cpu };
                    break;
                case "gpu":
                    if (Num(value, "-gpu adapter") is { } g) gpuAdapter = (int)g;
                    break;
                case "dx9":
                    dx9 = true;
                    warnings.Add("-dx9 ignored: texgen always writes the DX10 DDS header");
                    break;
                case "dx10": break; // default
                default:
                    if (IgnoredValueFlags.Contains(name) || IgnoredBoolFlags.Contains(name))
                        warnings.Add($"-{name} has no effect in texgen (ignored)");
                    else
                        warnings.Add($"Unrecognized option -{name}{(takesValue ? $" {value}" : "")}");
                    break;
            }
        }

        // -srgbo promotes the output format to its sRGB variant (after the whole argv is
        // parsed, since -f may appear in any position).
        if (srgbOut) o = o with { Format = Dxgi.ToSrgb(o.Format) };
        if (o.PremultiplyAlpha && o.StraightAlpha)
        {
            warnings.Add("-pmalpha and -alpha are mutually exclusive; ignoring -alpha");
            o = o with { StraightAlpha = false };
        }

        return new ParsedCommand
        {
            Options = o,
            Output = output,
            Files = files,
            Warnings = warnings,
            Recursive = recursive,
            FileList = fileList,
            Overwrite = overwrite,
            NoLogo = noLogo,
            Timing = timing,
            GpuAdapter = gpuAdapter,
            NoGpu = noGpu,
            Dx9 = dx9,
        };
    }

    // texconv's -if names; texgen filters with a (sRGB-aware) box filter, which is what
    // texconv's POINT/LINEAR/BOX/FANT reduce to for the 2:1 mip case.
    private static bool IsBoxLikeFilter(string value)
    {
        var v = value.ToUpperInvariant();
        foreach (var suffix in new[] { "_DITHER_DIFFUSION", "_DITHER" })
            if (v.EndsWith(suffix, StringComparison.Ordinal)) v = v[..^suffix.Length];
        return v is "POINT" or "LINEAR" or "BOX" or "FANT";
    }

    /// <summary>Build an output filename from the parsed naming options (DDS unless -ft).</summary>
    public static string OutputFilename(string inputName, OutputNaming output)
    {
        var baseName = Path.GetFileNameWithoutExtension(inputName);
        if (output.ToLower) baseName = baseName.ToLowerInvariant();
        var ext = string.IsNullOrEmpty(output.FileType) ? "dds" : output.FileType.TrimStart('.');
        if (output.ToLower) ext = ext.ToLowerInvariant();
        return $"{output.Prefix}{baseName}{output.Suffix}.{ext}";
    }
}
