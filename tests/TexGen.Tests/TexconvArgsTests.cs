namespace TexGen.Tests;

public class TexconvArgsTests
{
    private static ParsedCommand Parse(string args) => TexconvArgs.Parse(args.Split(' ', StringSplitOptions.RemoveEmptyEntries));

    [Fact]
    public void FormatFromNameResolvesCanonicalNamesAndAliases()
    {
        Assert.Equal(DxgiFormat.BC7_UNORM, Dxgi.FormatFromName("BC7_UNORM"));
        Assert.Equal(DxgiFormat.BC3_UNORM, Dxgi.FormatFromName("bc3_unorm"));
        Assert.Equal(DxgiFormat.BC3_UNORM, Dxgi.FormatFromName("DXT5"));
        Assert.Equal(DxgiFormat.BC1_UNORM, Dxgi.FormatFromName("DXT1"));
        Assert.Equal(DxgiFormat.BC7_UNORM, Dxgi.FormatFromName("DXGI_FORMAT_BC7_UNORM"));
        Assert.Equal(DxgiFormat.BC7_UNORM, Dxgi.FormatFromName("BPTC"));
        Assert.Equal(DxgiFormat.R16G16B16A16_FLOAT, Dxgi.FormatFromName("FP16"));
        Assert.Null(Dxgi.FormatFromName("nope"));
        Assert.Null(Dxgi.FormatFromName("98"));
    }

    [Fact]
    public void ParsesATypicalCommand()
    {
        var cmd = Parse("-f BC7_UNORM -m 0 -w 256 -h 256 -aw 2 texture.png");
        Assert.Equal(DxgiFormat.BC7_UNORM, cmd.Options.Format);
        Assert.Equal(0, cmd.Options.MipLevels);
        Assert.Equal(256, cmd.Options.Width);
        Assert.Equal(256, cmd.Options.Height);
        Assert.Equal(2f, cmd.Options.AlphaWeight);
        Assert.Equal(["texture.png"], cmd.Files);
    }

    [Fact]
    public void MapsBcFlagsAndSrgb()
    {
        var o = Parse("-f BC7_UNORM -bc qx -srgb").Options;
        Assert.True(o.Bc7Quick);
        Assert.True(o.Bc7Use3Subsets);
        Assert.True(o.SrgbFilter);
    }

    [Fact]
    public void CollectsWarningsForUnsupportedFlags()
    {
        var w = Parse("-f DXT5 -r -bc z -zzz").Warnings;
        Assert.Contains(w, x => x.Contains("'z'"));
        Assert.Contains(w, x => x.Contains("Unrecognized", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void CpuCodecFlags()
    {
        var cmd = Parse("-f BC1_UNORM -bc du -at 0.25 -nogpu");
        Assert.True(cmd.Options.Dither);
        Assert.True(cmd.Options.UniformWeighting);
        Assert.Equal(0.25f, cmd.Options.AlphaThreshold);
        Assert.Equal(CodecPreference.Cpu, cmd.Options.Codec);
        Assert.Equal(TexGen.Cpu.BcFlags.DitherRgb | TexGen.Cpu.BcFlags.DitherA | TexGen.Cpu.BcFlags.Uniform, cmd.Options.CpuFlags);
        Assert.Empty(cmd.Warnings);
    }

    [Fact]
    public void FileSystemFlagsAreHonored()
    {
        var cmd = Parse("-r -y -o out -flist list.txt -nologo -timing -gpu 1 -nogpu *.png");
        Assert.True(cmd.Recursive);
        Assert.True(cmd.Overwrite);
        Assert.Equal("out", cmd.Output.OutputDir);
        Assert.Equal("list.txt", cmd.FileList);
        Assert.True(cmd.NoLogo);
        Assert.True(cmd.Timing);
        Assert.Equal(1, cmd.GpuAdapter);
        Assert.True(cmd.NoGpu);
        Assert.Equal(["*.png"], cmd.Files);
        Assert.Empty(cmd.Warnings);
    }

    [Fact]
    public void BuildsOutputFilenamesWithPrefixSuffixLowercase()
    {
        var output = Parse("-px pre_ -sx _suf -l").Output;
        Assert.Equal("pre_mytex_suf.dds", TexconvArgs.OutputFilename("MyTex.PNG", output));
    }

    [Fact]
    public void HonorsFtForTheOutputExtension()
    {
        Assert.Equal("tex.tga", TexconvArgs.OutputFilename("tex.png", Parse("-ft tga").Output));
        Assert.Equal("tex.png", TexconvArgs.OutputFilename("tex.dds", Parse("-ft .png").Output));
    }

    [Fact]
    public void WarnsOnMissingOrNonNumericValues()
    {
        var cmd = Parse("-w abc -h 64 -m");
        Assert.Null(cmd.Options.Width);
        Assert.Equal(64, cmd.Options.Height);
        Assert.Null(cmd.Options.MipLevels);
        Assert.Contains(cmd.Warnings, w => w.Contains("-w width"));
        Assert.Contains(cmd.Warnings, w => w.Contains("-m mip levels"));
    }

    [Fact]
    public void ValueConsumingPassthroughFlagsAreIgnoredNotUnrecognized()
    {
        var cmd = Parse("-nits 100 -fl 12.1 in.png");
        Assert.Equal(["in.png"], cmd.Files); // values consumed, not treated as inputs
        Assert.DoesNotContain(cmd.Warnings, w => w.Contains("Unrecognized", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(cmd.Warnings, w => w.Contains("no effect"));
    }

    [Fact]
    public void AcceptsColonValueSyntaxAndSlashFlags()
    {
        var cmd = Parse("/f:BC1_UNORM /m 3 /home/user/in.png");
        Assert.Equal(DxgiFormat.BC1_UNORM, cmd.Options.Format);
        Assert.Equal(3, cmd.Options.MipLevels);
        Assert.Equal(["/home/user/in.png"], cmd.Files);
    }

    [Fact]
    public void SrgbFlagSemantics()
    {
        var i = Parse("-f BC3_UNORM -srgbi").Options;
        Assert.True(i.SrgbFilter);
        Assert.Equal(DxgiFormat.BC3_UNORM, i.Format);

        var o = Parse("-f BC3_UNORM -srgbo").Options;
        Assert.Null(o.SrgbFilter);
        Assert.Equal(DxgiFormat.BC3_UNORM_SRGB, o.Format);

        var both = Parse("-srgb -f BC7_UNORM").Options; // -f after, still promoted
        Assert.True(both.SrgbFilter);
        Assert.Equal(DxgiFormat.BC7_UNORM_SRGB, both.Format);
    }

    [Fact]
    public void ParsesTheTransformFlags()
    {
        var cmd = Parse("-f BC7_UNORM -hflip -vflip -swizzle abgr -c 00FF00 -pmalpha");
        Assert.True(cmd.Options.HFlip);
        Assert.True(cmd.Options.VFlip);
        Assert.Equal("abgr", cmd.Options.Swizzle);
        Assert.Equal(0x00ff00u, cmd.Options.ColorKey);
        Assert.True(cmd.Options.PremultiplyAlpha);
        Assert.Empty(cmd.Warnings);
    }

    [Fact]
    public void WarnsOnBadSwizzleColorkeyAndPmalphaAlphaConflict()
    {
        var cmd = Parse("-swizzle rq -c zzz -pmalpha -alpha");
        Assert.Contains(cmd.Warnings, w => w.Contains("invalid mask"));
        Assert.Contains(cmd.Warnings, w => w.Contains("colorkey"));
        Assert.Contains(cmd.Warnings, w => w.Contains("mutually exclusive"));
        Assert.False(cmd.Options.StraightAlpha);
    }
}
