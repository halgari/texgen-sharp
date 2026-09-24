//-------------------------------------------------------------------------------------
// Converter.cs
//
// High-level conversion entry point, modeling the texconv pipeline:
//   load -> CPU transforms -> (resize) -> (generate mips) -> (premultiply)
//        -> (convert/compress per level) -> ScratchImage
//
// Resize, mips, premultiply and block compression run on the GPU (ILGPU). The source
// is uploaded once and every stage stays device-resident; only the final encoded
// blocks come back to the host.
//-------------------------------------------------------------------------------------

using TexGen.Gpu;

namespace TexGen;

public sealed record ConvertOptions
{
    /// <summary>Target DXGI format (texconv -f).</summary>
    public DxgiFormat Format { get; init; } = DxgiFormat.R8G8B8A8_UNORM;
    /// <summary>Mip levels: 1 = base only (default), 0 = full chain, N = first N levels.</summary>
    public int? MipLevels { get; init; }
    /// <summary>Resize target (texconv -w / -h). Null keeps the source size.</summary>
    public int? Width { get; init; }
    public int? Height { get; init; }
    /// <summary>
    /// Treat the source data as sRGB when filtering (resize/mips). Defaults to the source
    /// format's sRGB-ness; set explicitly to override (texconv -srgbi).
    /// </summary>
    public bool? SrgbFilter { get; init; }
    /// <summary>BC7 alpha weighting (texconv -aw).</summary>
    public float? AlphaWeight { get; init; }
    /// <summary>BC7 quick mode — modes 4/5/6 only (texconv -bc q).</summary>
    public bool Bc7Quick { get; init; }
    /// <summary>BC7 also try 3-subset modes 0/2 (texconv -bc x).</summary>
    public bool Bc7Use3Subsets { get; init; }
    /// <summary>Mirror horizontally / vertically (texconv -hflip / -vflip).</summary>
    public bool HFlip { get; init; }
    public bool VFlip { get; init; }
    /// <summary>Channel swizzle mask, e.g. "abgr" or "rgb1" (texconv -swizzle).</summary>
    public string? Swizzle { get; init; }
    /// <summary>Colorkey 0xRRGGBB: matching pixels become transparent black (texconv -c).</summary>
    public uint? ColorKey { get; init; }
    /// <summary>Premultiply color by alpha after mip generation (texconv -pmalpha).</summary>
    public bool PremultiplyAlpha { get; init; }
    /// <summary>Undo premultiplied alpha on the source (texconv -alpha).</summary>
    public bool StraightAlpha { get; init; }
}

public static class Converter
{
    private static bool IsBc6h(DxgiFormat f) => f is DxgiFormat.BC6H_UF16 or DxgiFormat.BC6H_SF16;

    /// <summary>True for targets that are encoded from linear float data.</summary>
    private static bool IsFloatTarget(DxgiFormat f) => IsBc6h(f) || PixelConvert.IsFloatFormat(f);

    /// <summary>
    /// Convert a source Image to the requested format/shape. The source must be in the
    /// R8G8B8A8 family (what the image loader produces for LDR files) or linear
    /// R32G32B32A32_FLOAT (what the .hdr loader produces; float targets only).
    /// </summary>
    /// <param name="device">GPU device; defaults to <see cref="GpuDevice.Shared"/> when a GPU stage is needed.</param>
    public static ScratchImage Convert(Image source, ConvertOptions options, GpuDevice? device = null)
    {
        var format = options.Format;
        bool floatSource = source.Format == DxgiFormat.R32G32B32A32_FLOAT;
        if (!floatSource && !source.IsRgba8)
            throw new ArgumentException($"Convert: source must be R8G8B8A8(_SRGB) or R32G32B32A32_FLOAT; got {source.Format}");
        if (!Dxgi.IsCompressed(format) && !PixelConvert.CanEncode(format))
            throw new NotSupportedException($"Convert: output format {format} is not supported");
        if (Dxgi.IsCompressed(format) && !IsBc6h(format) && !BcnEncoder.Supports(format)
            && format is not (DxgiFormat.BC7_UNORM or DxgiFormat.BC7_UNORM_SRGB))
            throw new NotSupportedException($"Convert: GPU compression to {format} is not supported");
        if (floatSource && !IsFloatTarget(format))
            throw new ArgumentException($"Convert: HDR float source supports only BC6H and float targets; got {format}");

        // The swizzle/colorkey/premultiply transforms are RGBA8-only. Reject the
        // combinations that would otherwise fail deep inside a transform.
        bool rgba8OnlyTransform = !string.IsNullOrEmpty(options.Swizzle) || options.ColorKey.HasValue
            || options.PremultiplyAlpha || options.StraightAlpha;
        if (floatSource && rgba8OnlyTransform)
            throw new ArgumentException(
                "Convert: -swizzle/-c/-pmalpha/-alpha are not supported for HDR float sources (RGBA8-only transforms)");
        if (IsFloatTarget(format) && options.PremultiplyAlpha)
            throw new ArgumentException(
                $"Convert: -pmalpha is not supported for BC6H/float targets such as {format} (premultiply runs in linear float after expansion)");

        // CPU transforms, in upstream texconv's relative order (undo-pmalpha, flip,
        // swizzle, colorkey). Like texconv-js, these run before resize so the RGBA8 and
        // HDR-float paths share one code path — only filtered-edge pixels can differ.
        var src = source;
        if (options.StraightAlpha) src = Transforms.PremultiplyAlpha(src, undo: true);
        if (options.HFlip || options.VFlip) src = Transforms.Flip(src, options.HFlip, options.VFlip);
        if (!string.IsNullOrEmpty(options.Swizzle)) src = Transforms.Swizzle(src, options.Swizzle);
        if (options.ColorKey is { } key) src = Transforms.ColorKey(src, key);

        // BC6H / float targets are linear: expand 8-bit sources to linear float up front
        // so resize/mips/encode all run in linear light.
        if (!floatSource && IsFloatTarget(format)) src = Hdr.Rgba8ToLinearFloat(src, options.SrgbFilter ?? true);

        int dstW = options.Width is > 0 ? options.Width.Value : src.Width;
        int dstH = options.Height is > 0 ? options.Height.Value : src.Height;
        bool wantResize = dstW != src.Width || dstH != src.Height;
        int mipRequest = options.MipLevels ?? 1;
        int fullChain = ScratchImage.CountMips(dstW, dstH);
        int mipCount = mipRequest <= 0 ? fullChain : Math.Min(mipRequest, fullChain);
        bool srgbFilter = options.SrgbFilter ?? Dxgi.IsSrgb(src.Format);

        bool needGpu = wantResize || mipCount > 1 || Dxgi.IsCompressed(format);
        if (!needGpu)
        {
            // Pure CPU path: at most a premultiply + pixel format conversion.
            if (options.PremultiplyAlpha) src = Transforms.PremultiplyAlpha(src);
            return ScratchImage.From2D(PixelConvert.Encode(src, format));
        }

        device ??= GpuDevice.Shared;
        return ConvertOnGpu(device, src, options, dstW, dstH, mipCount, srgbFilter);
    }

    private static ScratchImage ConvertOnGpu(GpuDevice device, Image src, ConvertOptions options,
        int dstW, int dstH, int mipCount, bool srgbFilter)
    {
        var format = options.Format;
        var textures = new List<GpuTexture>(mipCount + 1);
        var blocks = new List<GpuBlocks>(mipCount);
        try
        {
            // 1. Upload + resize (if requested).
            var tex = GpuTexture.Upload(device, src);
            textures.Add(tex);
            if (dstW != src.Width || dstH != src.Height)
            {
                tex = device.ImageOps.Resize(tex, dstW, dstH, srgbFilter);
                textures.Add(tex);
            }

            // 2. Mip chain by repeated halving.
            var levels = new List<GpuTexture>(mipCount) { tex };
            for (int i = 1; i < mipCount; i++)
            {
                var prev = levels[^1];
                var next = device.ImageOps.Resize(prev, Math.Max(1, prev.Width >> 1), Math.Max(1, prev.Height >> 1), srgbFilter);
                textures.Add(next);
                levels.Add(next);
            }

            // 3. Premultiply after mip generation (upstream texconv's stage order).
            if (options.PremultiplyAlpha)
            {
                // The base level may still be the uploaded source; premultiplying in place is fine
                // because nothing reads the unpremultiplied data after mip generation.
                foreach (var level in levels) device.ImageOps.Premultiply(level);
            }

            // 4. Encode every level (all launches are queued before any download).
            if (!Dxgi.IsCompressed(format))
            {
                var outImages = levels.Select(l => PixelConvert.Encode(l.Download(), format)).ToList();
                return outImages.Count == 1 ? ScratchImage.From2D(outImages[0]) : ScratchImage.FromMipChain(outImages);
            }

            foreach (var level in levels) blocks.Add(Encode(device, level, options));
            var images = blocks.Select(b => b.Download()).ToList();
            device.Accelerator.Synchronize();
            return images.Count == 1 ? ScratchImage.From2D(images[0]) : ScratchImage.FromMipChain(images);
        }
        finally
        {
            foreach (var b in blocks) b.Dispose();
            foreach (var t in textures) t.Dispose();
        }
    }

    /// <summary>Route a compressed-target conversion to the appropriate GPU encoder.</summary>
    private static GpuBlocks Encode(GpuDevice device, GpuTexture level, ConvertOptions options) => options.Format switch
    {
        DxgiFormat.BC6H_UF16 or DxgiFormat.BC6H_SF16 => device.Bc6h.Encode(level, options.Format),
        DxgiFormat.BC7_UNORM or DxgiFormat.BC7_UNORM_SRGB => device.Bc7.Encode(level, options.Format,
            options.AlphaWeight ?? 1f, options.Bc7Quick, options.Bc7Use3Subsets),
        _ => device.Bcn.Encode(level, options.Format),
    };
}
