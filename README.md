# texgen-sharp

A C# / **.NET 10** port of [texconv-js](https://github.com/halgari/texconv-js), itself a
port of Microsoft [DirectXTex](https://github.com/microsoft/DirectXTex)'s `texconv`
texture converter. Resize, mip generation and all block compression (BC1–BC7,
including BC6H HDR) run on the GPU through [ILGPU](https://ilgpu.net): the kernels
are plain C#, JIT-compiled to PTX for NVIDIA (CUDA), to OpenCL for other GPUs, and
runnable on ILGPU's CPU accelerator when no GPU is available.

The BC6H/BC7 encoders are faithful ports of DirectXTex's `BC6HEncode.hlsl` /
`BC7Encode.hlsl` DirectCompute shaders (via texconv-js's WGSL translation),
including the same multi-pass mode search with ping-pong error buffers.

## Quick start

```bash
dotnet build -c Release
# texconv-style command line:
dotnet run -c Release --project src/TexGen.Cli -- -f BC7_UNORM -m 0 -srgb -o out textures/*.png
```

The `texgen` executable accepts texconv's flags, so existing scripts and tools can
call it unchanged:

```
texgen -f BC3_UNORM -m 0 -y -o out *.png
texgen -f BC6H_UF16 sky.hdr
texgen -f DXT1 -w 512 -h 512 -px small_ -l Diffuse.PNG
```

Library use:

```csharp
using TexGen;
using TexGen.Gpu;

using var gpu = GpuDevice.Create();                 // CUDA -> OpenCL -> CPU
var image = ImageIO.Load("albedo.png");             // RGBA8 (sRGB color-managed)
var dds = Converter.Convert(image, new ConvertOptions
{
    Format = DxgiFormat.BC7_UNORM_SRGB,
    MipLevels = 0,                                   // full chain
    SrgbFilter = true,
}, gpu);
File.WriteAllBytes("albedo.dds", Dds.Write(dds));
```

## Requirements

- .NET 10 SDK
- For GPU acceleration: an NVIDIA GPU with a current driver (CUDA; no CUDA
  toolkit needed, ILGPU emits PTX directly), or an OpenCL 2.0+ device.
  Without one, everything runs on the ILGPU CPU accelerator (`-nogpu` forces it).

## Formats

| | |
|---|---|
| **Input** | PNG, JPEG, BMP, GIF, WebP, ICO, WBMP (SkiaSharp, color-managed to sRGB like the browser), Radiance `.hdr`, DDS (BC1–BC7, BC6H and the uncompressed formats below) |
| **Output (GPU-compressed)** | BC1, BC2, BC3 (+ `_SRGB`), BC4_UNORM, BC5_UNORM, BC6H_UF16, BC6H_SF16, BC7 (+ `_SRGB`) |
| **Output (uncompressed)** | R8G8B8A8 (+ `_SRGB`), B8G8R8A8 (+ `_SRGB`), B8G8R8X8, R16G16B16A16_UNORM/FLOAT, R32G32B32A32_FLOAT, R32_FLOAT, R16_UNORM, R8_UNORM, R8G8_UNORM |
| **Containers** | DDS (DX10 header), plus PNG/JPEG/WebP via `-ft` |

## Supported texconv options

| Flag | Meaning |
|------|---------|
| `-f <fmt>` | output format (DXGI names, `DXGI_FORMAT_` prefix optional; aliases `DXT1`–`DXT5`, `BPTC`, `BPTC_FLOAT`, `RGBA`, `BGRA`, `BGR`, `FP16`, `FP32`) |
| `-w` / `-h` | resize width / height (GPU box filter) |
| `-m <n>` | mip levels (`0` = full chain) |
| `-if <filter>` | accepted; POINT/LINEAR/BOX/FANT map to the (sRGB-aware) box filter, others warn |
| `-srgb` / `-srgbi` / `-srgbo` | input is sRGB (filtering) / output tagged `_SRGB` / both |
| `-hflip` / `-vflip` | mirror horizontally / vertically |
| `-swizzle <mask>` | channel swizzle (`rgba`/`xyzw`/`0`/`1`, e.g. `abgr`, `rgb1`) |
| `-c <hex>` | colorkey: matching pixels → transparent black |
| `-pmalpha` / `-alpha` | premultiply alpha (after mips, on the GPU; DDS tagged premultiplied) / undo premultiplied alpha |
| `-aw <n>` | BC7 alpha weight |
| `-bc q` / `-bc x` | BC7 quick (modes 4–6) / also try 3-subset modes 0/2 |
| `-px` / `-sx` / `-l` / `-ft` | output name prefix / suffix / lowercase / file type (`dds`, `png`, `jpg`, `webp`) |
| `-o <dir>` | output directory (default: current directory, like texconv) |
| `-y` | overwrite existing outputs (otherwise an error, like texconv) |
| `-r` | recursive wildcard search |
| `-flist <file>` | read input paths from a text file (`#` comments allowed) |
| `-gpu <n>` / `-nogpu` | select the GPU adapter / run kernels on the CPU |
| `-nologo`, `-timing` | suppress banner / print processing time |

Other texconv flags (`-nmap`, `-at`, `-wicq`, `-nits`, `-fl`, `-dx9`, …) are accepted, their
values consumed, and reported as ignored warnings. Both `-flag` and `/flag`, and the
`-flag:value` form, are understood.

## Performance

See [Benchmarks](#benchmarks) below; run your own with:

```bash
dotnet run -c Release --project bench/TexGen.Bench -- --sizes 1024,4096 --mips
```

## Develop

```bash
dotnet build
dotnet test                          # GPU tests use the best available accelerator
TEXGEN_BACKEND=cpu dotnet test       # same tests on the ILGPU CPU accelerator
```

GPU tests verify the encoders by compressing synthetic images, decoding with the CPU
reference decoders (`src/TexGen.Core/Decoders`), and asserting PSNR against the
thresholds used by texconv-js's suite.

## Architecture

```
src/TexGen.Core/
  Dxgi.cs          DXGI_FORMAT enum + metadata (bits/pixel, block size, sRGB) + ComputePitch
  Image.cs         Image / ScratchImage / TexMetadata (mirrors DirectXTex)
  Dds.cs           DDS container read/write (DX10 extended header, legacy FourCC reader)
  ImageIO.cs       SkiaSharp decode (color-managed), .hdr/.dds input, PNG/JPEG/WebP output
  Hdr.cs           Radiance .hdr (RGBE) reader + LDR→linear-float + tonemap helpers
  Transforms.cs    flip / swizzle / colorkey / premultiply (CPU, texconv semantics)
  PixelConvert.cs  uncompressed DXGI format conversions
  Converter.cs     high-level Convert(image, options) pipeline
  TexconvArgs.cs   texconv command-line parser
  Decoders/        CPU reference decoders: BC1–5, BC6H, BC7
  Gpu/             ILGPU device selection, device-resident textures, kernels:
                   ImageKernels (resize/mips/premultiply), BcnEncoder (BC1–5),
                   Bc7Encoder, Bc6hEncoder
src/TexGen.Cli/    texgen executable (texconv-compatible front end)
bench/             throughput benchmark
tests/             xUnit tests (CPU unit tests + GPU integration tests)
```

The GPU pipeline uploads the source once; resize, every mip level, premultiply and
encoding run device-resident, and only the final compressed blocks are downloaded.

## License

MIT (matching DirectXTex and texconv-js).
