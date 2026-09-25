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
  Without one, compression uses the DirectXTex CPU codec (`-nogpu` forces it).

## Formats

| | |
|---|---|
| **Input** | PNG, JPEG, BMP, GIF, WebP, ICO, WBMP (SkiaSharp, color-managed to sRGB like the browser), Radiance `.hdr`, DDS (BC1–BC7, BC6H and the uncompressed formats below) |
| **Output (block-compressed)** | BC1, BC2, BC3 (+ `_SRGB`), BC4_UNORM, BC5_UNORM, BC6H_UF16, BC6H_SF16, BC7 (+ `_SRGB`) on GPU or CPU; BC4_SNORM, BC5_SNORM on CPU |
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
| `-bc q` / `-bc x` | BC7 quick (GPU: modes 4–6, CPU: mode 6) / also try 3-subset modes 0/2 |
| `-bc d` / `-bc u` / `-at <n>` | CPU codec: dither / uniform weighting / BC1 alpha threshold |
| `-px` / `-sx` / `-l` / `-ft` | output name prefix / suffix / lowercase / file type (`dds`, `png`, `jpg`, `webp`) |
| `-o <dir>` | output directory (default: current directory, like texconv) |
| `-y` | overwrite existing outputs (otherwise an error, like texconv) |
| `-r` | recursive wildcard search |
| `-flist <file>` | read input paths from a text file (`#` comments allowed) |
| `-gpu <n>` / `-nogpu` | select the GPU adapter / use the DirectXTex CPU codec |
| `-nologo`, `-timing` | suppress banner / print processing time |

Other texconv flags (`-nmap`, `-wicq`, `-nits`, `-fl`, `-dx9`, …) are accepted, their
values consumed, and reported as ignored warnings. Both `-flag` and `/flag`, and the
`-flag:value` form, are understood.

## Performance

End-to-end `Converter.Convert` time (upload → encode → download, excluding file I/O) on an
**NVIDIA RTX 5090** (CUDA), for a noisy photo-like test image, median of 5 runs
(`bench/TexGen.Bench`):

| Format | 1024² | 4096² | 8192² | 4096² + full mip chain |
|---|---|---|---|---|
| BC1 | 0.5 ms | 5.1 ms | 17.6 ms (3.8 GPix/s) | 6.9 ms |
| BC3 | 0.7 ms | 7.4 ms | 26.9 ms | 11.1 ms |
| BC4 | 0.6 ms | 6.6 ms | 22.0 ms | 9.9 ms |
| BC5 | 0.8 ms | 8.0 ms | 30.4 ms | 12.9 ms |
| BC7 | 3.8 ms | 58.7 ms | 236 ms (284 MPix/s) | 81.9 ms |
| BC7 `-bc q` | | 8.5 ms | | |
| BC7 `-bc x` | | 87.7 ms | | |
| BC6H (UF16) | 6.3 ms | 78.4 ms | 298 ms | 99.5 ms |

The first conversion of each encoder family pays a one-time ILGPU kernel compile
(≈0.3 s for BC1–5, ≈0.5 s each for BC7 and BC6H). The texgen CLI converts a whole
folder in one process, so that cost is paid once per run.

Output quality, checked with Pillow's independent DDS decoder on real images: BC7
≈44–51 dB, BC1/BC3 ≈37 dB, BC4/BC5 ≈54 dB, BC6H ≈49 dB (PSNR).

### GPU vs CPU

Without a GPU, or with `-nogpu`, compression uses a C# port of **DirectXTex's CPU codec**
(`BC.cpp`, `BC4BC5.cpp`, `BC6HBC7.cpp` — texconv's own non-GPU encoders). The BC6H/BC7
port was verified byte-identical to natively compiled DirectXTex over ~80k blocks.
Blocks are encoded in parallel on all cores. Same image, RTX 5090 vs 32-core CPU,
end-to-end:

| Format | Size | GPU | CPU codec | GPU speedup |
|---|---|---|---|---|
| BC1 | 4096² | 5.2 ms | 58 ms | 11× |
| BC3 | 4096² | 7.2 ms | 41 ms | 6× |
| BC4 | 4096² | 6.6 ms | 27 ms | 4× |
| BC5 | 4096² | 8.0 ms | 46 ms | 6× |
| BC7 quick (`-bc q`) | 4096² | 8.8 ms | 2.2 s | 250× |
| BC7 | 1024² | 4.0 ms | 4.6 s | 1,150× |
| BC6H | 1024² | 6.3 ms | 2.2 s | 340× |

The BC6H/BC7 CPU encoders score candidate endpoints with SIMD, on all 16 pixels of a
block at once, which makes them 2–2.5× faster than DirectXTex's scalar loops. The
output stays byte-identical, since each lane reproduces the scalar search exactly. The
width is chosen at startup: AVX-512, then AVX2 (256-bit), then SSE/NEON (128-bit), then
scalar. Every level is tested against scalar and against native DirectXTex output
(`TEXGEN_SIMD=scalar|vector` forces a lower level).

| CPU codec, 1024² | scalar | 128-bit | 256-bit | AVX-512 |
|---|---|---|---|---|
| BC7 | 10.5 s | 5.8 s | 4.9 s | 4.6 s |
| BC6H | 4.4 s | 2.8 s | 2.5 s | 2.2 s |

Quality is comparable. The CPU BC1–3 encoders are ~1 dB better on noisy images and
~0.6 dB behind on smooth gradients; BC6H/BC7 are within a few tenths of a dB. The CPU
codec also adds BC4/BC5 **SNORM**, texconv's `-bc d` (dither) / `-bc u` (uniform
weighting), and `-at` (BC1 alpha threshold). Select it explicitly with
`ConvertOptions.Codec = CodecPreference.Cpu`.

The GPU kernels themselves can also run on ILGPU's CPU accelerator, which CI uses to
test them without a GPU. That mode emulates GPU barriers with OS threads, so it is
only for testing.

Run the benchmark yourself:

```bash
dotnet run -c Release --project bench/TexGen.Bench -- --sizes 1024,4096 --mips
dotnet run -c Release --project bench/TexGen.Bench -- --formats BC7_UNORM --quick
```

## Differences from texconv-js

- A real `texgen` executable, so the file-system flags texconv-js could only warn about
  (`-o`, `-y`, `-r`, `-flist`, `-gpu`, `-nogpu`, `-timing`) work.
- Extra inputs and outputs: DDS input, uncompressed FP16/FP32/R8/RG8/R16 outputs, and
  PNG/JPEG/WebP via `-ft`. The browser demo UI is not ported.
- GPU work stays on the device across resize, mips and encode, instead of one
  upload/readback per stage.
- The BC6H/BC7 kernels read texels byte-exactly with edge clamping. EncodeBlock gains
  one barrier that the HLSL omitted (it relied on warp lockstep).
- Image decoding uses SkiaSharp rather than the browser, so AVIF/HEIC input depends on
  the platform's Skia build. The OpenCL path is untested; development used CUDA.

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
  Cpu/             DirectXTex CPU codec port (BC1–5, BC6H, BC7) + parallel block driver
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
