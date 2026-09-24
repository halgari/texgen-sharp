using ILGPU.Runtime;

namespace TexGen.Gpu;

// Placeholder — replaced by the ILGPU port of bc7.wgsl.
internal sealed class Bc7Encoder
{
    public Bc7Encoder(Accelerator acc) { }

    public GpuBlocks Encode(GpuTexture src, DxgiFormat format, float alphaWeight, bool quick, bool use3Subsets) =>
        throw new NotImplementedException();
}
