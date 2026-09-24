using ILGPU.Runtime;

namespace TexGen.Gpu;

// Placeholder — replaced by the ILGPU port of bc6h.wgsl.
internal sealed class Bc6hEncoder
{
    public Bc6hEncoder(Accelerator acc) { }

    public GpuBlocks Encode(GpuTexture src, DxgiFormat format) => throw new NotImplementedException();
}
