//-------------------------------------------------------------------------------------
// Cpu/Bc6hBc7Codec.cs
//
// Entry points of the DirectXTex CPU BC6H/BC7 codec (D3DXEncodeBC6HU/S, D3DXEncodeBC7).
// Each call encodes one 4x4 block of 16 row-major pixels into 16 bytes; callers
// parallelize over blocks. See Bc6hCpu.cs / Bc7Cpu.cs for the encoders.
//-------------------------------------------------------------------------------------

using System.Numerics;

namespace TexGen.Cpu;

internal static class Bc6hBc7Codec
{
    /// <summary>BC6H_UF16: linear float input; negatives encode as zero.</summary>
    public static void EncodeBC6HU(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) =>
        Bc6hCpu.Encode(bc, color, signed: false);

    /// <summary>BC6H_SF16: linear float input; negative values are preserved.</summary>
    public static void EncodeBC6HS(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) =>
        Bc6hCpu.Encode(bc, color, signed: true);

    /// <summary>BC7: RGBA input in [0,1]. Honors BcFlags.Use3Subsets and BcFlags.ForceBc7Mode6.</summary>
    public static void EncodeBC7(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) =>
        Bc7Cpu.Encode(bc, color, flags);
}
