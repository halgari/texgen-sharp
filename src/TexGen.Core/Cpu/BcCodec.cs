//-------------------------------------------------------------------------------------
// Cpu/BcCodec.cs — placeholder; replaced by the port of DirectXTex BC.cpp + BC4BC5.cpp.
//-------------------------------------------------------------------------------------

using System.Numerics;

namespace TexGen.Cpu;

internal static class BcCodec
{
    public static void EncodeBC1(Span<byte> bc, ReadOnlySpan<Vector4> color, float threshold, BcFlags flags) => throw new NotImplementedException();
    public static void EncodeBC2(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) => throw new NotImplementedException();
    public static void EncodeBC3(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) => throw new NotImplementedException();
    public static void EncodeBC4U(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) => throw new NotImplementedException();
    public static void EncodeBC4S(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) => throw new NotImplementedException();
    public static void EncodeBC5U(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) => throw new NotImplementedException();
    public static void EncodeBC5S(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) => throw new NotImplementedException();
}
