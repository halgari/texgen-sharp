//-------------------------------------------------------------------------------------
// Cpu/Bc6hBc7Codec.cs — placeholder; replaced by the port of DirectXTex BC6HBC7.cpp.
//-------------------------------------------------------------------------------------

using System.Numerics;

namespace TexGen.Cpu;

internal static class Bc6hBc7Codec
{
    public static void EncodeBC6HU(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) => throw new NotImplementedException();
    public static void EncodeBC6HS(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) => throw new NotImplementedException();
    public static void EncodeBC7(Span<byte> bc, ReadOnlySpan<Vector4> color, BcFlags flags) => throw new NotImplementedException();
}
