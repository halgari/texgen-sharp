//-------------------------------------------------------------------------------------
// Gpu/GpuMath.cs
//
// Small unmanaged vector types used inside ILGPU kernels. They stand in for the
// HLSL/WGSL built-in vectors (uint4, int3, float3, float4) the original encoders
// were written with. Everything is a plain struct with aggressively inlined
// operators so ILGPU lowers it to scalar registers.
//-------------------------------------------------------------------------------------

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TexGen.Gpu;

[StructLayout(LayoutKind.Sequential)]
public struct UInt4
{
    public uint X, Y, Z, W;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public UInt4(uint x, uint y, uint z, uint w) { X = x; Y = y; Z = z; W = w; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public UInt4(uint v) { X = v; Y = v; Z = v; W = v; }

    public uint this[int i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        readonly get => i == 0 ? X : i == 1 ? Y : i == 2 ? Z : W;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            if (i == 0) X = value;
            else if (i == 1) Y = value;
            else if (i == 2) Z = value;
            else W = value;
        }
    }

    public static UInt4 operator +(UInt4 a, UInt4 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W);
    public static UInt4 operator -(UInt4 a, UInt4 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z, a.W - b.W);
    public static UInt4 operator *(UInt4 a, UInt4 b) => new(a.X * b.X, a.Y * b.Y, a.Z * b.Z, a.W * b.W);
    public static UInt4 operator *(UInt4 a, uint b) => new(a.X * b, a.Y * b, a.Z * b, a.W * b);
    public static UInt4 operator +(UInt4 a, uint b) => new(a.X + b, a.Y + b, a.Z + b, a.W + b);
    public static UInt4 operator >>(UInt4 a, int s) => new(a.X >> s, a.Y >> s, a.Z >> s, a.W >> s);
    public static UInt4 operator <<(UInt4 a, int s) => new(a.X << s, a.Y << s, a.Z << s, a.W << s);
    public static UInt4 operator &(UInt4 a, uint m) => new(a.X & m, a.Y & m, a.Z & m, a.W & m);
    public static UInt4 operator |(UInt4 a, UInt4 b) => new(a.X | b.X, a.Y | b.Y, a.Z | b.Z, a.W | b.W);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt4 Min(UInt4 a, UInt4 b) =>
        new(Math.Min(a.X, b.X), Math.Min(a.Y, b.Y), Math.Min(a.Z, b.Z), Math.Min(a.W, b.W));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt4 Max(UInt4 a, UInt4 b) =>
        new(Math.Max(a.X, b.X), Math.Max(a.Y, b.Y), Math.Max(a.Z, b.Z), Math.Max(a.W, b.W));
}

[StructLayout(LayoutKind.Sequential)]
public struct Int3
{
    public int X, Y, Z;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Int3(int x, int y, int z) { X = x; Y = y; Z = z; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Int3(int v) { X = v; Y = v; Z = v; }

    public int this[int i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        readonly get => i == 0 ? X : i == 1 ? Y : Z;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            if (i == 0) X = value;
            else if (i == 1) Y = value;
            else Z = value;
        }
    }

    public static Int3 operator +(Int3 a, Int3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static Int3 operator -(Int3 a, Int3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    public static Int3 operator *(Int3 a, Int3 b) => new(a.X * b.X, a.Y * b.Y, a.Z * b.Z);
    public static Int3 operator *(Int3 a, int b) => new(a.X * b, a.Y * b, a.Z * b);
    public static Int3 operator -(Int3 a) => new(-a.X, -a.Y, -a.Z);
    public static Int3 operator >>(Int3 a, int s) => new(a.X >> s, a.Y >> s, a.Z >> s);
    public static Int3 operator <<(Int3 a, int s) => new(a.X << s, a.Y << s, a.Z << s);
    public static Int3 operator &(Int3 a, int m) => new(a.X & m, a.Y & m, a.Z & m);
}

[StructLayout(LayoutKind.Sequential)]
public struct Float3
{
    public float X, Y, Z;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Float3(float x, float y, float z) { X = x; Y = y; Z = z; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Float3(float v) { X = v; Y = v; Z = v; }

    public float this[int i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        readonly get => i == 0 ? X : i == 1 ? Y : Z;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            if (i == 0) X = value;
            else if (i == 1) Y = value;
            else Z = value;
        }
    }

    public static Float3 operator +(Float3 a, Float3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static Float3 operator -(Float3 a, Float3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    public static Float3 operator *(Float3 a, Float3 b) => new(a.X * b.X, a.Y * b.Y, a.Z * b.Z);
    public static Float3 operator *(Float3 a, float b) => new(a.X * b, a.Y * b, a.Z * b);
    public static Float3 operator *(float b, Float3 a) => new(a.X * b, a.Y * b, a.Z * b);
    public static Float3 operator /(Float3 a, float b) => new(a.X / b, a.Y / b, a.Z / b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(Float3 a, Float3 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float3 Min(Float3 a, Float3 b) => new(MathF.Min(a.X, b.X), MathF.Min(a.Y, b.Y), MathF.Min(a.Z, b.Z));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float3 Max(Float3 a, Float3 b) => new(MathF.Max(a.X, b.X), MathF.Max(a.Y, b.Y), MathF.Max(a.Z, b.Z));
}

[StructLayout(LayoutKind.Sequential)]
public struct Float4
{
    public float X, Y, Z, W;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Float4(float x, float y, float z, float w) { X = x; Y = y; Z = z; W = w; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Float4(Float3 v, float w) { X = v.X; Y = v.Y; Z = v.Z; W = w; }

    public readonly Float3 Xyz
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => new(X, Y, Z);
    }

    public float this[int i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        readonly get => i == 0 ? X : i == 1 ? Y : i == 2 ? Z : W;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            if (i == 0) X = value;
            else if (i == 1) Y = value;
            else if (i == 2) Z = value;
            else W = value;
        }
    }

    public static Float4 operator +(Float4 a, Float4 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W);
    public static Float4 operator *(Float4 a, float b) => new(a.X * b, a.Y * b, a.Z * b, a.W * b);
    public static Float4 operator /(Float4 a, float b) => new(a.X / b, a.Y / b, a.Z / b, a.W / b);
}

/// <summary>Pixel packing helpers shared by kernels (RGBA8 is stored as one uint, R in the low byte).</summary>
public static class Texel
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static UInt4 Unpack(uint p) => new(p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF, p >> 24);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Float4 UnpackUnorm(uint p) =>
        new((p & 0xFF) / 255f, ((p >> 8) & 0xFF) / 255f, ((p >> 16) & 0xFF) / 255f, (p >> 24) / 255f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint PackUnorm(Float4 c)
    {
        uint r = (uint)MathF.Round(GMath.Clamp(c.X, 0f, 1f) * 255f);
        uint g = (uint)MathF.Round(GMath.Clamp(c.Y, 0f, 1f) * 255f);
        uint b = (uint)MathF.Round(GMath.Clamp(c.Z, 0f, 1f) * 255f);
        uint a = (uint)MathF.Round(GMath.Clamp(c.W, 0f, 1f) * 255f);
        return r | (g << 8) | (b << 16) | (a << 24);
    }
}

/// <summary>
/// Kernel-safe scalar helpers. BCL methods that can throw (Math.Clamp) are not
/// compilable by ILGPU, so kernels use these instead.
/// </summary>
public static class GMath
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Clamp(float v, float lo, float hi) => MathF.Min(MathF.Max(v, lo), hi);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Clamp(int v, int lo, int hi) => Math.Min(Math.Max(v, lo), hi);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Clamp(uint v, uint lo, uint hi) => Math.Min(Math.Max(v, lo), hi);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Saturate(float v) => MathF.Min(MathF.Max(v, 0f), 1f);
}
