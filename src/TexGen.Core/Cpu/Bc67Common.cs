//-------------------------------------------------------------------------------------
// Cpu/Bc67Common.cs
//
// Shared pieces of the DirectXTex CPU BC6H/BC7 codec (port of BC6HBC7.cpp): the
// partition / fix-up / weight tables, LDRColorA, INTColor, endpoint pairs, the
// 128-bit block writer (CBits), and the OptimizeRGB / OptimizeRGBA / ComputeError
// helpers. Everything is struct-based with fixed-size inline arrays so encoding a
// block performs no heap allocation.
//-------------------------------------------------------------------------------------

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TexGen.Cpu;

internal static class Bc67Tables
{
    public const int WeightMax = 64;
    public const int WeightShift = 6;
    public const int WeightRound = 32;

    public const float Epsilon = (0.25f / 64.0f) * (0.25f / 64.0f);

    public static ReadOnlySpan<float> C3 => [2.0f / 2.0f, 1.0f / 2.0f, 0.0f / 2.0f];
    public static ReadOnlySpan<float> D3 => [0.0f / 2.0f, 1.0f / 2.0f, 2.0f / 2.0f];
    public static ReadOnlySpan<float> C4 => [3.0f / 3.0f, 2.0f / 3.0f, 1.0f / 3.0f, 0.0f / 3.0f];
    public static ReadOnlySpan<float> D4 => [0.0f / 3.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f / 3.0f];

    public static ReadOnlySpan<int> Weights2 => [0, 21, 43, 64];
    public static ReadOnlySpan<int> Weights3 => [0, 9, 18, 27, 37, 46, 55, 64];
    public static ReadOnlySpan<int> Weights4 => [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

    public static ReadOnlySpan<int> Weights(int prec) => prec switch
    {
        2 => Weights2,
        3 => Weights3,
        _ => Weights4,
    };

    /// <summary>g_aPartitionTable[partitions][shape][pixel].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Partition(int partitions, int shape, int pixel) => PartitionTable[(partitions * 64 + shape) * 16 + pixel];

    /// <summary>g_aFixUp[partitions][shape][region].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int FixUpIndex(int partitions, int shape, int region) => FixUp[(partitions * 64 + shape) * 3 + region];

    public static bool IsFixUpOffset(int partitions, int shape, int offset)
    {
        for (int p = 0; p <= partitions; p++)
        {
            if (offset == FixUpIndex(partitions, shape, p)) return true;
        }
        return false;
    }

    // g_aPartitionTable[3][64][16] flattened: [(partitions * 64 + shape) * 16 + pixel]
    internal static ReadOnlySpan<byte> PartitionTable =>
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
        0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1,
        0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0,
        0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
        0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1,
        0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
        0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
        0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
        0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
        0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1,
        0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
        0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
        0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
        0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,
        0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1,
        0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0,
        0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1,
        0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1,
        0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2,
        0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1,
        0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1,
        0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2,
        0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2,
        0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
        0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2,
        0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
        0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2,
        0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2,
        0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0,
        0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2,
        0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0,
        0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2,
        0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1,
        0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2,
        0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2,
        0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0,
        0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2,
        0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1,
        0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2,
        0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2,
        0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1,
        0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1,
        0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 2, 2, 2,
        0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2,
        0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0,
        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
        0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0,
        0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1,
        0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1,
        0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2,
        0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1,
        0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1,
        0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1,
        0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2,
        0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1,
        0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2,
        0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2,
        0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2,
        0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2,
        0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2,
        0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2,
        0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2,
        0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2,
        0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1,
        0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2,
        0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0,
    ];

    // g_aFixUp[3][64][3] flattened: [(partitions * 64 + shape) * 3 + region]
    internal static ReadOnlySpan<byte> FixUp =>
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 15, 0, 0, 15, 0, 0, 15, 0, 0, 15, 0,
        0, 15, 0, 0, 15, 0, 0, 15, 0, 0, 15, 0,
        0, 15, 0, 0, 15, 0, 0, 15, 0, 0, 15, 0,
        0, 15, 0, 0, 15, 0, 0, 15, 0, 0, 15, 0,
        0, 15, 0, 0, 2, 0, 0, 8, 0, 0, 2, 0,
        0, 2, 0, 0, 8, 0, 0, 8, 0, 0, 15, 0,
        0, 2, 0, 0, 8, 0, 0, 2, 0, 0, 2, 0,
        0, 8, 0, 0, 8, 0, 0, 2, 0, 0, 2, 0,
        0, 15, 0, 0, 15, 0, 0, 6, 0, 0, 8, 0,
        0, 2, 0, 0, 8, 0, 0, 15, 0, 0, 15, 0,
        0, 2, 0, 0, 8, 0, 0, 2, 0, 0, 2, 0,
        0, 2, 0, 0, 15, 0, 0, 15, 0, 0, 6, 0,
        0, 6, 0, 0, 2, 0, 0, 6, 0, 0, 8, 0,
        0, 15, 0, 0, 15, 0, 0, 2, 0, 0, 2, 0,
        0, 15, 0, 0, 15, 0, 0, 15, 0, 0, 15, 0,
        0, 15, 0, 0, 2, 0, 0, 2, 0, 0, 15, 0,
        0, 3, 15, 0, 3, 8, 0, 15, 8, 0, 15, 3,
        0, 8, 15, 0, 3, 15, 0, 15, 3, 0, 15, 8,
        0, 8, 15, 0, 8, 15, 0, 6, 15, 0, 6, 15,
        0, 6, 15, 0, 5, 15, 0, 3, 15, 0, 3, 8,
        0, 3, 15, 0, 3, 8, 0, 8, 15, 0, 15, 3,
        0, 3, 15, 0, 3, 8, 0, 6, 15, 0, 10, 8,
        0, 5, 3, 0, 8, 15, 0, 8, 6, 0, 6, 10,
        0, 8, 15, 0, 5, 15, 0, 15, 10, 0, 15, 8,
        0, 8, 15, 0, 15, 3, 0, 3, 15, 0, 5, 10,
        0, 6, 10, 0, 10, 8, 0, 8, 9, 0, 15, 10,
        0, 15, 6, 0, 3, 15, 0, 15, 8, 0, 5, 15,
        0, 15, 3, 0, 15, 6, 0, 15, 6, 0, 15, 8,
        0, 3, 15, 0, 15, 3, 0, 5, 15, 0, 5, 15,
        0, 5, 15, 0, 8, 15, 0, 5, 15, 0, 10, 15,
        0, 5, 15, 0, 10, 15, 0, 8, 15, 0, 13, 15,
        0, 15, 3, 0, 12, 15, 0, 3, 15, 0, 3, 8,
    ];
}

/// <summary>8-bit RGBA color (LDRColorA).</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct LdrColorA
{
    public byte R, G, B, A;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public LdrColorA(byte r, byte g, byte b, byte a)
    {
        R = r;
        G = g;
        B = b;
        A = a;
    }

    [UnscopedRef]
    public ref byte this[int i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref Unsafe.Add(ref R, i);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte Lerp(int c0, int c1, int w) =>
        (byte)((uint)(c0 * (Bc67Tables.WeightMax - w) + c1 * w + Bc67Tables.WeightRound) >> Bc67Tables.WeightShift);

    public static void InterpolateRgb(in LdrColorA c0, in LdrColorA c1, int wc, int wcprec, ref LdrColorA o)
    {
        int w = Bc67Tables.Weights(wcprec)[wc];
        o.R = Lerp(c0.R, c1.R, w);
        o.G = Lerp(c0.G, c1.G, w);
        o.B = Lerp(c0.B, c1.B, w);
    }

    public static void InterpolateA(in LdrColorA c0, in LdrColorA c1, int wa, int waprec, ref LdrColorA o)
    {
        int w = Bc67Tables.Weights(waprec)[wa];
        o.A = Lerp(c0.A, c1.A, w);
    }

    public static void Interpolate(in LdrColorA c0, in LdrColorA c1, int wc, int wa, int wcprec, int waprec, ref LdrColorA o)
    {
        InterpolateRgb(c0, c1, wc, wcprec, ref o);
        InterpolateA(c0, c1, wa, waprec, ref o);
    }

    /// <summary>HDRColorA::ToLDRColorA (values already scaled to 0..255).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LdrColorA FromScaled(in HdrColorA c) =>
        new((byte)(c.R + 0.01f), (byte)(c.G + 0.01f), (byte)(c.B + 0.01f), (byte)(c.A + 0.01f));
}

internal struct LdrEndPntPair
{
    public LdrColorA A;
    public LdrColorA B;
}

/// <summary>Integer RGB color used by the BC6H encoder (INTColor).</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct IntColor
{
    private const int F16SMask = 0x8000;
    private const int F16EMMask = 0x7fff;
    public const int F16Max = 0x7bff;

    public int R, G, B, Pad;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public IntColor(int r, int g, int b)
    {
        R = r;
        G = g;
        B = b;
        Pad = 0;
    }

    [UnscopedRef]
    public ref int this[int i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref Unsafe.Add(ref R, i);
    }

    public void Add(in IntColor c)
    {
        R += c.R;
        G += c.G;
        B += c.B;
    }

    public void Sub(in IntColor c)
    {
        R -= c.R;
        G -= c.G;
        B -= c.B;
    }

    public void And(in IntColor c)
    {
        R &= c.R;
        G &= c.G;
        B &= c.B;
    }

    /// <summary>Convert to half floats (XMStoreHalf4, round-to-nearest-even) and then to BC6H integers.</summary>
    public void Set(in HdrColorA c, bool signed)
    {
        R = F16ToInt(BitConverter.HalfToUInt16Bits((Half)c.R), signed);
        G = F16ToInt(BitConverter.HalfToUInt16Bits((Half)c.G), signed);
        B = F16ToInt(BitConverter.HalfToUInt16Bits((Half)c.B), signed);
    }

    public void Clamp(int min, int max)
    {
        R = Math.Min(max, Math.Max(min, R));
        G = Math.Min(max, Math.Max(min, G));
        B = Math.Min(max, Math.Max(min, B));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SignExtend(int x, int nb) => ((x & (1 << (nb - 1))) != 0 ? (~0 ^ ((1 << nb) - 1)) : 0) | x;

    public void SignExtend(in LdrColorA prec)
    {
        R = SignExtend(R, prec.R);
        G = SignExtend(G, prec.G);
        B = SignExtend(B, prec.B);
    }

    private static int F16ToInt(ushort input, bool signed)
    {
        int output;
        if (signed)
        {
            int s = input & F16SMask;
            input = (ushort)(input & F16EMMask);
            output = input > F16Max ? F16Max : input;
            output = s != 0 ? -output : output;
        }
        else
        {
            output = (input & F16SMask) != 0 ? 0 : input;
        }
        return output;
    }
}

internal struct IntEndPntPair
{
    public IntColor A;
    public IntColor B;
}

[InlineArray(16)]
internal struct Buffer16<T>
{
    private T _element0;
}

[InlineArray(3)]
internal struct Buffer3<T>
{
    private T _element0;
}

[InlineArray(2)]
internal struct Buffer2<T>
{
    private T _element0;
}

/// <summary>The 128-bit block being assembled (CBits&lt;16&gt;).</summary>
[InlineArray(16)]
internal struct Bits128
{
    private byte _element0;

    public void SetBit(ref int startBit, int value)
    {
        int index = startBit >> 3;
        int shift = startBit - (index << 3);
        this[index] = (byte)((this[index] & ~(1 << shift)) | (value << shift));
        startBit++;
    }

    public void SetBits(ref int startBit, int numBits, int value)
    {
        if (numBits == 0) return;
        int index = startBit >> 3;
        int shift = startBit - (index << 3);
        if (shift + numBits > 8)
        {
            int firstIndexBits = 8 - shift;
            int nextIndexBits = numBits - firstIndexBits;
            this[index] = (byte)((this[index] & ~(((1 << firstIndexBits) - 1) << shift)) | (value << shift));
            this[index + 1] = (byte)((this[index + 1] & ~((1 << nextIndexBits) - 1)) | (value >> firstIndexBits));
        }
        else
        {
            this[index] = (byte)((this[index] & ~(((1 << numBits) - 1) << shift)) | (value << shift));
        }
        startBit += numBits;
    }
}

/// <summary>Endpoint optimizers and error metrics shared by BC6H and BC7.</summary>
internal static class Bc67Math
{
    /// <summary>FLT_MIN (smallest normalized float).</summary>
    private const float FltMin = 1.175494351e-38f;

    /// <summary>
    /// OptimizeRGB: least-squares endpoint fit over the RGB channels of the indexed
    /// points (alpha of the outputs is left untouched, as in DirectXTex).
    /// </summary>
    public static void OptimizeRgb(ReadOnlySpan<HdrColorA> points, ref HdrColorA pX, ref HdrColorA pY, int steps,
        int cPixels, ReadOnlySpan<int> index)
    {
        var pC = steps == 3 ? Bc67Tables.C3 : Bc67Tables.C4;
        var pD = steps == 3 ? Bc67Tables.D3 : Bc67Tables.D4;

        // Find Min and Max points, as starting point
        var x = new HdrColorA(float.MaxValue, float.MaxValue, float.MaxValue, 0.0f);
        var y = new HdrColorA(-float.MaxValue, -float.MaxValue, -float.MaxValue, 0.0f);

        for (int i = 0; i < cPixels; i++)
        {
            ref readonly var pt = ref points[index[i]];
            if (pt.R < x.R) x.R = pt.R;
            if (pt.G < x.G) x.G = pt.G;
            if (pt.B < x.B) x.B = pt.B;
            if (pt.R > y.R) y.R = pt.R;
            if (pt.G > y.G) y.G = pt.G;
            if (pt.B > y.B) y.B = pt.B;
        }

        // Diagonal axis
        float abR = y.R - x.R, abG = y.G - x.G, abB = y.B - x.B;
        float fAB = abR * abR + abG * abG + abB * abB;

        // Single color block.. no need to root-find
        if (fAB < FltMin)
        {
            pX.R = x.R; pX.G = x.G; pX.B = x.B;
            pY.R = y.R; pY.G = y.G; pY.B = y.B;
            return;
        }

        // Try all four axis directions, to determine which diagonal best fits data
        float fABInv = 1.0f / fAB;
        float dirR = abR * fABInv, dirG = abG * fABInv, dirB = abB * fABInv;
        float midR = (x.R + y.R) * 0.5f, midG = (x.G + y.G) * 0.5f, midB = (x.B + y.B) * 0.5f;

        float fDir0 = 0, fDir1 = 0, fDir2 = 0, fDir3 = 0;
        for (int i = 0; i < cPixels; i++)
        {
            ref readonly var pt = ref points[index[i]];
            float ptR = (pt.R - midR) * dirR;
            float ptG = (pt.G - midG) * dirG;
            float ptB = (pt.B - midB) * dirB;
            float f;
            f = ptR + ptG + ptB; fDir0 += f * f;
            f = ptR + ptG - ptB; fDir1 += f * f;
            f = ptR - ptG + ptB; fDir2 += f * f;
            f = ptR - ptG - ptB; fDir3 += f * f;
        }

        float fDirMax = fDir0;
        int iDirMax = 0;
        if (fDir1 > fDirMax) { fDirMax = fDir1; iDirMax = 1; }
        if (fDir2 > fDirMax) { fDirMax = fDir2; iDirMax = 2; }
        if (fDir3 > fDirMax) { iDirMax = 3; }

        if ((iDirMax & 2) != 0) (x.G, y.G) = (y.G, x.G);
        if ((iDirMax & 1) != 0) (x.B, y.B) = (y.B, x.B);

        // Two color block.. no need to root-find
        if (fAB < 1.0f / 4096.0f)
        {
            pX.R = x.R; pX.G = x.G; pX.B = x.B;
            pY.R = y.R; pY.G = y.G; pY.B = y.B;
            return;
        }

        // Use Newton's Method to find local minima of sum-of-squares error.
        float fSteps = steps - 1;
        Span<float> stepR = stackalloc float[4];
        Span<float> stepG = stackalloc float[4];
        Span<float> stepB = stackalloc float[4];

        for (int iteration = 0; iteration < 8; iteration++)
        {
            // Calculate new steps
            for (int s = 0; s < steps; s++)
            {
                stepR[s] = x.R * pC[s] + y.R * pD[s];
                stepG[s] = x.G * pC[s] + y.G * pD[s];
                stepB[s] = x.B * pC[s] + y.B * pD[s];
            }

            // Calculate color direction
            dirR = y.R - x.R;
            dirG = y.G - x.G;
            dirB = y.B - x.B;
            float fLen = dirR * dirR + dirG * dirG + dirB * dirB;
            if (fLen < (1.0f / 4096.0f)) break;

            float fScale = fSteps / fLen;
            dirR *= fScale;
            dirG *= fScale;
            dirB *= fScale;

            // Evaluate function, and derivatives
            float d2X = 0, d2Y = 0;
            float dXR = 0, dXG = 0, dXB = 0, dYR = 0, dYG = 0, dYB = 0;

            for (int i = 0; i < cPixels; i++)
            {
                ref readonly var pt = ref points[index[i]];
                float fDot = (pt.R - x.R) * dirR + (pt.G - x.G) * dirG + (pt.B - x.B) * dirB;

                int iStep;
                if (fDot <= 0.0f) iStep = 0;
                else if (fDot >= fSteps) iStep = steps - 1;
                else iStep = (int)(fDot + 0.5f);

                float diffR = stepR[iStep] - pt.R;
                float diffG = stepG[iStep] - pt.G;
                float diffB = stepB[iStep] - pt.B;

                float fC = pC[iStep] * (1.0f / 8.0f);
                float fD = pD[iStep] * (1.0f / 8.0f);

                d2X += fC * pC[iStep];
                dXR += fC * diffR;
                dXG += fC * diffG;
                dXB += fC * diffB;

                d2Y += fD * pD[iStep];
                dYR += fD * diffR;
                dYG += fD * diffG;
                dYB += fD * diffB;
            }

            // Move endpoints
            if (d2X > 0.0f)
            {
                float f = -1.0f / d2X;
                x.R += dXR * f;
                x.G += dXG * f;
                x.B += dXB * f;
            }

            if (d2Y > 0.0f)
            {
                float f = -1.0f / d2Y;
                y.R += dYR * f;
                y.G += dYG * f;
                y.B += dYB * f;
            }

            const float eps = Bc67Tables.Epsilon;
            if (dXR * dXR < eps && dXG * dXG < eps && dXB * dXB < eps && dYR * dYR < eps && dYG * dYG < eps && dYB * dYB < eps)
                break;
        }

        pX.R = x.R; pX.G = x.G; pX.B = x.B;
        pY.R = y.R; pY.G = y.G; pY.B = y.B;
    }

    /// <summary>OptimizeRGBA: least-squares endpoint fit over all four channels.</summary>
    public static void OptimizeRgba(ReadOnlySpan<HdrColorA> points, out HdrColorA pX, out HdrColorA pY, int steps,
        int cPixels, ReadOnlySpan<int> index)
    {
        var pC = steps == 3 ? Bc67Tables.C3 : Bc67Tables.C4;
        var pD = steps == 3 ? Bc67Tables.D3 : Bc67Tables.D4;

        // Find Min and Max points, as starting point
        var x = new HdrColorA(1.0f, 1.0f, 1.0f, 1.0f);
        var y = new HdrColorA(0.0f, 0.0f, 0.0f, 0.0f);

        for (int i = 0; i < cPixels; i++)
        {
            ref readonly var pt = ref points[index[i]];
            if (pt.R < x.R) x.R = pt.R;
            if (pt.G < x.G) x.G = pt.G;
            if (pt.B < x.B) x.B = pt.B;
            if (pt.A < x.A) x.A = pt.A;
            if (pt.R > y.R) y.R = pt.R;
            if (pt.G > y.G) y.G = pt.G;
            if (pt.B > y.B) y.B = pt.B;
            if (pt.A > y.A) y.A = pt.A;
        }

        // Diagonal axis
        var ab = y - x;
        float fAB = HdrColorA.Dot(ab, ab);

        // Single color block.. no need to root-find
        if (fAB < FltMin)
        {
            pX = x;
            pY = y;
            return;
        }

        // Try all four axis directions, to determine which diagonal best fits data
        float fABInv = 1.0f / fAB;
        var dir = ab * fABInv;
        var mid = (x + y) * 0.5f;

        Span<float> fDir = stackalloc float[8];
        fDir.Clear();

        for (int i = 0; i < cPixels; i++)
        {
            ref readonly var p = ref points[index[i]];
            float ptR = (p.R - mid.R) * dir.R;
            float ptG = (p.G - mid.G) * dir.G;
            float ptB = (p.B - mid.B) * dir.B;
            float ptA = (p.A - mid.A) * dir.A;

            float f;
            f = ptR + ptG + ptB + ptA; fDir[0] += f * f;
            f = ptR + ptG + ptB - ptA; fDir[1] += f * f;
            f = ptR + ptG - ptB + ptA; fDir[2] += f * f;
            f = ptR + ptG - ptB - ptA; fDir[3] += f * f;
            f = ptR - ptG + ptB + ptA; fDir[4] += f * f;
            f = ptR - ptG + ptB - ptA; fDir[5] += f * f;
            f = ptR - ptG - ptB + ptA; fDir[6] += f * f;
            f = ptR - ptG - ptB - ptA; fDir[7] += f * f;
        }

        float fDirMax = fDir[0];
        int iDirMax = 0;
        for (int iDir = 1; iDir < 8; iDir++)
        {
            if (fDir[iDir] > fDirMax)
            {
                fDirMax = fDir[iDir];
                iDirMax = iDir;
            }
        }

        if ((iDirMax & 4) != 0) (x.G, y.G) = (y.G, x.G);
        if ((iDirMax & 2) != 0) (x.B, y.B) = (y.B, x.B);
        if ((iDirMax & 1) != 0) (x.A, y.A) = (y.A, x.A);

        // Two color block.. no need to root-find
        if (fAB < 1.0f / 4096.0f)
        {
            pX = x;
            pY = y;
            return;
        }

        // Use Newton's Method to find local minima of sum-of-squares error.
        float fSteps = steps - 1;
        Span<HdrColorA> pSteps = stackalloc HdrColorA[4];

        for (int iteration = 0; iteration < 8; iteration++)
        {
            // Calculate new steps
            for (int s = 0; s < steps; s++) pSteps[s] = x * pC[s] + y * pD[s];

            // Calculate color direction
            dir = y - x;
            float fLen = HdrColorA.Dot(dir, dir);
            if (fLen < (1.0f / 4096.0f)) break;

            float fScale = fSteps / fLen;
            dir = dir * fScale;

            // Evaluate function, and derivatives
            float d2X = 0, d2Y = 0;
            var dX = new HdrColorA(0, 0, 0, 0);
            var dY = new HdrColorA(0, 0, 0, 0);

            for (int i = 0; i < cPixels; i++)
            {
                ref readonly var p = ref points[index[i]];
                float fDot = HdrColorA.Dot(p - x, dir);

                int iStep;
                if (fDot <= 0.0f) iStep = 0;
                else if (fDot >= fSteps) iStep = steps - 1;
                else iStep = (int)(fDot + 0.5f);

                var diff = pSteps[iStep] - p;
                float fC = pC[iStep] * (1.0f / 8.0f);
                float fD = pD[iStep] * (1.0f / 8.0f);

                d2X += fC * pC[iStep];
                dX = dX + diff * fC;

                d2Y += fD * pD[iStep];
                dY = dY + diff * fD;
            }

            // Move endpoints
            if (d2X > 0.0f) x = x + dX * (-1.0f / d2X);
            if (d2Y > 0.0f) y = y + dY * (-1.0f / d2Y);

            if (HdrColorA.Dot(dX, dX) < Bc67Tables.Epsilon && HdrColorA.Dot(dY, dY) < Bc67Tables.Epsilon) break;
        }

        pX = x;
        pY = y;
    }

    /// <summary>
    /// ComputeError: best palette entry for one pixel (RGBA metric, or RGB + separate
    /// alpha when <paramref name="indexPrec2"/> is non-zero). Searches stop as soon as
    /// the error increases, exactly like DirectXTex.
    /// </summary>
    public static float ComputeError(in LdrColorA pixel, ReadOnlySpan<LdrColorA> palette, int indexPrec, int indexPrec2,
        out int bestIndex, out int bestIndex2)
    {
        int numIndices = 1 << indexPrec;
        int numIndices2 = 1 << indexPrec2;
        float fTotalErr = 0;
        float fBestErr = float.MaxValue;
        bestIndex = 0;
        bestIndex2 = 0;

        float pr = pixel.R, pg = pixel.G, pb = pixel.B, pa = pixel.A;

        if (indexPrec2 == 0)
        {
            for (int i = 0; i < numIndices && fBestErr > 0; i++)
            {
                ref readonly var c = ref palette[i];
                float dr = pr - c.R, dg = pg - c.G, db = pb - c.B, da = pa - c.A;
                // Compute ErrorMetric
                float fErr = dr * dr + dg * dg + db * db + da * da;
                if (fErr > fBestErr) break; // error increased, so we're done searching
                if (fErr < fBestErr)
                {
                    fBestErr = fErr;
                    bestIndex = i;
                }
            }
            fTotalErr += fBestErr;
        }
        else
        {
            for (int i = 0; i < numIndices && fBestErr > 0; i++)
            {
                ref readonly var c = ref palette[i];
                float dr = pr - c.R, dg = pg - c.G, db = pb - c.B;
                // Compute ErrorMetricRGB
                float fErr = dr * dr + dg * dg + db * db;
                if (fErr > fBestErr) break; // error increased, so we're done searching
                if (fErr < fBestErr)
                {
                    fBestErr = fErr;
                    bestIndex = i;
                }
            }
            fTotalErr += fBestErr;
            fBestErr = float.MaxValue;
            for (int i = 0; i < numIndices2 && fBestErr > 0; i++)
            {
                // Compute ErrorMetricAlpha
                float ea = pa - palette[i].A;
                float fErr = ea * ea;
                if (fErr > fBestErr) break; // error increased, so we're done searching
                if (fErr < fBestErr)
                {
                    fBestErr = fErr;
                    bestIndex2 = i;
                }
            }
            fTotalErr += fBestErr;
        }

        return fTotalErr;
    }
}
