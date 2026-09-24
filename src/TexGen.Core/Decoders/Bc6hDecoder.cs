//-------------------------------------------------------------------------------------
// Decoders/Bc6hDecoder.cs
//
// Reference BC6H decoder (CPU) used for verification, preview and DDS input. A port of
// texconv-js bc6h_decode.ts, itself a direct port of DirectXTex's D3DX_BC6H::Decode
// (BC6HBC7.cpp). Handles BC6H_UF16 and BC6H_SF16, producing R32G32B32A32_FLOAT.
//-------------------------------------------------------------------------------------

using System.Runtime.InteropServices;

namespace TexGen.Decoders;

public static class Bc6hDecoder
{
    // 2-subset partition table (g_aPartitionTable[1]): bit i = region (0/1) of pixel i.
    private static readonly ushort[] Partition2 =
    {
        0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80,
        0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000,
        0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
        0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
        0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
        0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
        0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
        0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
    };

    // Fix-up (anchor) index for region 1 in 2-region partitions (g_aFixUp[1][shape][1]).
    private static readonly byte[] FixUp2 =
    {
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
        15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
        6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
    };

    private static readonly int[] Weights3 = { 0, 9, 18, 27, 37, 46, 55, 64 };
    private static readonly int[] Weights4 = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

    // D3DX_BC6H::EField values. W = e0.A, X = e0.B, Y = e1.A, Z = e1.B.
    private const byte NA = 0, M = 1, D = 2;
    private const byte RW = 3, RX = 4, RY = 5, RZ = 6;
    private const byte GW = 7, GX = 8, GY = 9, GZ = 10;
    private const byte BW = 11, BX = 12, BY = 13, BZ = 14;

    // D3DX_BC6H::ms_aDesc — per-mode bit layout: (field, bit) pairs, 82 per mode.
    private static readonly byte[][] ModeDesc =
    {
        // Mode 1 (0x00) - 10 5 5 5
        [M,0, M,1, GY,4, BY,4, BZ,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, RW,8, RW,9, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, GW,8, GW,9, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BW,8, BW,9, RX,0, RX,1, RX,2, RX,3, RX,4,
         GZ,4, GY,0, GY,1, GY,2, GY,3, GX,0, GX,1, GX,2, GX,3, GX,4,
         BZ,0, GZ,0, GZ,1, GZ,2, GZ,3, BX,0, BX,1, BX,2, BX,3, BX,4,
         BZ,1, BY,0, BY,1, BY,2, BY,3, RY,0, RY,1, RY,2, RY,3, RY,4,
         BZ,2, RZ,0, RZ,1, RZ,2, RZ,3, RZ,4, BZ,3, D,0, D,1, D,2,
         D,3, D,4],
        // Mode 2 (0x01) - 7 6 6 6
        [M,0, M,1, GY,5, GZ,4, GZ,5, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, BZ,0, BZ,1, BY,4, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, BY,5, BZ,2, GY,4, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BZ,3, BZ,5, BZ,4, RX,0, RX,1, RX,2, RX,3, RX,4,
         RX,5, GY,0, GY,1, GY,2, GY,3, GX,0, GX,1, GX,2, GX,3, GX,4,
         GX,5, GZ,0, GZ,1, GZ,2, GZ,3, BX,0, BX,1, BX,2, BX,3, BX,4,
         BX,5, BY,0, BY,1, BY,2, BY,3, RY,0, RY,1, RY,2, RY,3, RY,4,
         RY,5, RZ,0, RZ,1, RZ,2, RZ,3, RZ,4, RZ,5, D,0, D,1, D,2,
         D,3, D,4],
        // Mode 3 (0x02) - 11 5 4 4
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, RW,8, RW,9, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, GW,8, GW,9, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BW,8, BW,9, RX,0, RX,1, RX,2, RX,3, RX,4,
         RW,10, GY,0, GY,1, GY,2, GY,3, GX,0, GX,1, GX,2, GX,3, GW,10,
         BZ,0, GZ,0, GZ,1, GZ,2, GZ,3, BX,0, BX,1, BX,2, BX,3, BW,10,
         BZ,1, BY,0, BY,1, BY,2, BY,3, RY,0, RY,1, RY,2, RY,3, RY,4,
         BZ,2, RZ,0, RZ,1, RZ,2, RZ,3, RZ,4, BZ,3, D,0, D,1, D,2,
         D,3, D,4],
        // Mode 4 (0x06) - 11 4 5 4
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, RW,8, RW,9, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, GW,8, GW,9, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BW,8, BW,9, RX,0, RX,1, RX,2, RX,3, RW,10,
         GZ,4, GY,0, GY,1, GY,2, GY,3, GX,0, GX,1, GX,2, GX,3, GX,4,
         GW,10, GZ,0, GZ,1, GZ,2, GZ,3, BX,0, BX,1, BX,2, BX,3, BW,10,
         BZ,1, BY,0, BY,1, BY,2, BY,3, RY,0, RY,1, RY,2, RY,3, BZ,0,
         BZ,2, RZ,0, RZ,1, RZ,2, RZ,3, GY,4, BZ,3, D,0, D,1, D,2,
         D,3, D,4],
        // Mode 5 (0x0a) - 11 4 4 5
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, RW,8, RW,9, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, GW,8, GW,9, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BW,8, BW,9, RX,0, RX,1, RX,2, RX,3, RW,10,
         BY,4, GY,0, GY,1, GY,2, GY,3, GX,0, GX,1, GX,2, GX,3, GW,10,
         BZ,0, GZ,0, GZ,1, GZ,2, GZ,3, BX,0, BX,1, BX,2, BX,3, BX,4,
         BW,10, BY,0, BY,1, BY,2, BY,3, RY,0, RY,1, RY,2, RY,3, BZ,1,
         BZ,2, RZ,0, RZ,1, RZ,2, RZ,3, BZ,4, BZ,3, D,0, D,1, D,2,
         D,3, D,4],
        // Mode 6 (0x0e) - 9 5 5 5
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, RW,8, BY,4, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, GW,8, GY,4, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BW,8, BZ,4, RX,0, RX,1, RX,2, RX,3, RX,4,
         GZ,4, GY,0, GY,1, GY,2, GY,3, GX,0, GX,1, GX,2, GX,3, GX,4,
         BZ,0, GZ,0, GZ,1, GZ,2, GZ,3, BX,0, BX,1, BX,2, BX,3, BX,4,
         BZ,1, BY,0, BY,1, BY,2, BY,3, RY,0, RY,1, RY,2, RY,3, RY,4,
         BZ,2, RZ,0, RZ,1, RZ,2, RZ,3, RZ,4, BZ,3, D,0, D,1, D,2,
         D,3, D,4],
        // Mode 7 (0x12) - 8 6 5 5
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, GZ,4, BY,4, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, BZ,2, GY,4, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BZ,3, BZ,4, RX,0, RX,1, RX,2, RX,3, RX,4,
         RX,5, GY,0, GY,1, GY,2, GY,3, GX,0, GX,1, GX,2, GX,3, GX,4,
         BZ,0, GZ,0, GZ,1, GZ,2, GZ,3, BX,0, BX,1, BX,2, BX,3, BX,4,
         BZ,1, BY,0, BY,1, BY,2, BY,3, RY,0, RY,1, RY,2, RY,3, RY,4,
         RY,5, RZ,0, RZ,1, RZ,2, RZ,3, RZ,4, RZ,5, D,0, D,1, D,2,
         D,3, D,4],
        // Mode 8 (0x16) - 8 5 6 5
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, BZ,0, BY,4, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, GY,5, GY,4, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, GZ,5, BZ,4, RX,0, RX,1, RX,2, RX,3, RX,4,
         GZ,4, GY,0, GY,1, GY,2, GY,3, GX,0, GX,1, GX,2, GX,3, GX,4,
         GX,5, GZ,0, GZ,1, GZ,2, GZ,3, BX,0, BX,1, BX,2, BX,3, BX,4,
         BZ,1, BY,0, BY,1, BY,2, BY,3, RY,0, RY,1, RY,2, RY,3, RY,4,
         BZ,2, RZ,0, RZ,1, RZ,2, RZ,3, RZ,4, BZ,3, D,0, D,1, D,2,
         D,3, D,4],
        // Mode 9 (0x1a) - 8 5 5 6
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, BZ,1, BY,4, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, BY,5, GY,4, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BZ,5, BZ,4, RX,0, RX,1, RX,2, RX,3, RX,4,
         GZ,4, GY,0, GY,1, GY,2, GY,3, GX,0, GX,1, GX,2, GX,3, GX,4,
         BZ,0, GZ,0, GZ,1, GZ,2, GZ,3, BX,0, BX,1, BX,2, BX,3, BX,4,
         BX,5, BY,0, BY,1, BY,2, BY,3, RY,0, RY,1, RY,2, RY,3, RY,4,
         BZ,2, RZ,0, RZ,1, RZ,2, RZ,3, RZ,4, BZ,3, D,0, D,1, D,2,
         D,3, D,4],
        // Mode 10 (0x1e) - 6 6 6 6
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, GZ,4, BZ,0, BZ,1, BY,4, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GY,5, BY,5, BZ,2, GY,4, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, GZ,5, BZ,3, BZ,5, BZ,4, RX,0, RX,1, RX,2, RX,3, RX,4,
         RX,5, GY,0, GY,1, GY,2, GY,3, GX,0, GX,1, GX,2, GX,3, GX,4,
         GX,5, GZ,0, GZ,1, GZ,2, GZ,3, BX,0, BX,1, BX,2, BX,3, BX,4,
         BX,5, BY,0, BY,1, BY,2, BY,3, RY,0, RY,1, RY,2, RY,3, RY,4,
         RY,5, RZ,0, RZ,1, RZ,2, RZ,3, RZ,4, RZ,5, D,0, D,1, D,2,
         D,3, D,4],
        // Mode 11 (0x03) - 10 10
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, RW,8, RW,9, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, GW,8, GW,9, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BW,8, BW,9, RX,0, RX,1, RX,2, RX,3, RX,4,
         RX,5, RX,6, RX,7, RX,8, RX,9, GX,0, GX,1, GX,2, GX,3, GX,4,
         GX,5, GX,6, GX,7, GX,8, GX,9, BX,0, BX,1, BX,2, BX,3, BX,4,
         BX,5, BX,6, BX,7, BX,8, BX,9, NA,0, NA,0, NA,0, NA,0, NA,0,
         NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0,
         NA,0, NA,0],
        // Mode 12 (0x07) - 11 9
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, RW,8, RW,9, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, GW,8, GW,9, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BW,8, BW,9, RX,0, RX,1, RX,2, RX,3, RX,4,
         RX,5, RX,6, RX,7, RX,8, RW,10, GX,0, GX,1, GX,2, GX,3, GX,4,
         GX,5, GX,6, GX,7, GX,8, GW,10, BX,0, BX,1, BX,2, BX,3, BX,4,
         BX,5, BX,6, BX,7, BX,8, BW,10, NA,0, NA,0, NA,0, NA,0, NA,0,
         NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0,
         NA,0, NA,0],
        // Mode 13 (0x0b) - 12 8
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, RW,8, RW,9, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, GW,8, GW,9, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BW,8, BW,9, RX,0, RX,1, RX,2, RX,3, RX,4,
         RX,5, RX,6, RX,7, RW,11, RW,10, GX,0, GX,1, GX,2, GX,3, GX,4,
         GX,5, GX,6, GX,7, GW,11, GW,10, BX,0, BX,1, BX,2, BX,3, BX,4,
         BX,5, BX,6, BX,7, BW,11, BW,10, NA,0, NA,0, NA,0, NA,0, NA,0,
         NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0,
         NA,0, NA,0],
        // Mode 14 (0x0f) - 16 4
        [M,0, M,1, M,2, M,3, M,4, RW,0, RW,1, RW,2, RW,3, RW,4,
         RW,5, RW,6, RW,7, RW,8, RW,9, GW,0, GW,1, GW,2, GW,3, GW,4,
         GW,5, GW,6, GW,7, GW,8, GW,9, BW,0, BW,1, BW,2, BW,3, BW,4,
         BW,5, BW,6, BW,7, BW,8, BW,9, RX,0, RX,1, RX,2, RX,3, RW,15,
         RW,14, RW,13, RW,12, RW,11, RW,10, GX,0, GX,1, GX,2, GX,3, GW,15,
         GW,14, GW,13, GW,12, GW,11, GW,10, BX,0, BX,1, BX,2, BX,3, BW,15,
         BW,14, BW,13, BW,12, BW,11, BW,10, NA,0, NA,0, NA,0, NA,0, NA,0,
         NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0, NA,0,
         NA,0, NA,0],
    };

    private readonly record struct ModeInfo(int Partitions, bool Transformed, int IndexPrec, int[] Prec);

    // D3DX_BC6H::ms_aInfo. Prec = [r0A, g0A, b0A, r0B, g0B, b0B, r1A, g1A, b1A, r1B, g1B, b1B].
    private static readonly ModeInfo[] Info =
    {
        new(1, true, 3, [10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5]),    // Mode 1
        new(1, true, 3, [7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6]),       // Mode 2
        new(1, true, 3, [11, 11, 11, 5, 4, 4, 5, 4, 4, 5, 4, 4]),    // Mode 3
        new(1, true, 3, [11, 11, 11, 4, 5, 4, 4, 5, 4, 4, 5, 4]),    // Mode 4
        new(1, true, 3, [11, 11, 11, 4, 4, 5, 4, 4, 5, 4, 4, 5]),    // Mode 5
        new(1, true, 3, [9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5]),       // Mode 6
        new(1, true, 3, [8, 8, 8, 6, 5, 5, 6, 5, 5, 6, 5, 5]),       // Mode 7
        new(1, true, 3, [8, 8, 8, 5, 6, 5, 5, 6, 5, 5, 6, 5]),       // Mode 8
        new(1, true, 3, [8, 8, 8, 5, 5, 6, 5, 5, 6, 5, 5, 6]),       // Mode 9
        new(1, false, 3, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),      // Mode 10
        new(0, false, 4, [10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0]), // Mode 11
        new(0, true, 4, [11, 11, 11, 9, 9, 9, 0, 0, 0, 0, 0, 0]),     // Mode 12
        new(0, true, 4, [12, 12, 12, 8, 8, 8, 0, 0, 0, 0, 0, 0]),     // Mode 13
        new(0, true, 4, [16, 16, 16, 4, 4, 4, 0, 0, 0, 0, 0, 0]),     // Mode 14
    };

    // D3DX_BC6H::ms_aModeToInfo — 5-bit mode field -> Info index (-1 = invalid/reserved).
    private static readonly sbyte[] ModeToInfo =
    {
        0, 1, 2, 10, -1, -1, 3, 11, -1, -1, 4, 12, -1, -1, 5, 13,
        -1, -1, 6, -1, -1, -1, 7, -1, -1, -1, 8, -1, -1, -1, 9, -1,
    };

    private ref struct BitReader(ReadOnlySpan<byte> bytes)
    {
        private readonly ReadOnlySpan<byte> _bytes = bytes;
        public int Bit;

        public int ReadBit()
        {
            int b = (_bytes[Bit >> 3] >> (Bit & 7)) & 1;
            Bit++;
            return b;
        }

        public int Read(int n)
        {
            int v = 0;
            for (int i = 0; i < n; i++) v |= ReadBit() << i;
            return v;
        }
    }

    // SIGN_EXTEND(x, nb) from BC6HBC7.cpp.
    private static int SignExtend(int x, int nb) => ((x & (1 << (nb - 1))) != 0 ? ~0 ^ ((1 << nb) - 1) : 0) | x;

    // D3DX_BC6H::Unquantize
    private static int Unquantize(int comp, int bitsPerComp, bool signed)
    {
        if (signed)
        {
            if (bitsPerComp >= 16) return comp;
            bool s = comp < 0;
            if (s) comp = -comp;
            int unq;
            if (comp == 0) unq = 0;
            else if (comp >= (1 << (bitsPerComp - 1)) - 1) unq = 0x7FFF;
            else unq = ((comp << 15) + 0x4000) >> (bitsPerComp - 1);
            return s ? -unq : unq;
        }
        if (bitsPerComp >= 15) return comp;
        if (comp == 0) return 0;
        if (comp == (1 << bitsPerComp) - 1) return 0xFFFF;
        return ((comp << 16) + 0x8000) >> bitsPerComp;
    }

    // D3DX_BC6H::FinishUnquantize
    private static int FinishUnquantize(int comp, bool signed)
    {
        if (signed) return comp < 0 ? -((-comp * 31) >> 5) : (comp * 31) >> 5; // scale by 31/32
        return (comp * 31) >> 6; // scale by 31/64
    }

    // INTColor::INT2F16
    private static ushort Int2F16(int input, bool signed)
    {
        if (!signed) return (ushort)input;
        int s = 0;
        if (input < 0)
        {
            s = 0x8000;
            input = -input;
        }
        return (ushort)(s | input);
    }

    private static void FillBlack(Span<float> o)
    {
        for (int i = 0; i < 16; i++)
        {
            o[i * 4] = 0;
            o[i * 4 + 1] = 0;
            o[i * 4 + 2] = 0;
            o[i * 4 + 3] = 1;
        }
    }

    /// <summary>Decode one 16-byte BC6H block into 16 RGBA float pixels (row-major).</summary>
    public static void DecodeBlock(ReadOnlySpan<byte> block, bool signed, Span<float> o)
    {
        var r = new BitReader(block);
        int uMode = r.Read(2);
        if (uMode != 0x00 && uMode != 0x01) uMode = (r.Read(3) << 2) | uMode;

        int infoIdx = ModeToInfo[uMode];
        if (infoIdx < 0)
        {
            // Per the BC6H spec, reserved/invalid modes decode to opaque black.
            FillBlack(o);
            return;
        }

        var desc = ModeDesc[infoIdx];
        var info = Info[infoIdx];

        // Endpoints [e0.A, e0.B, e1.A, e1.B] x [r, g, b].
        Span<int> ep = stackalloc int[12];
        int uShape = 0;

        int headerBits = info.Partitions > 0 ? 82 : 65;
        while (r.Bit < headerBits)
        {
            int cur = r.Bit;
            if (r.ReadBit() == 0) continue;
            int field = desc[cur * 2];
            int bit = desc[cur * 2 + 1];
            if (field == D)
            {
                uShape |= 1 << bit;
            }
            else if (field is >= RW and <= BZ)
            {
                ep[((field - RW) & 3) * 3 + ((field - RW) >> 2)] |= 1 << bit;
            }
            else
            {
                // Invalid header bits encountered during decoding.
                FillBlack(o);
                return;
            }
        }

        var prec = info.Prec;
        // Sign extend necessary endpoints.
        if (signed)
            for (int c = 0; c < 3; c++) ep[c] = SignExtend(ep[c], prec[c]);
        if (signed || info.Transformed)
        {
            for (int p = 0; p <= info.Partitions; p++)
            {
                if (p != 0)
                    for (int c = 0; c < 3; c++) ep[p * 6 + c] = SignExtend(ep[p * 6 + c], prec[p * 6 + c]);
                for (int c = 0; c < 3; c++) ep[p * 6 + 3 + c] = SignExtend(ep[p * 6 + 3 + c], prec[p * 6 + 3 + c]);
            }
        }

        // Inverse transform the endpoints (deltas anchored at e0.A).
        if (info.Transformed)
        {
            for (int c = 0; c < 3; c++)
            {
                int mask = (1 << prec[c]) - 1;
                for (int e = 1; e < 4; e++)
                {
                    ep[e * 3 + c] = (ep[e * 3 + c] + ep[c]) & mask;
                    if (signed) ep[e * 3 + c] = SignExtend(ep[e * 3 + c], prec[c]);
                }
            }
        }

        // Unquantize endpoints (always with region-0 endpoint-A precision, per the C++).
        Span<int> unq = stackalloc int[12];
        for (int e = 0; e < 4; e++)
            for (int c = 0; c < 3; c++)
                unq[e * 3 + c] = Unquantize(ep[e * 3 + c], prec[c], signed);

        // Read indices and interpolate.
        var weights = info.Partitions > 0 ? Weights3 : Weights4;
        for (int i = 0; i < 16; i++)
        {
            bool fixup = i == 0 || (info.Partitions == 1 && i == FixUp2[uShape]);
            int numBits = fixup ? info.IndexPrec - 1 : info.IndexPrec;
            if (r.Bit + numBits > 128)
            {
                FillBlack(o);
                return;
            }
            int index = r.Read(numBits);
            if (index >= (info.Partitions > 0 ? 8 : 16))
            {
                FillBlack(o);
                return;
            }

            int region = info.Partitions > 0 ? (Partition2[uShape] >> i) & 1 : 0;
            int a = region * 6, b = region * 6 + 3;
            int w = weights[index];
            int iw = 64 - w;
            for (int c = 0; c < 3; c++)
            {
                int v = FinishUnquantize((unq[a + c] * iw + unq[b + c] * w + 32) >> 6, signed);
                o[i * 4 + c] = (float)BitConverter.UInt16BitsToHalf(Int2F16(v, signed));
            }
            o[i * 4 + 3] = 1f;
        }
    }

    /// <summary>Decode a BC6H_UF16 / BC6H_SF16 image to R32G32B32A32_FLOAT.</summary>
    public static Image Decode(Image src)
    {
        bool signed = src.Format switch
        {
            DxgiFormat.BC6H_UF16 => false,
            DxgiFormat.BC6H_SF16 => true,
            _ => throw new NotSupportedException($"Bc6hDecoder: unsupported format {src.Format} (expected BC6H_UF16 or BC6H_SF16)"),
        };
        var dst = Image.Create(DxgiFormat.R32G32B32A32_FLOAT, src.Width, src.Height);
        BlockDecode.Scatter(src, dst, 16, 16,
            (block, px) => DecodeBlock(block, signed, MemoryMarshal.Cast<byte, float>(px)));
        return dst;
    }
}
