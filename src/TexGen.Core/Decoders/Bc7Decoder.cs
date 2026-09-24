//-------------------------------------------------------------------------------------
// Decoders/Bc7Decoder.cs
//
// Reference BC7 decoder (CPU), used for verification, preview and DDS input.
// Implements the BPTC UNORM decode per the D3D11 spec (port of texconv-js
// bc7_decode.ts).
//-------------------------------------------------------------------------------------

namespace TexGen.Decoders;

public static class Bc7Decoder
{
    // 2-subset partition table: bit i = subset (0/1) of pixel i.
    internal static readonly ushort[] Partition2 =
    [
        0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80,
        0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000,
        0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
        0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
        0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
        0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
        0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
        0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
    ];

    // 3-subset partition table: 2 bits per pixel = subset (0/1/2).
    internal static readonly uint[] Partition3 =
    [
        0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
        0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
        0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
        0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
        0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
        0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
        0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
        0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
    ];

    // Anchor (fix-up) index for subset 1 in 2-subset partitions.
    private static readonly byte[] Anchor2 =
    [
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
        15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
        6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
    ];

    // Anchors for subsets 1 and 2 in 3-subset partitions.
    private static readonly byte[] Anchor3A =
    [
        3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3,
        3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
        8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15,
        3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
    ];

    private static readonly byte[] Anchor3B =
    [
        15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8,
        15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
        15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8,
        15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
    ];

    private static readonly int[] Weights2 = [0, 21, 43, 64];
    private static readonly int[] Weights3 = [0, 9, 18, 27, 37, 46, 55, 64];
    private static readonly int[] Weights4 = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

    /// <param name="Ns">subsets</param>
    /// <param name="Pb">partition bits</param>
    /// <param name="Rb">rotation bits</param>
    /// <param name="Isb">index selector bits</param>
    /// <param name="Cb">color bits</param>
    /// <param name="Ab">alpha bits</param>
    /// <param name="Epb">per-endpoint p-bits</param>
    /// <param name="Spb">shared p-bits</param>
    /// <param name="Ib">index bits</param>
    /// <param name="Ib2">secondary index bits</param>
    private readonly record struct ModeInfo(int Ns, int Pb, int Rb, int Isb, int Cb, int Ab, int Epb, int Spb, int Ib, int Ib2);

    private static readonly ModeInfo[] Modes =
    [
        new(3, 4, 0, 0, 4, 0, 1, 0, 3, 0),
        new(2, 6, 0, 0, 6, 0, 0, 1, 3, 0),
        new(3, 6, 0, 0, 5, 0, 0, 0, 2, 0),
        new(2, 6, 0, 0, 7, 0, 1, 0, 2, 0),
        new(1, 0, 2, 1, 5, 6, 0, 0, 2, 3),
        new(1, 0, 2, 0, 7, 8, 0, 0, 2, 2),
        new(1, 0, 0, 0, 7, 7, 1, 0, 4, 0),
        new(2, 6, 0, 0, 5, 5, 1, 0, 2, 0),
    ];

    /// <summary>LSB-first bit reader over one 16-byte block.</summary>
    private ref struct BitReader(ReadOnlySpan<byte> bytes)
    {
        private readonly ReadOnlySpan<byte> _bytes = bytes;
        private int _bit;

        public int Read(int n)
        {
            int v = 0;
            for (int i = 0; i < n; i++)
            {
                v |= ((_bytes[_bit >> 3] >> (_bit & 7)) & 1) << i;
                _bit++;
            }
            return v;
        }
    }

    private static int SubsetOf(in ModeInfo m, int partition, int pixel) => m.Ns switch
    {
        1 => 0,
        2 => (Partition2[partition] >> pixel) & 1,
        _ => (int)((Partition3[partition] >> (pixel * 2)) & 3),
    };

    private static bool IsAnchor(in ModeInfo m, int partition, int pixel)
    {
        if (pixel == 0) return true;
        if (m.Ns == 2) return pixel == Anchor2[partition];
        if (m.Ns == 3) return pixel == Anchor3A[partition] || pixel == Anchor3B[partition];
        return false;
    }

    private static int Expand(int value, int bits)
    {
        int v = (value << (8 - bits)) & 0xff;
        return (v | (v >> bits)) & 0xff;
    }

    private static int[] WeightsFor(int bits) => bits == 2 ? Weights2 : bits == 3 ? Weights3 : Weights4;

    /// <summary>Decode one 16-byte BC7 block into 16 RGBA8 pixels (row-major, 64 bytes).</summary>
    public static void DecodeBlock(ReadOnlySpan<byte> block, Span<byte> rgba64)
    {
        // Mode = index of the first set bit (LSB first).
        int mode = 0;
        while (mode < 8 && ((block[mode >> 3] >> (mode & 7)) & 1) == 0) mode++;
        if (mode >= 8)
        {
            // Reserved/invalid mode -> zero.
            rgba64[..64].Clear();
            return;
        }

        var m = Modes[mode];
        var r = new BitReader(block);
        r.Read(mode + 1); // consume mode bits

        int partition = m.Pb != 0 ? r.Read(m.Pb) : 0;
        int rotation = m.Rb != 0 ? r.Read(m.Rb) : 0;
        int indexSelector = m.Isb != 0 ? r.Read(m.Isb) : 0;

        int numEndpoints = m.Ns * 2;
        Span<int> rC = stackalloc int[6], gC = stackalloc int[6], bC = stackalloc int[6], aC = stackalloc int[6];
        for (int i = 0; i < numEndpoints; i++) rC[i] = r.Read(m.Cb);
        for (int i = 0; i < numEndpoints; i++) gC[i] = r.Read(m.Cb);
        for (int i = 0; i < numEndpoints; i++) bC[i] = r.Read(m.Cb);
        for (int i = 0; i < numEndpoints; i++) aC[i] = m.Ab != 0 ? r.Read(m.Ab) : 255;

        // p-bits
        Span<int> pbits = stackalloc int[6];
        if (m.Epb != 0)
        {
            for (int i = 0; i < numEndpoints; i++) pbits[i] = r.Read(1);
        }
        else if (m.Spb != 0)
        {
            Span<int> sp = stackalloc int[3];
            for (int s = 0; s < m.Ns; s++) sp[s] = r.Read(1);
            for (int i = 0; i < numEndpoints; i++) pbits[i] = sp[i >> 1];
        }

        // Assemble full 8-bit endpoints.
        Span<int> epR = stackalloc int[6], epG = stackalloc int[6], epB = stackalloc int[6], epA = stackalloc int[6];
        for (int i = 0; i < numEndpoints; i++)
        {
            int cb = m.Cb, ab = m.Ab;
            int rr = rC[i], gg = gC[i], bb = bC[i], aa = aC[i];
            if (m.Epb != 0 || m.Spb != 0)
            {
                rr = (rr << 1) | pbits[i];
                gg = (gg << 1) | pbits[i];
                bb = (bb << 1) | pbits[i];
                if (m.Ab != 0) aa = (aa << 1) | pbits[i];
                cb += 1;
                if (m.Ab != 0) ab += 1;
            }
            epR[i] = Expand(rr, cb);
            epG[i] = Expand(gg, cb);
            epB[i] = Expand(bb, cb);
            epA[i] = m.Ab != 0 ? Expand(aa, ab) : 255;
        }

        // Indices: primary first (anchors carry one fewer bit), then secondary (modes 4/5).
        Span<int> primary = stackalloc int[16];
        for (int p = 0; p < 16; p++)
        {
            int bits = m.Ib - (IsAnchor(m, partition, p) ? 1 : 0);
            primary[p] = bits > 0 ? r.Read(bits) : 0;
        }
        Span<int> secondary = stackalloc int[16];
        if (m.Ib2 != 0)
        {
            for (int p = 0; p < 16; p++)
            {
                int bits = m.Ib2 - (p == 0 ? 1 : 0);
                secondary[p] = bits > 0 ? r.Read(bits) : 0;
            }
        }

        var wPrimary = WeightsFor(m.Ib);
        var wSecondary = WeightsFor(m.Ib2);

        for (int p = 0; p < 16; p++)
        {
            int s = SubsetOf(m, partition, p);
            int e0 = s * 2, e1 = s * 2 + 1;

            int wc, wa;
            if (m.Ib2 != 0)
            {
                // Modes 4/5: separate color and alpha index sets, optionally swapped.
                if (indexSelector != 0)
                {
                    wc = wSecondary[secondary[p]];
                    wa = wPrimary[primary[p]];
                }
                else
                {
                    wc = wPrimary[primary[p]];
                    wa = wSecondary[secondary[p]];
                }
            }
            else
            {
                wc = wPrimary[primary[p]];
                wa = wc;
            }

            int R = (epR[e0] * (64 - wc) + epR[e1] * wc + 32) >> 6;
            int G = (epG[e0] * (64 - wc) + epG[e1] * wc + 32) >> 6;
            int B = (epB[e0] * (64 - wc) + epB[e1] * wc + 32) >> 6;
            int A = (epA[e0] * (64 - wa) + epA[e1] * wa + 32) >> 6;

            // Rotation (modes 4/5): swap alpha with the selected channel.
            if (rotation == 1) (A, R) = (R, A);
            else if (rotation == 2) (A, G) = (G, A);
            else if (rotation == 3) (A, B) = (B, A);

            int o = p * 4;
            rgba64[o] = (byte)R;
            rgba64[o + 1] = (byte)G;
            rgba64[o + 2] = (byte)B;
            rgba64[o + 3] = (byte)A;
        }
    }

    /// <summary>Decode a full BC7 image to an R8G8B8A8_UNORM image.</summary>
    public static Image Decode(Image src)
    {
        if (src.Format is not (DxgiFormat.BC7_UNORM or DxgiFormat.BC7_UNORM_SRGB or DxgiFormat.BC7_TYPELESS))
            throw new NotSupportedException($"Bc7Decoder: unsupported format {src.Format}");
        var dst = Image.Create(DxgiFormat.R8G8B8A8_UNORM, src.Width, src.Height);
        BlockDecode.Scatter(src, dst, 16, 4, DecodeBlock);
        return dst;
    }
}
