using System.Diagnostics;

namespace TexGen.Tests;

/// <summary>
/// When <see cref="FallbackConverter"/> stops using the GPU, with stand-ins for the two conversions so no
/// hardware is needed. A GPU result is 8 pixels wide and a CPU result 16, so each test can see which ran.
/// </summary>
public sealed class FallbackConverterTests : IDisposable
{
    private const int GpuWidth = 8;
    private const int CpuWidth = 16;

    private readonly ManualResetEventSlim _neverFinishes = new(false);
    private readonly List<string> _log = new();
    private int _gpuCalls;
    private int _cpuCalls;

    public void Dispose()
    {
        // Lets the abandoned "hung" conversions end with the test.
        _neverFinishes.Set();
    }

    private static readonly Image Source = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 4, 4);
    private static readonly ConvertOptions Options = new();

    private static ScratchImage Result(int width) => ScratchImage.From2D(Image.Create(DxgiFormat.R8G8B8A8_UNORM, width, width));

    private ScratchImage Cpu(Image image, ConvertOptions options)
    {
        Interlocked.Increment(ref _cpuCalls);
        return Result(CpuWidth);
    }

    private Func<Image, ConvertOptions, ScratchImage> Gpu(Action behave) => (_, _) =>
    {
        Interlocked.Increment(ref _gpuCalls);
        behave();
        return Result(GpuWidth);
    };

    private FallbackConverter Converter(Func<Image, ConvertOptions, ScratchImage>? gpu,
        Func<Image, ConvertOptions, ScratchImage>? cpu = null, int timeoutMs = 5000)
    {
        return new FallbackConverter(gpu, cpu ?? Cpu, TimeSpan.FromMilliseconds(timeoutMs), message =>
        {
            lock (_log) _log.Add(message);
        });
    }

    [Fact]
    public void UsesTheGpuWhileItWorks()
    {
        var converter = Converter(Gpu(() => { }));

        Assert.Equal(GpuWidth, converter.Convert(Source, Options).BaseImage.Width);
        Assert.Equal(GpuWidth, converter.Convert(Source, Options).BaseImage.Width);

        Assert.True(converter.UsingGpu);
        Assert.Equal(0, _cpuCalls);
        Assert.Empty(_log);
    }

    [Fact]
    public void AGpuErrorMovesThisAndEveryLaterConversionToTheCpu()
    {
        var converter = Converter(Gpu(() => throw new InvalidOperationException("device lost")));

        Assert.Equal(CpuWidth, converter.Convert(Source, Options).BaseImage.Width);
        Assert.Equal(CpuWidth, converter.Convert(Source, Options).BaseImage.Width);

        Assert.Equal(1, _gpuCalls);
        Assert.False(converter.UsingGpu);
        Assert.Contains("device lost", converter.GpuFailure);
        Assert.Single(_log);
    }

    /// <summary>An image the CPU cannot convert either says nothing about the GPU.</summary>
    [Fact]
    public void AnImageNeitherCanConvertIsThrownAndTheGpuStays()
    {
        var converter = Converter(Gpu(() => throw new ArgumentException("bad options")),
            (_, _) => throw new ArgumentException("bad options"));

        Assert.Throws<ArgumentException>(() => converter.Convert(Source, Options));

        Assert.True(converter.UsingGpu);
        Assert.Empty(_log);
    }

    [Fact]
    public void AHungGpuIsAbandonedAndTheCpuTakesOver()
    {
        var converter = Converter(Gpu(() => _neverFinishes.Wait()), timeoutMs: 200);
        var watch = Stopwatch.StartNew();

        Assert.Equal(CpuWidth, converter.Convert(Source, Options).BaseImage.Width);
        Assert.Equal(CpuWidth, converter.Convert(Source, Options).BaseImage.Width);

        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(5));
        Assert.Equal(1, _gpuCalls);
        Assert.Contains("had not finished", converter.GpuFailure);
    }

    /// <summary>A hung conversion never hands the GPU on, so the ones queued behind it must not wait for it.</summary>
    [Fact]
    public async Task ConversionsQueuedBehindAHungGpuMoveToTheCpu()
    {
        var converter = Converter(Gpu(() => _neverFinishes.Wait()), timeoutMs: 300);

        var all = Enumerable.Range(0, 6).Select(_ => Task.Run(() => converter.Convert(Source, Options))).ToArray();
        var everyOne = Task.WhenAll(all);

        Assert.Same(everyOne, await Task.WhenAny(everyOne, Task.Delay(TimeSpan.FromSeconds(10))));
        Assert.All(all, t => Assert.Equal(CpuWidth, t.Result.BaseImage.Width));
        Assert.Equal(1, _gpuCalls);
    }

    /// <summary>The timeout runs from when a conversion gets the GPU, so a long queue is not taken for a hang.</summary>
    [Fact]
    public async Task WaitingForTheGpuDoesNotCountTowardsTheTimeout()
    {
        var converter = Converter(Gpu(() => Thread.Sleep(150)), timeoutMs: 400);

        var all = Enumerable.Range(0, 6).Select(_ => Task.Run(() => converter.Convert(Source, Options))).ToArray();
        await Task.WhenAll(all);

        Assert.All(all, t => Assert.Equal(GpuWidth, t.Result.BaseImage.Width));
        Assert.True(converter.UsingGpu);
        Assert.Equal(0, _cpuCalls);
    }

    /// <summary>A hung GPU is given up on even when the CPU cannot convert that image either.</summary>
    [Fact]
    public void AHungGpuIsGivenUpOnWhateverTheCpuMakesOfTheImage()
    {
        var converter = Converter(Gpu(() => _neverFinishes.Wait()), (_, _) => throw new ArgumentException("bad options"),
            timeoutMs: 200);

        Assert.Throws<ArgumentException>(() => converter.Convert(Source, Options));

        Assert.False(converter.UsingGpu);
    }

    [Fact]
    public void WithoutAGpuEverythingIsConvertedOnTheCpu()
    {
        var converter = Converter(null);

        Assert.Equal(CpuWidth, converter.Convert(Source, Options).BaseImage.Width);
        Assert.False(converter.UsingGpu);
        Assert.Empty(_log);
    }

    /// <summary>The real thing, on the CPU: texconv's -nogpu.</summary>
    [Fact]
    public void ACpuBackendConvertsOnTheCpuFromTheStart()
    {
        using var converter = new FallbackConverter(TexGen.Gpu.GpuBackend.Cpu);
        var image = Image.Create(DxgiFormat.R8G8B8A8_UNORM, 8, 8);

        var result = converter.Convert(image, new ConvertOptions { Format = DxgiFormat.BC1_UNORM });

        Assert.Equal(DxgiFormat.BC1_UNORM, result.Metadata.Format);
        Assert.False(converter.UsingGpu);
    }
}
