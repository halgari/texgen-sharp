//-------------------------------------------------------------------------------------
// FallbackConverter.cs
//
// Converter.Convert on the GPU while the GPU works, and on the CPU from the moment it
// does not. A driver that throws, a device that will not open, and a kernel that never
// comes back all end the same way: the image in hand is converted on the CPU, and so is
// every image after it.
//-------------------------------------------------------------------------------------

using TexGen.Gpu;

namespace TexGen;

/// <summary>
/// Converts images on the GPU, falling back to the CPU for good the first time the GPU fails.
/// Safe to call from many threads at once.
/// <para>
/// A GPU failure is only believed once the CPU has converted the same image: an image the
/// CPU cannot convert either is a problem with the image or the options, so that exception
/// is thrown and the GPU stays in use. A conversion that has not finished within
/// <see cref="GpuTimeout"/> counts as a failure too. Its thread is abandoned rather than
/// waited on, because a hung kernel does not return, and the device is never disposed.
/// </para>
/// <para>
/// The GPU is used by one conversion at a time, as <see cref="Converter"/> requires, and the
/// timeout runs from when a conversion gets the GPU, not from when it started waiting for
/// it, so a queue of large images is not mistaken for a hang.
/// </para>
/// </summary>
public sealed class FallbackConverter : IDisposable
{
    /// <summary>Long enough for an 8K BC7 on a slow GPU; a hang costs this once.</summary>
    public static readonly TimeSpan DefaultGpuTimeout = TimeSpan.FromMinutes(2);

    private readonly Func<Image, ConvertOptions, ScratchImage>? _gpu;
    private readonly Func<Image, ConvertOptions, ScratchImage> _cpu;
    private readonly Action<string>? _log;
    private readonly Func<Image, ConvertOptions, bool> _needsGpu;
    private readonly SemaphoreSlim _gpuTurn = new(1, 1);
    private readonly Lock _devices = new();
    private readonly GpuBackend _backend;
    private readonly int _adapterIndex;
    private GpuDevice? _gpuDevice;
    private GpuDevice? _cpuDevice;
    private volatile string? _gpuFailure;

    /// <param name="backend">
    /// The GPU to try. <see cref="GpuBackend.Cpu"/> converts on the CPU from the start, as texconv's
    /// <c>-nogpu</c> does.
    /// </param>
    /// <param name="adapterIndex">Which adapter of that backend, as texconv's <c>-gpu</c>.</param>
    /// <param name="gpuTimeout">How long a conversion may hold the GPU; <see cref="DefaultGpuTimeout"/> when null.</param>
    /// <param name="log">Told once, in a sentence, when and why conversion moved to the CPU.</param>
    public FallbackConverter(GpuBackend backend = GpuBackend.Auto, int adapterIndex = 0, TimeSpan? gpuTimeout = null,
        Action<string>? log = null)
    {
        _backend = backend;
        _adapterIndex = adapterIndex;
        _log = log;
        GpuTimeout = gpuTimeout ?? DefaultGpuTimeout;
        _needsGpu = Converter.NeedsGpu;
        _cpu = (image, options) => Converter.Convert(image, options with { Codec = CodecPreference.Cpu }, CpuDevice());
        if (backend == GpuBackend.Cpu) _gpuFailure = "the CPU was asked for";
        else _gpu = (image, options) => Converter.Convert(image, options, GpuDevice());
    }

    /// <summary>For tests: the two conversions are given rather than built from devices.</summary>
    internal FallbackConverter(Func<Image, ConvertOptions, ScratchImage>? gpu,
        Func<Image, ConvertOptions, ScratchImage> cpu, TimeSpan gpuTimeout, Action<string>? log = null)
    {
        _gpu = gpu;
        _cpu = cpu;
        _log = log;
        GpuTimeout = gpuTimeout;
        _needsGpu = (_, _) => true;
        if (gpu is null) _gpuFailure = "no GPU was given";
    }

    public TimeSpan GpuTimeout { get; }

    /// <summary>True until the GPU has failed once (or was never to be used).</summary>
    public bool UsingGpu => _gpuFailure is null;

    /// <summary>Why conversion moved to the CPU, or null while it is still on the GPU.</summary>
    public string? GpuFailure => _gpuFailure;

    /// <summary>The GPU in use, once one has been opened; null before that and when there is none.</summary>
    public string? GpuDescription
    {
        get
        {
            lock (_devices) return _gpuDevice?.Description;
        }
    }

    public ScratchImage Convert(Image source, ConvertOptions options, CancellationToken token = default)
    {
        if (_gpu is null || _gpuFailure is not null || !_needsGpu(source, options)) return _cpu(source, options);

        // Wait for the GPU, but stop waiting the moment another conversion finds it broken:
        // a hung conversion never gives its turn back.
        while (!_gpuTurn.Wait(250, token))
            if (_gpuFailure is not null) return _cpu(source, options);
        if (_gpuFailure is not null)
        {
            _gpuTurn.Release();
            return _cpu(source, options);
        }

        var work = OnOwnThread(() => _gpu(source, options));
        // The turn passes on when the GPU is done with this image, whoever stopped waiting for it.
        _ = work.ContinueWith(_ => _gpuTurn.Release(), TaskScheduler.Default);

        bool finished;
        try
        {
            finished = work.Wait(GpuTimeout, token);
        }
        catch (AggregateException ex) when (ex.InnerException is not null)
        {
            // Thrown here only if the CPU cannot do it either, which makes it the image's problem.
            var result = _cpu(source, options);
            Fail(ex.InnerException is NoGpuException
                ? $"there is no GPU to use ({ex.InnerException.Message})"
                : $"the GPU failed on an image the CPU converted ({ex.InnerException.GetType().Name}: {ex.InnerException.Message})");
            return result;
        }

        if (finished) return work.Result;

        // Failed before the CPU is tried: the hung conversion still holds the GPU, and whatever
        // the CPU makes of this image, nothing else may queue behind it.
        Fail($"a GPU conversion had not finished after {GpuTimeout.TotalSeconds:0} seconds");
        return _cpu(source, options);
    }

    private sealed class NoGpuException(string message, Exception? inner = null) : Exception(message, inner);

    /// <summary>
    /// Runs a GPU conversion on a thread of its own rather than the thread pool. A busy pool would start it late,
    /// and the wait would count against the timeout; and a conversion that hangs keeps its thread for good, which
    /// should not be one the pool expects back. Background, so a hung one does not keep the process alive.
    /// </summary>
    private static Task<ScratchImage> OnOwnThread(Func<ScratchImage> convert)
    {
        var done = new TaskCompletionSource<ScratchImage>(TaskCreationOptions.RunContinuationsAsynchronously);
        new Thread(() =>
        {
            try
            {
                done.SetResult(convert());
            }
            catch (Exception ex)
            {
                done.SetException(ex);
            }
        }) { IsBackground = true, Name = "texgen GPU conversion" }.Start();
        return done.Task;
    }

    private void Fail(string why)
    {
        if (Interlocked.CompareExchange(ref _gpuFailure, why, null) is null)
            _log?.Invoke($"Converting on the CPU from now on: {why}");
    }

    private GpuDevice GpuDevice()
    {
        lock (_devices)
        {
            if (_gpuDevice is not null) return _gpuDevice;
            GpuDevice device;
            try
            {
                device = Gpu.GpuDevice.Create(_backend, _adapterIndex);
            }
            catch (Exception ex)
            {
                throw new NoGpuException(ex.Message, ex);
            }

            if (!device.IsHardware)
            {
                device.Dispose();
                throw new NoGpuException("neither CUDA nor OpenCL found an adapter");
            }

            return _gpuDevice = device;
        }
    }

    private GpuDevice CpuDevice()
    {
        lock (_devices) return _cpuDevice ??= Gpu.GpuDevice.Create(GpuBackend.Cpu);
    }

    public void Dispose()
    {
        lock (_devices)
        {
            _cpuDevice?.Dispose();
            // A device that has hung may hang again being disposed; it is left to the process.
            if (_gpuFailure is null) _gpuDevice?.Dispose();
        }
    }
}
