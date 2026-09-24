//-------------------------------------------------------------------------------------
// Gpu/GpuDevice.cs
//
// ILGPU context + accelerator acquisition. Prefers CUDA, then OpenCL, then the ILGPU
// CPU accelerator (which runs the very same kernels on host threads, so the whole
// pipeline works — slowly — without a supported GPU; texconv -nogpu maps to it).
// Kernels are compiled lazily per device and cached for the device's lifetime.
//-------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace TexGen.Gpu;

public enum GpuBackend
{
    /// <summary>Best available: CUDA, then OpenCL, then CPU.</summary>
    Auto,
    Cuda,
    OpenCL,
    /// <summary>ILGPU CPU accelerator (no GPU; texconv -nogpu).</summary>
    Cpu,
}

public sealed class GpuDevice : IDisposable
{
    public Context Context { get; }
    public Accelerator Accelerator { get; }

    private BcnEncoder? _bcn;
    private Bc7Encoder? _bc7;
    private Bc6hEncoder? _bc6h;
    private ImageKernels? _image;

    private GpuDevice(Context context, Accelerator accelerator)
    {
        Context = context;
        Accelerator = accelerator;
    }

    /// <summary>True when running on real GPU hardware (not the CPU accelerator).</summary>
    public bool IsHardware => Accelerator.AcceleratorType != AcceleratorType.CPU;

    public string Name => Accelerator.Name;

    public string Description => $"{Accelerator.AcceleratorType} {Accelerator.Name}";

    internal BcnEncoder Bcn => _bcn ??= new BcnEncoder(Accelerator);
    internal Bc7Encoder Bc7 => _bc7 ??= new Bc7Encoder(Accelerator);
    internal Bc6hEncoder Bc6h => _bc6h ??= new Bc6hEncoder(Accelerator);
    internal ImageKernels ImageOps => _image ??= new ImageKernels(Accelerator);

    /// <summary>
    /// Create a device. <paramref name="adapterIndex"/> selects among the hardware
    /// adapters of the chosen backend (texconv -gpu N); ignored for the CPU backend.
    /// </summary>
    public static GpuDevice Create(GpuBackend backend = GpuBackend.Auto, int adapterIndex = 0)
    {
        var context = Context.Create(b =>
        {
            b.Arrays(ArrayMode.InlineMutableStaticArrays)
             .Optimize(OptimizationLevel.O2)
             .EnableAlgorithms();
            if (backend is GpuBackend.Auto or GpuBackend.Cuda) b.Cuda();
            if (backend is GpuBackend.Auto or GpuBackend.OpenCL) b.OpenCL();
            // 4-thread warps x 16 warps = 64-thread groups (the BC6H/BC7 group size); the
            // default CPU device caps groups at 16 threads.
            b.CPU(new CPUDevice(numThreadsPerWarp: 4, numWarpsPerMultiprocessor: 16,
                numMultiprocessors: Math.Max(1, Environment.ProcessorCount / 4)));
        });

        try
        {
            Device? chosen = null;
            if (backend != GpuBackend.Cpu)
            {
                var order = backend switch
                {
                    GpuBackend.Cuda => new[] { AcceleratorType.Cuda },
                    GpuBackend.OpenCL => new[] { AcceleratorType.OpenCL },
                    _ => new[] { AcceleratorType.Cuda, AcceleratorType.OpenCL },
                };
                foreach (var type in order)
                {
                    var devices = context.Devices.Where(d => d.AcceleratorType == type).ToList();
                    if (devices.Count == 0) continue;
                    if (adapterIndex < 0 || adapterIndex >= devices.Count)
                        throw new ArgumentOutOfRangeException(nameof(adapterIndex),
                            $"GPU adapter {adapterIndex} not found ({devices.Count} {type} adapter(s) available)");
                    chosen = devices[adapterIndex];
                    break;
                }
                if (chosen is null && backend != GpuBackend.Auto)
                    throw new InvalidOperationException($"No {backend} GPU adapter is available");
            }
            chosen ??= context.Devices.First(d => d.AcceleratorType == AcceleratorType.CPU);
            return new GpuDevice(context, chosen.CreateAccelerator(context));
        }
        catch
        {
            context.Dispose();
            throw;
        }
    }

    /// <summary>Describe every adapter ILGPU can see (for diagnostics / -gpu listing).</summary>
    public static IReadOnlyList<string> ListAdapters()
    {
        using var context = Context.Create(b => b.Cuda().OpenCL().CPU());
        return context.Devices.Select(d => $"{d.AcceleratorType}: {d.Name}").ToList();
    }

    private static GpuDevice? _shared;
    private static readonly Lock SharedLock = new();

    /// <summary>A lazily created process-wide device (<see cref="GpuBackend.Auto"/>).</summary>
    public static GpuDevice Shared
    {
        get
        {
            lock (SharedLock) return _shared ??= Create();
        }
    }

    public void Dispose()
    {
        Accelerator.Dispose();
        Context.Dispose();
    }
}
