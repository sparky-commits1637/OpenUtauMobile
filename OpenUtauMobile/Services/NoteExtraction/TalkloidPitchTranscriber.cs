using System;
using System.IO;
using System.Text.Json;
using Avalonia.Platform;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Core;
using OpenUtau.Core.Analysis;
using OpenUtau.Core.Ustx;
using Serilog;

namespace OpenUtauMobile.Services.NoteExtraction;

/// <summary>
/// TalkLoid 音高模型的前端与解码参数。
/// 默认值即训练笔记本中的设置（16 kHz、hop 160、torchaudio 默认 HTK 梅尔、功率谱、(ln(mel)+5)/3），
/// 输出为 softmax 分类：类 0 = 清音，类 c&gt;0 → 50 Hz × 2^((c-1)/48)。可用依赖目录中的 talkloid_pitch.json 覆盖。
/// </summary>
internal sealed class TalkloidPitchConfig
{
    public int SampleRate { get; init; } = 16000;
    public int FftSize { get; init; } = 1024;
    public int WindowSize { get; init; } = 1024;
    public int HopSize { get; init; } = 160;
    public int MelBands { get; init; } = 80;
    public double MelMinHz { get; init; } = 20;
    public double MelMaxHz { get; init; } = 8000;
    /// <summary>true 使用 HTK 梅尔刻度，false 使用 Slaney（librosa 默认）。</summary>
    public bool HtkMel { get; init; } = true;
    /// <summary>是否对三角滤波器做 Slaney 面积归一化（torchaudio 默认不做）。</summary>
    public bool SlaneyNorm { get; init; }
    /// <summary>对 STFT 幅度取的幂；2 为功率谱（torchaudio 默认）。</summary>
    public double Power { get; init; } = 2;
    /// <summary>送入模型前的归一化：(ln(mel) + InputOffset) / InputScale。</summary>
    public double InputOffset { get; init; } = 5;
    public double InputScale { get; init; } = 3;
    public double LogFloor { get; init; } = 1e-5;
    /// <summary>第 0 个分箱的中心（音分，相对 10 Hz）。类 c 对应 50 Hz×2^((c-1)/48)，故类 0 = 1200·log2(5) − 25。</summary>
    public double CentsOffset { get; init; } = 2761.3138711;
    public double CentsStep { get; init; } = 25;
    /// <summary>表示“无音高”的分箱；峰值落在此处按清音处理，也不参与加权平均。设为 -1 关闭。</summary>
    public int UnvoicedBin { get; init; }
    /// <summary>对输出做 softmax（训练用交叉熵）；设为 false 则按原值比较。</summary>
    public bool ApplySoftmax { get; init; } = true;
    /// <summary>峰值概率低于该值视为清音；0 即纯 argmax。</summary>
    public float VoicedThreshold { get; init; }
    /// <summary>在峰值分箱两侧各取多少个分箱做加权平均。</summary>
    public int AverageRadius { get; init; }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    public static TalkloidPitchConfig Load(string? path)
    {
        TalkloidPitchConfig config = path != null && File.Exists(path)
            ? JsonSerializer.Deserialize<TalkloidPitchConfig>(File.ReadAllText(path), JsonOptions) ?? new()
            : new();
        config.Validate();
        return config;
    }

    private void Validate()
    {
        bool powerOfTwo = FftSize >= 2 && (FftSize & (FftSize - 1)) == 0;
        if (SampleRate <= 0 || !powerOfTwo || WindowSize < 2 || WindowSize > FftSize || HopSize <= 0
            || MelBands <= 0 || MelMinHz < 0 || MelMaxHz <= MelMinHz || MelMaxHz > SampleRate / 2.0
            || LogFloor <= 0 || CentsStep <= 0 || AverageRadius < 0)
        {
            throw new InvalidDataException("Invalid talkloid_pitch.json settings.");
        }
    }
}

/// <summary>
/// 使用内置（或依赖目录覆盖的）talkloid_pitch.onnx 提取音高曲线。
/// 输入 mel [batch, frames, 80]，输出 f0_logits [batch, frames, bins]。
/// </summary>
internal sealed class TalkloidPitchTranscriber : IDisposable
{
    /// <summary>依赖目录名；其中的 talkloid_pitch.onnx / talkloid_pitch.json 优先于内置资源。</summary>
    public const string PackageId = "talkloid_pitch";
    private const string ModelFileName = "talkloid_pitch.onnx";
    private const string ConfigFileName = "talkloid_pitch.json";
    private static readonly Uri BundledModelUri = new("avares://OpenUtauMobile/Assets/Models/talkloid_pitch.onnx");

    private readonly TalkloidPitchConfig config;
    private readonly InferenceSession session;
    private readonly string inputName;
    private readonly float[][] melFilters;
    private readonly double[] window;
    private readonly double[] cosTable;
    private readonly double[] sinTable;
    private RunOptions? runOptions;
    private bool disposed;

    public TalkloidPitchTranscriber()
    {
        string? directory = PackageManager.Inst.GetInstalledPath(PackageId);
        config = TalkloidPitchConfig.Load(directory == null ? null : Path.Combine(directory, ConfigFileName));
        session = Onnx.getInferenceSession(LoadModelBytes(directory), OnnxRunnerChoice.CPU);
        inputName = session.InputNames[0];
        melFilters = BuildMelFilters(config);
        window = BuildWindow(config);
        int half = config.FftSize / 2;
        cosTable = new double[half];
        sinTable = new double[half];
        for (int k = 0; k < half; k++)
        {
            double angle = -2 * Math.PI * k / config.FftSize;
            cosTable[k] = Math.Cos(angle);
            sinTable[k] = Math.Sin(angle);
        }
        runOptions = new RunOptions();
        Log.Information("TalkLoid pitch session ready. input={Input} hop={Hop} sampleRate={Rate}",
            inputName, config.HopSize, config.SampleRate);
    }

    /// <summary>内置模型或依赖目录覆盖模型是否可用。</summary>
    public static bool IsAvailable()
    {
        string? directory = PackageManager.Inst.GetInstalledPath(PackageId);
        return directory != null && File.Exists(Path.Combine(directory, ModelFileName))
            || AssetLoader.Exists(BundledModelUri);
    }

    private static byte[] LoadModelBytes(string? directory)
    {
        if (directory != null && File.Exists(Path.Combine(directory, ModelFileName)))
        {
            return File.ReadAllBytes(Path.Combine(directory, ModelFileName));
        }
        using Stream stream = AssetLoader.Open(BundledModelUri);
        using MemoryStream buffer = new();
        stream.CopyTo(buffer);
        return buffer.ToArray();
    }

    /// <summary>推理 [startMs, endMs) 的音频，返回逐帧 MIDI 音高（空洞已线性插值）。</summary>
    public RmvpeResult? Infer(UWavePart wavePart, double startMs = 0, double endMs = 0)
    {
        float[] audio = ExtractionAudio.ReadMono(wavePart, startMs, endMs, config.SampleRate);
        // 反射填充要求信号长度大于 FftSize / 2。
        if (audio.Length <= config.FftSize / 2)
        {
            return null;
        }
        DenseTensor<float> mel = ComputeMel(audio, out int frames);
        try
        {
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.Run(
                new[] { NamedOnnxValue.CreateFromTensor(inputName, mel) }, session.OutputNames, runOptions);
            Tensor<float> logits = outputs[0].AsTensor<float>();
            int bins = logits.Dimensions[^1];
            float[] flat = logits.ToArray();
            if (flat.Length != frames * bins)
            {
                throw new InvalidDataException($"Unexpected TalkLoid output size: {flat.Length}, frames={frames}");
            }
            return new RmvpeResult
            {
                TimeStepSeconds = (double)config.HopSize / config.SampleRate,
                MidiPitch = Decode(flat, frames, bins),
            };
        }
        catch (OnnxRuntimeException) when (runOptions is { Terminate: true })
        {
            return null;
        }
    }

    private float[] Decode(float[] logits, int frames, int bins)
    {
        float[] midi = new float[frames];
        double[] probabilities = new double[bins];
        for (int frame = 0; frame < frames; frame++)
        {
            int best = 0;
            double maxLogit = double.NegativeInfinity;
            for (int bin = 0; bin < bins; bin++)
            {
                probabilities[bin] = logits[(frame * bins) + bin];
                maxLogit = Math.Max(maxLogit, probabilities[bin]);
            }
            if (config.ApplySoftmax)
            {
                double total = 0;
                for (int bin = 0; bin < bins; bin++)
                {
                    probabilities[bin] = Math.Exp(probabilities[bin] - maxLogit);
                    total += probabilities[bin];
                }
                for (int bin = 0; bin < bins; bin++)
                {
                    probabilities[bin] /= total;
                }
            }
            for (int bin = 1; bin < bins; bin++)
            {
                if (probabilities[bin] > probabilities[best])
                {
                    best = bin;
                }
            }
            if (best == config.UnvoicedBin || probabilities[best] < config.VoicedThreshold)
            {
                midi[frame] = float.NaN;
                continue;
            }
            int from = Math.Max(0, best - config.AverageRadius);
            int to = Math.Min(bins - 1, best + config.AverageRadius);
            double weightSum = 0;
            double centsSum = 0;
            for (int bin = from; bin <= to; bin++)
            {
                if (bin == config.UnvoicedBin)
                {
                    continue;
                }
                weightSum += probabilities[bin];
                centsSum += probabilities[bin] * (config.CentsOffset + (config.CentsStep * bin));
            }
            double f0 = 10.0 * Math.Pow(2.0, centsSum / weightSum / 1200.0);
            midi[frame] = (float)(69.0 + (12.0 * Math.Log2(f0 / 440.0)));
        }
        FillUnvoiced(midi);
        return midi;
    }

    /// <summary>与 RMVPE 路径一致：在清音空洞内线性插值，首尾沿用最近的浊音。</summary>
    private static void FillUnvoiced(float[] midi)
    {
        int previous = -1;
        for (int i = 0; i < midi.Length; i++)
        {
            if (float.IsNaN(midi[i]))
            {
                continue;
            }
            for (int gap = previous + 1; gap < i; gap++)
            {
                midi[gap] = previous < 0
                    ? midi[i]
                    : midi[previous] + ((midi[i] - midi[previous]) * (gap - previous) / (i - previous));
            }
            previous = i;
        }
        if (previous >= 0)
        {
            for (int i = previous + 1; i < midi.Length; i++)
            {
                midi[i] = midi[previous];
            }
        }
    }

    /// <summary>center=True 的 STFT 功率谱 → 梅尔滤波 → ln(max(x, floor)) → (x+5)/3。</summary>
    private DenseTensor<float> ComputeMel(float[] audio, out int frames)
    {
        int fft = config.FftSize;
        int pad = fft / 2;
        frames = 1 + (audio.Length / config.HopSize);
        int bins = (fft / 2) + 1;
        int windowOffset = (fft - config.WindowSize) / 2;
        DenseTensor<float> mel = new(new[] { 1, frames, config.MelBands });
        double[] re = new double[fft];
        double[] im = new double[fft];
        double[] magnitude = new double[bins];
        for (int frame = 0; frame < frames; frame++)
        {
            int origin = (frame * config.HopSize) - pad;
            Array.Clear(re);
            Array.Clear(im);
            for (int n = 0; n < config.WindowSize; n++)
            {
                re[windowOffset + n] = audio[Reflect(origin + windowOffset + n, audio.Length)] * window[n];
            }
            Fft(re, im);
            for (int k = 0; k < bins; k++)
            {
                double square = (re[k] * re[k]) + (im[k] * im[k]);
                magnitude[k] = config.Power == 2 ? square : Math.Pow(square, config.Power / 2);
            }
            for (int band = 0; band < config.MelBands; band++)
            {
                double sum = 0;
                float[] filter = melFilters[band];
                for (int k = 0; k < bins; k++)
                {
                    sum += filter[k] * magnitude[k];
                }
                mel[0, frame, band] = (float)((Math.Log(Math.Max(sum, config.LogFloor)) + config.InputOffset) / config.InputScale);
            }
        }
        return mel;
    }

    private static int Reflect(int index, int length)
    {
        if (index < 0)
        {
            return -index;
        }
        return index >= length ? (2 * (length - 1)) - index : index;
    }

    /// <summary>原地基 2 FFT。</summary>
    private void Fft(double[] re, double[] im)
    {
        int n = re.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }
        for (int length = 2; length <= n; length <<= 1)
        {
            int half = length / 2;
            int step = n / length;
            for (int start = 0; start < n; start += length)
            {
                for (int k = 0; k < half; k++)
                {
                    double wr = cosTable[k * step];
                    double wi = sinTable[k * step];
                    int a = start + k;
                    int b = a + half;
                    double tr = (re[b] * wr) - (im[b] * wi);
                    double ti = (re[b] * wi) + (im[b] * wr);
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }

    private static double[] BuildWindow(TalkloidPitchConfig config)
    {
        // 周期 Hann 窗，与 torch.hann_window 默认一致。
        double[] window = new double[config.WindowSize];
        for (int n = 0; n < window.Length; n++)
        {
            window[n] = 0.5 - (0.5 * Math.Cos(2 * Math.PI * n / config.WindowSize));
        }
        return window;
    }

    private static double HzToMel(double hz, bool htk)
    {
        if (htk)
        {
            return 2595.0 * Math.Log10(1.0 + (hz / 700.0));
        }
        const double LinearStep = 200.0 / 3;
        const double LogStartHz = 1000.0;
        double logStartMel = LogStartHz / LinearStep;
        double logStep = Math.Log(6.4) / 27.0;
        return hz >= LogStartHz ? logStartMel + (Math.Log(hz / LogStartHz) / logStep) : hz / LinearStep;
    }

    private static double MelToHz(double mel, bool htk)
    {
        if (htk)
        {
            return 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);
        }
        const double LinearStep = 200.0 / 3;
        const double LogStartHz = 1000.0;
        double logStartMel = LogStartHz / LinearStep;
        double logStep = Math.Log(6.4) / 27.0;
        return mel >= logStartMel ? LogStartHz * Math.Exp(logStep * (mel - logStartMel)) : mel * LinearStep;
    }

    private static float[][] BuildMelFilters(TalkloidPitchConfig config)
    {
        int bins = (config.FftSize / 2) + 1;
        double minMel = HzToMel(config.MelMinHz, config.HtkMel);
        double maxMel = HzToMel(config.MelMaxHz, config.HtkMel);
        double[] edges = new double[config.MelBands + 2];
        for (int i = 0; i < edges.Length; i++)
        {
            edges[i] = MelToHz(minMel + ((maxMel - minMel) * i / (edges.Length - 1)), config.HtkMel);
        }
        float[][] filters = new float[config.MelBands][];
        for (int band = 0; band < config.MelBands; band++)
        {
            filters[band] = new float[bins];
            double lowerWidth = edges[band + 1] - edges[band];
            double upperWidth = edges[band + 2] - edges[band + 1];
            double norm = config.SlaneyNorm ? 2.0 / (edges[band + 2] - edges[band]) : 1.0;
            for (int k = 0; k < bins; k++)
            {
                double frequency = (double)k * config.SampleRate / config.FftSize;
                double rising = (frequency - edges[band]) / lowerWidth;
                double falling = (edges[band + 2] - frequency) / upperWidth;
                filters[band][k] = (float)(Math.Max(0.0, Math.Min(rising, falling)) * norm);
            }
        }
        return filters;
    }

    /// <summary>请求终止正在进行的推理。</summary>
    public void Interrupt()
    {
        if (!disposed && runOptions != null)
        {
            runOptions.Terminate = true;
        }
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }
        disposed = true;
        session.Dispose();
        runOptions?.Dispose();
        runOptions = null;
    }
}
