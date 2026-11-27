using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Core;
using Serilog;

namespace OpenUtauMobile.Services.NoteExtraction;

/// <summary>
/// 基于 wav2vec2-base-960h（ONNX）的 CTC 强制对齐：已知文本，求每个字母出现的时间，再汇总为音节时间。
/// 依赖目录 wav2vec2 需包含 wav2vec2.onnx 与 vocab.json（wav2vec2_meta.json 可选）。
/// </summary>
internal sealed class Wav2Vec2Aligner : IDisposable
{
    public const string PackageId = "wav2vec2";
    public const int SampleRate = 16000;

    private const string ModelFileName = "wav2vec2.onnx";
    private const string VocabFileName = "vocab.json";
    private const string MetaFileName = "wav2vec2_meta.json";

    /// <summary>wav2vec2 每帧 20 毫秒（50 帧/秒）。</summary>
    private const double FrameSeconds = 0.02;
    /// <summary>低于该时长的音频不做对齐（模型卷积层需要足够样本）。</summary>
    private const double MinAudioSeconds = 0.3;
    /// <summary>同一单词内，音节之间 CTC 空白帧全部算作前一个音节的延续。</summary>
    private const double WordGapExtendSeconds = 1.0;
    /// <summary>跨单词时，前一个音节最多向后延伸的秒数。</summary>
    private const double CrossWordExtendSeconds = 0.25;
    /// <summary>最后一个音节的尾部补偿。</summary>
    private const double TailExtendSeconds = 0.1;

    private readonly InferenceSession session;
    private readonly Dictionary<string, int> vocabulary;
    private readonly int blankId;
    private readonly int wordGapId;
    private RunOptions? runOptions;
    private bool disposed;

    public Wav2Vec2Aligner()
    {
        string directory = PackageManager.Inst.GetInstalledPath(PackageId) ?? throw new MissingPackageException(PackageId);
        if (!IsInstalled())
        {
            throw new MissingPackageException(PackageId);
        }
        vocabulary = JsonSerializer.Deserialize<Dictionary<string, int>>(
            File.ReadAllText(Path.Combine(directory, VocabFileName))) ?? throw new InvalidDataException("Empty wav2vec2 vocab.");
        blankId = vocabulary.TryGetValue("<pad>", out int pad) ? pad : 0;
        wordGapId = vocabulary.TryGetValue("|", out int gap) ? gap : -1;
        string metaPath = Path.Combine(directory, MetaFileName);
        if (File.Exists(metaPath))
        {
            // 元数据中显式给出的 id 优先于词表推断。
            using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(metaPath));
            if (meta.RootElement.TryGetProperty("blank_id", out JsonElement blank) && blank.TryGetInt32(out int blankValue))
            {
                blankId = blankValue;
            }
            if (meta.RootElement.TryGetProperty("word_gap_id", out JsonElement wordGap) && wordGap.TryGetInt32(out int wordGapValue))
            {
                wordGapId = wordGapValue;
            }
        }
        session = Onnx.getInferenceSession(Path.Combine(directory, ModelFileName), OnnxRunnerChoice.CPU);
        runOptions = new RunOptions();
        Log.Information("wav2vec2 aligner ready. vocab={Vocab} blank={Blank} gap={Gap}", vocabulary.Count, blankId, wordGapId);
    }

    /// <summary>依赖目录中模型与词表是否齐全。</summary>
    public static bool IsInstalled()
    {
        string? directory = PackageManager.Inst.GetInstalledPath(PackageId);
        return directory != null
            && File.Exists(Path.Combine(directory, ModelFileName))
            && File.Exists(Path.Combine(directory, VocabFileName));
    }

    /// <summary>
    /// 把 <paramref name="words"/>（每个单词是它的音节列表）与音频强制对齐，
    /// 返回与音节展开顺序一致的时间；无法对齐返回 null，词表里没有的字母组成的音节时间为 NaN。
    /// 时间相对于 <paramref name="audio"/> 的起点，单位秒。
    /// </summary>
    public TimedSyllable[]? Align(float[] audio, IReadOnlyList<List<string>> words)
    {
        if (audio.Length < MinAudioSeconds * SampleRate)
        {
            return null;
        }
        // 展开音节，并把每个字母映射为词表 id；单词之间插入词分隔符。
        List<string> syllableTexts = [];
        List<int> syllableWord = [];
        List<int> tokens = [];
        List<int> tokenSyllable = [];
        bool previousWordHadTokens = false;
        for (int w = 0; w < words.Count; w++)
        {
            bool wordHasTokens = false;
            foreach (string syllable in words[w])
            {
                int syllableIndex = syllableTexts.Count;
                syllableTexts.Add(syllable);
                syllableWord.Add(w);
                foreach (char c in syllable)
                {
                    if (!vocabulary.TryGetValue(char.ToUpperInvariant(c).ToString(), out int id) || id == blankId || id == wordGapId)
                    {
                        continue;
                    }
                    if (!wordHasTokens && previousWordHadTokens && wordGapId >= 0)
                    {
                        tokens.Add(wordGapId);
                        tokenSyllable.Add(-1);
                    }
                    wordHasTokens = true;
                    tokens.Add(id);
                    tokenSyllable.Add(syllableIndex);
                }
            }
            if (wordHasTokens)
            {
                previousWordHadTokens = true;
            }
        }
        if (tokens.Count == 0)
        {
            return null;
        }

        float[] logProbabilities = RunModel(audio, out int frames, out int vocabularySize);
        if (frames < tokens.Count)
        {
            return null;
        }
        int[]? path = Viterbi(logProbabilities, frames, vocabularySize, [.. tokens], blankId);
        if (path == null)
        {
            return null;
        }

        // 每个 token 占用的帧区间（奇数状态 2i+1 对应第 i 个 token）。
        int[] firstFrame = new int[tokens.Count];
        int[] lastFrame = new int[tokens.Count];
        Array.Fill(firstFrame, -1);
        for (int t = 0; t < frames; t++)
        {
            int state = path[t];
            if (state % 2 == 0)
            {
                continue;
            }
            int token = state / 2;
            if (firstFrame[token] < 0)
            {
                firstFrame[token] = t;
            }
            lastFrame[token] = t;
        }

        // 音节时间 = 其字母的最早起点到最晚终点。
        int count = syllableTexts.Count;
        double[] start = new double[count];
        double[] end = new double[count];
        Array.Fill(start, double.NaN);
        Array.Fill(end, double.NaN);
        for (int i = 0; i < tokens.Count; i++)
        {
            int syllable = tokenSyllable[i];
            if (syllable < 0 || firstFrame[i] < 0)
            {
                continue;
            }
            double tokenStart = firstFrame[i] * FrameSeconds;
            double tokenEnd = (lastFrame[i] + 1) * FrameSeconds;
            if (double.IsNaN(start[syllable]) || tokenStart < start[syllable])
            {
                start[syllable] = tokenStart;
            }
            if (double.IsNaN(end[syllable]) || tokenEnd > end[syllable])
            {
                end[syllable] = tokenEnd;
            }
        }

        // CTC 的峰很尖，字母之间大多是空白帧，所以把空白帧归给前一个音节，得到“演唱时长”。
        TimedSyllable[] result = new TimedSyllable[count];
        for (int i = 0; i < count; i++)
        {
            if (double.IsNaN(start[i]))
            {
                result[i] = new TimedSyllable(syllableTexts[i], double.NaN, double.NaN);
                continue;
            }
            int next = i + 1;
            while (next < count && double.IsNaN(start[next]))
            {
                next++;
            }
            double extended;
            if (next >= count)
            {
                extended = end[i] + TailExtendSeconds;
            }
            else
            {
                double limit = syllableWord[next] == syllableWord[i] ? WordGapExtendSeconds : CrossWordExtendSeconds;
                extended = Math.Max(end[i], Math.Min(start[next], end[i] + limit));
            }
            result[i] = new TimedSyllable(syllableTexts[i], start[i], extended);
        }
        return result;
    }

    /// <summary>运行模型并对 logits 做 log_softmax，返回 [帧, 词表] 展平的数组。</summary>
    private float[] RunModel(float[] audio, out int frames, out int vocabularySize)
    {
        // 与导出时一致：单段音频按 (x - mean) / sqrt(var + 1e-7) 归一化。
        double mean = 0;
        foreach (float sample in audio)
        {
            mean += sample;
        }
        mean /= audio.Length;
        double variance = 0;
        foreach (float sample in audio)
        {
            double delta = sample - mean;
            variance += delta * delta;
        }
        variance /= audio.Length;
        double scale = 1.0 / Math.Sqrt(variance + 1e-7);
        float[] normalized = new float[audio.Length];
        for (int i = 0; i < audio.Length; i++)
        {
            normalized[i] = (float)((audio[i] - mean) * scale);
        }

        try
        {
            DenseTensor<float> input = new(normalized, new[] { 1, normalized.Length });
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.Run(
                new[] { NamedOnnxValue.CreateFromTensor(session.InputNames[0], input) }, session.OutputNames, runOptions);
            DenseTensor<float> logits = outputs[0].AsTensor<float>().ToDenseTensor();
            frames = logits.Dimensions[1];
            vocabularySize = logits.Dimensions[2];
            float[] raw = logits.Buffer.ToArray();
            float[] result = new float[frames * vocabularySize];
            for (int t = 0; t < frames; t++)
            {
                int offset = t * vocabularySize;
                float max = float.NegativeInfinity;
                for (int v = 0; v < vocabularySize; v++)
                {
                    max = Math.Max(max, raw[offset + v]);
                }
                double sum = 0;
                for (int v = 0; v < vocabularySize; v++)
                {
                    sum += Math.Exp(raw[offset + v] - max);
                }
                float logSum = max + (float)Math.Log(sum);
                for (int v = 0; v < vocabularySize; v++)
                {
                    result[offset + v] = raw[offset + v] - logSum;
                }
            }
            return result;
        }
        catch (OnnxRuntimeException) when (runOptions is { Terminate: true })
        {
            throw new OperationCanceledException();
        }
    }

    /// <summary>
    /// CTC 维特比：状态序列为 空白、token0、空白、token1……、空白（共 2L+1 个）。
    /// 返回每一帧所处的状态；文本无法放进这么多帧时返回 null。
    /// </summary>
    private static int[]? Viterbi(float[] logProbabilities, int frames, int vocabularySize, int[] tokens, int blank)
    {
        int length = tokens.Length;
        int states = 2 * length + 1;
        float[] previous = new float[states];
        float[] current = new float[states];
        byte[] back = new byte[(long)frames * states];
        Array.Fill(previous, float.NegativeInfinity);
        previous[0] = logProbabilities[blank];
        previous[1] = logProbabilities[tokens[0]];
        for (int t = 1; t < frames; t++)
        {
            int offset = t * vocabularySize;
            for (int s = 0; s < states; s++)
            {
                bool isToken = s % 2 == 1;
                int id = isToken ? tokens[s / 2] : blank;
                float best = previous[s];
                byte pointer = 0;
                if (s >= 1 && previous[s - 1] > best)
                {
                    best = previous[s - 1];
                    pointer = 1;
                }
                // 跳过中间的空白，仅当相邻两个 token 不同。
                if (isToken && s >= 3 && tokens[s / 2] != tokens[s / 2 - 1] && previous[s - 2] > best)
                {
                    best = previous[s - 2];
                    pointer = 2;
                }
                current[s] = float.IsNegativeInfinity(best) ? float.NegativeInfinity : best + logProbabilities[offset + id];
                back[(long)t * states + s] = pointer;
            }
            (previous, current) = (current, previous);
        }
        int final = states - 1;
        if (states >= 2 && previous[states - 2] > previous[states - 1])
        {
            final = states - 2;
        }
        if (float.IsNegativeInfinity(previous[final]))
        {
            return null;
        }
        int[] path = new int[frames];
        int state = final;
        for (int t = frames - 1; t >= 0; t--)
        {
            path[t] = state;
            state -= back[(long)t * states + state];
        }
        return path;
    }

    /// <summary>中断正在运行的推理。</summary>
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
