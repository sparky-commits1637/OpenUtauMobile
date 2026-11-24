using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Core;
using Serilog;

namespace OpenUtauMobile.Services.NoteExtraction;

/// <summary>
/// Moonshine tiny（ONNX 四件套）语音识别，仅解码文本，用于给提取出的音符填歌词。
/// 依赖目录 moonshine-tiny 需包含 preprocess / encode / uncached_decode / cached_decode 的 .onnx 与 tokenizer.json。
/// </summary>
internal sealed partial class MoonshineTranscriber : IDisposable
{
    public const string PackageId = "moonshine-tiny";
    public const int SampleRate = 16000;
    /// <summary>Moonshine 单段音频上限为 64 秒，分段时应留出余量。</summary>
    public const double MaxSegmentSeconds = 30;
    /// <summary>低于该时长的片段不送入模型（模型最短 0.1 秒）。</summary>
    public const double MinSegmentSeconds = 0.2;

    private const int StartToken = 1;
    private const int EndToken = 2;
    // 与官方 moonshine_onnx 一致：每秒音频最多生成 6.5 个 token。
    private const double MaxTokensPerSecond = 6.5;
    private static readonly string[] ModelFiles =
        ["preprocess.onnx", "encode.onnx", "uncached_decode.onnx", "cached_decode.onnx", "tokenizer.json"];

    private readonly InferenceSession preprocess;
    private readonly InferenceSession encode;
    private readonly InferenceSession uncachedDecode;
    private readonly InferenceSession cachedDecode;
    private readonly string[] vocabulary;
    private RunOptions? runOptions;
    private bool disposed;

    public MoonshineTranscriber()
    {
        string directory = PackageManager.Inst.GetInstalledPath(PackageId) ?? throw new MissingPackageException(PackageId);
        if (!IsInstalled())
        {
            throw new MissingPackageException(PackageId);
        }
        preprocess = Onnx.getInferenceSession(Path.Combine(directory, ModelFiles[0]), OnnxRunnerChoice.CPU);
        encode = Onnx.getInferenceSession(Path.Combine(directory, ModelFiles[1]), OnnxRunnerChoice.CPU);
        uncachedDecode = Onnx.getInferenceSession(Path.Combine(directory, ModelFiles[2]), OnnxRunnerChoice.CPU);
        cachedDecode = Onnx.getInferenceSession(Path.Combine(directory, ModelFiles[3]), OnnxRunnerChoice.CPU);
        vocabulary = LoadVocabulary(Path.Combine(directory, ModelFiles[4]));
        runOptions = new RunOptions();
        Log.Information("Moonshine session ready. vocab={Vocab} cacheInputs={Cache}",
            vocabulary.Length, cachedDecode.InputNames.Count - 3);
    }

    /// <summary>依赖目录中四个模型与 tokenizer.json 是否齐全。</summary>
    public static bool IsInstalled()
    {
        string? directory = PackageManager.Inst.GetInstalledPath(PackageId);
        if (directory == null)
        {
            return false;
        }
        foreach (string file in ModelFiles)
        {
            if (!File.Exists(Path.Combine(directory, file)))
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>识别一段 16 kHz 单声道音频，返回文本；无内容时返回空串。</summary>
    public string Transcribe(float[] audio)
    {
        if (audio.Length < MinSegmentSeconds * SampleRate)
        {
            return string.Empty;
        }
        try
        {
            DenseTensor<float> waveform = new(audio, new[] { 1, audio.Length });
            DenseTensor<float> features;
            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = preprocess.Run(
                new[] { NamedOnnxValue.CreateFromTensor(preprocess.InputNames[0], waveform) }, preprocess.OutputNames, runOptions))
            {
                features = outputs[0].AsTensor<float>().ToDenseTensor();
            }
            int featureLength = features.Dimensions[features.Dimensions.Length - 2];
            DenseTensor<float> context;
            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = encode.Run(
                new[]
                {
                    NamedOnnxValue.CreateFromTensor(encode.InputNames[0], features),
                    NamedOnnxValue.CreateFromTensor(encode.InputNames[1], Scalar(featureLength)),
                }, encode.OutputNames, runOptions))
            {
                context = outputs[0].AsTensor<float>().ToDenseTensor();
            }
            return Decode(GenerateTokens(audio.Length, context));
        }
        catch (OnnxRuntimeException) when (runOptions is { Terminate: true })
        {
            throw new OperationCanceledException();
        }
    }

    private List<int> GenerateTokens(int sampleCount, DenseTensor<float> context)
    {
        int maxLength = Math.Max(1, (int)(sampleCount / (double)SampleRate * MaxTokensPerSecond));
        List<int> tokens = [];
        int sequenceLength = 1;
        List<DenseTensor<float>> cache = [];
        int next = RunDecoder(uncachedDecode, StartToken, context, sequenceLength, cache);
        for (int step = 0; step < maxLength; step++)
        {
            if (next == EndToken)
            {
                break;
            }
            tokens.Add(next);
            sequenceLength++;
            next = RunDecoder(cachedDecode, next, context, sequenceLength, cache);
        }
        return tokens;
    }

    /// <summary>
    /// 运行一步解码，返回贪心选出的下一个 token，并用输出替换 cache。
    /// 输入顺序：token、context、seq_len、cache...；输出顺序：logits、cache...。
    /// </summary>
    private int RunDecoder(InferenceSession session, int token, DenseTensor<float> context, int sequenceLength,
        List<DenseTensor<float>> cache)
    {
        List<NamedOnnxValue> inputs =
        [
            NamedOnnxValue.CreateFromTensor(session.InputNames[0], new DenseTensor<int>(new[] { token }, new[] { 1, 1 })),
            NamedOnnxValue.CreateFromTensor(session.InputNames[1], context),
            NamedOnnxValue.CreateFromTensor(session.InputNames[2], Scalar(sequenceLength)),
        ];
        for (int i = 0; i < cache.Count; i++)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(session.InputNames[3 + i], cache[i]));
        }
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.Run(inputs, session.OutputNames, runOptions);
        Tensor<float> logits = outputs[0].AsTensor<float>();
        int vocabularySize = logits.Dimensions[logits.Dimensions.Length - 1];
        float[] flat = logits.ToArray();
        int offset = flat.Length - vocabularySize;
        int best = 0;
        for (int i = 1; i < vocabularySize; i++)
        {
            if (flat[offset + i] > flat[offset + best])
            {
                best = i;
            }
        }
        cache.Clear();
        for (int i = 1; i < outputs.Count; i++)
        {
            cache.Add(outputs[i].AsTensor<float>().ToDenseTensor());
        }
        return best;
    }

    private static DenseTensor<int> Scalar(int value) => new(new[] { value }, new[] { 1 });

    [GeneratedRegex(@"^<0x([0-9A-Fa-f]{2})>$")]
    private static partial Regex ByteTokenRegex();

    /// <summary>sentencepiece 风格还原：▁ 表示空格，&lt;0xNN&gt; 为字节回退。</summary>
    private string Decode(List<int> tokens)
    {
        List<byte> bytes = [];
        foreach (int token in tokens)
        {
            if (token <= EndToken || token >= vocabulary.Length)
            {
                continue;
            }
            string piece = vocabulary[token];
            Match match = ByteTokenRegex().Match(piece);
            if (match.Success)
            {
                bytes.Add(Convert.ToByte(match.Groups[1].Value, 16));
            }
            else if (!(piece.StartsWith('<') && piece.EndsWith('>') && piece.Length > 2))
            {
                bytes.AddRange(Encoding.UTF8.GetBytes(piece.Replace('▁', ' ')));
            }
        }
        return Encoding.UTF8.GetString(bytes.ToArray()).Trim();
    }

    /// <summary>从 tokenizer.json 的 model.vocab 建立 id → 词片 表，兼容对象与数组两种写法。</summary>
    private static string[] LoadVocabulary(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement vocab = document.RootElement.GetProperty("model").GetProperty("vocab");
        if (vocab.ValueKind == JsonValueKind.Array)
        {
            List<string> pieces = [];
            foreach (JsonElement item in vocab.EnumerateArray())
            {
                pieces.Add((item.ValueKind == JsonValueKind.Array ? item[0] : item).GetString() ?? string.Empty);
            }
            return [.. pieces];
        }
        Dictionary<int, string> map = [];
        int maxId = -1;
        foreach (JsonProperty property in vocab.EnumerateObject())
        {
            int id = property.Value.GetInt32();
            map[id] = property.Name;
            maxId = Math.Max(maxId, id);
        }
        string[] result = new string[maxId + 1];
        Array.Fill(result, string.Empty);
        foreach (KeyValuePair<int, string> pair in map)
        {
            result[pair.Key] = pair.Value;
        }
        return result;
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
        preprocess.Dispose();
        encode.Dispose();
        uncachedDecode.Dispose();
        cachedDecode.Dispose();
        runOptions?.Dispose();
        runOptions = null;
    }
}
