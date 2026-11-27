using System;
using NWaves.Operations;
using NWaves.Signals;
using OpenUtau.Core.Ustx;

namespace OpenUtauMobile.Services.NoteExtraction;

/// <summary>音高与歌词识别共用的音频读取工具。</summary>
internal static class ExtractionAudio
{
    /// <summary>
    /// 读取波形分片中 [startMs, endMs) 的片段，混为单声道并重采样到目标采样率。
    /// endMs 不大于 0 时读取到末尾。
    /// </summary>
    public static float[] ReadMono(UWavePart wavePart, double startMs, double endMs, int targetRate)
    {
        float[]? samples = wavePart.Samples;
        if (samples == null || wavePart.channels <= 0)
        {
            return [];
        }
        int channels = wavePart.channels;
        int total = samples.Length / channels;
        int start = Math.Clamp((int)(startMs * wavePart.sampleRate / 1000), 0, total);
        int end = endMs > 0 ? Math.Clamp((int)(endMs * wavePart.sampleRate / 1000), start, total) : total;
        if (end <= start)
        {
            return [];
        }
        float[] mono = new float[end - start];
        for (int i = 0; i < mono.Length; i++)
        {
            float sum = 0;
            int offset = (start + i) * channels;
            for (int channel = 0; channel < channels; channel++)
            {
                sum += samples[offset + channel];
            }
            mono[i] = sum / channels;
        }
        if (wavePart.sampleRate != targetRate)
        {
            mono = Operation.Resample(new DiscreteSignal(wavePart.sampleRate, mono), targetRate).Samples;
        }
        return mono;
    }
}
