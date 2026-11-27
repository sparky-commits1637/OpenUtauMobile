using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using OpenUtau.Core.Ustx;

namespace OpenUtauMobile.Services.NoteExtraction;

/// <summary>带演唱时间的音节，时间相对于所属音频片段起点（秒）；无法对齐时时间为 NaN。</summary>
internal readonly record struct TimedSyllable(string Text, double StartSeconds, double EndSeconds)
{
    /// <summary>是否有有效的对齐时间。</summary>
    public bool HasTime => double.IsFinite(StartSeconds) && double.IsFinite(EndSeconds);
}

/// <summary>把识别出的文本拆成音节并按顺序写入音符歌词。</summary>
internal static partial class LyricAssigner
{
    /// <summary>音符数多于音节数时，多出的音符使用的延音歌词。</summary>
    public const string Continuation = "+";

    [GeneratedRegex(@"[\p{L}]+")]
    private static partial Regex WordRegex();

    // 元音组 + 可选尾辅音：词尾保留全部辅音，词中辅音丛只带走第一个。
    [GeneratedRegex(@"[^aeiouy]*[aeiouy]+(?:[^aeiouy]+$|[^aeiouy](?=[^aeiouy]))?")]
    private static partial Regex SyllableRegex();

    /// <summary>同一单词内音节与音符起点重合到此范围内（秒）仍视为同一时刻。</summary>
    private const double NearestToleranceSeconds = 0.15;

    /// <summary>启发式英文音节切分，按单词分组；撇号被忽略，无元音的词整体作为一个音节。</summary>
    public static List<List<string>> SplitWords(string text)
    {
        List<List<string>> words = [];
        foreach (Match word in WordRegex().Matches(text.Replace("'", string.Empty).ToLowerInvariant()))
        {
            List<string> syllables = [];
            MatchCollection parts = SyllableRegex().Matches(word.Value);
            if (parts.Count == 0)
            {
                syllables.Add(word.Value);
            }
            foreach (Match part in parts)
            {
                syllables.Add(part.Value);
            }
            words.Add(syllables);
        }
        return words;
    }

    /// <summary>启发式英文音节切分（不分单词）。</summary>
    public static List<string> SplitSyllables(string text) => [.. SplitWords(text).SelectMany(word => word)];

    /// <summary>
    /// 按时间为音符写歌词：每个音节落在与其时间重叠最多的音符上，顺序只增不减，
    /// 因此 GAME 合并或拆分音符只影响该处，不会让后面的歌词整体错位。
    /// 没有分到音节的音符写延音；两个音节落在同一个音符上时保留重叠更大的一个，不拆分音符。
    /// </summary>
    /// <param name="notes">按时间排序的音符。</param>
    /// <param name="noteStartSeconds">各音符起点（秒），与音节时间使用同一原点。</param>
    /// <param name="noteEndSeconds">各音符终点（秒）。</param>
    /// <param name="syllables">带时间的音节；没有时间的音节被跳过。</param>
    public static void AssignTimed(IReadOnlyList<UNote> notes, IReadOnlyList<double> noteStartSeconds,
        IReadOnlyList<double> noteEndSeconds, IReadOnlyList<TimedSyllable> syllables)
    {
        int count = notes.Count;
        string?[] lyrics = new string?[count];
        double[] strength = new double[count];
        int floor = 0;
        foreach (TimedSyllable syllable in syllables)
        {
            if (!syllable.HasTime)
            {
                continue;
            }
            int target = -1;
            double bestOverlap = 0;
            for (int i = floor; i < count; i++)
            {
                if (noteStartSeconds[i] >= syllable.EndSeconds)
                {
                    break;
                }
                double overlap = Math.Min(noteEndSeconds[i], syllable.EndSeconds) - Math.Max(noteStartSeconds[i], syllable.StartSeconds);
                if (overlap > bestOverlap)
                {
                    bestOverlap = overlap;
                    target = i;
                }
            }
            if (target < 0)
            {
                // 没有任何重叠时，取起点最接近的音符，但必须在容差之内。
                double bestDistance = NearestToleranceSeconds;
                for (int i = floor; i < count; i++)
                {
                    double distance = Math.Abs(noteStartSeconds[i] - syllable.StartSeconds);
                    if (distance <= bestDistance)
                    {
                        bestDistance = distance;
                        target = i;
                    }
                }
            }
            if (target < 0)
            {
                continue;
            }
            floor = target;
            if (lyrics[target] == null || bestOverlap > strength[target])
            {
                lyrics[target] = syllable.Text;
                strength[target] = bestOverlap;
            }
        }
        for (int i = 0; i < count; i++)
        {
            notes[i].lyric = lyrics[i] ?? Continuation;
        }
    }

    /// <summary>按时间顺序为音符写歌词；音节不足的音符写延音，多余音节丢弃。</summary>
    public static void Assign(IReadOnlyList<UNote> notes, IReadOnlyList<string> syllables)
    {
        if (syllables.Count == 0)
        {
            return;
        }
        for (int i = 0; i < notes.Count; i++)
        {
            notes[i].lyric = i < syllables.Count ? syllables[i] : Continuation;
        }
    }
}
