using System.Collections.Generic;
using System.Text.RegularExpressions;
using OpenUtau.Core.Ustx;

namespace OpenUtauMobile.Services.NoteExtraction;

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

    /// <summary>启发式英文音节切分；撇号被忽略，无元音的词整体作为一个音节。</summary>
    public static List<string> SplitSyllables(string text)
    {
        List<string> syllables = [];
        foreach (Match word in WordRegex().Matches(text.Replace("'", string.Empty).ToLowerInvariant()))
        {
            MatchCollection parts = SyllableRegex().Matches(word.Value);
            if (parts.Count == 0)
            {
                syllables.Add(word.Value);
                continue;
            }
            foreach (Match part in parts)
            {
                syllables.Add(part.Value);
            }
        }
        return syllables;
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
