using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenUtau.Classic;
using OpenUtau.Core.Ustx;
using OpenUtauMobile.Helpers;
using Serilog;

namespace OpenUtauMobile.Services;

/// <summary>声库检查报告。</summary>
/// <param name="Title">弹窗标题。</param>
/// <param name="Text">给普通用户阅读的报告正文。</param>
public sealed record VoicebankCheckReport(string Title, string Text);

/// <summary>
/// 检查 UTAU 声库的健康状况：复用 Core 的声库检查器找出损坏的设置，
/// 再补充没被用到的音频、可删除的缓存文件和指向不存在文件的路径，并整理成普通人能读懂的文字。
/// </summary>
public static class VoicebankCheckService
{
    private const int MaxExamples = 3;

    private static readonly string[] CacheExtensions = [".frq", ".llsm", ".mrq", ".pmk", ".uspec", ".dmp"];
    private static readonly string[] JunkFileNames = ["thumbs.db", "desktop.ini", ".ds_store"];

    /// <summary>错误键到问题分组的映射；同一分组的多条记录会合并成一句话。</summary>
    private static readonly Dictionary<string, string> GroupByKey = new()
    {
        ["singererror.txtnotfound"] = "noCharacterTxt",
        ["singererror.failedload"] = "loadFailed",
        ["singererror.yamlnotfound"] = "noYaml",
        ["singererror.invalidoto"] = "brokenLine",
        ["singererror.soundmissing"] = "missingFile",
        ["singererror.soundnotopened"] = "cannotOpen",
        ["singererror.invalidduration"] = "cannotOpen",
        ["singererror.samplerate"] = "soundFormat",
        ["singererror.bitdepth"] = "soundFormat",
        ["singererror.mono"] = "stereo",
        ["singererror.duplicatealias"] = "duplicate",
        ["singererror.aliasstartwithspace"] = "oddNames",
        ["singererror.aliasmultiplespaces"] = "oddNames",
        ["singererror.aliasfullwidthspaces"] = "oddNames",
        ["singererror.aliasisnfd"] = "oddNames",
        ["singererror.wavisnfd"] = "fileNameForm",
        ["singererror.wrongcase"] = "caseMismatch",
        ["singererror.duplicatename"] = "caseDuplicate",
        ["singererror.offsetshort"] = "timing",
        ["singererror.offsetoutofduration"] = "timing",
        ["singererror.preuttershort"] = "timing",
        ["singererror.preutteroutofduration"] = "timing",
        ["singererror.consonantshort"] = "timing",
        ["singererror.consonantoutofduration"] = "timing",
        ["singererror.overlapoutofduration"] = "timing",
        ["singererror.cutoffpreutter"] = "timing",
        ["singererror.cutoffoverlap"] = "timing",
        ["singererror.cutoffconsonant"] = "timing",
        ["singererror.cutoffoutofduration"] = "timing",
    };

    /// <summary>在后台线程检查声库并生成报告。</summary>
    public static Task<VoicebankCheckReport> CheckAsync(USinger singer) => Task.Run(() => Check(singer));

    private static VoicebankCheckReport Check(USinger singer)
    {
        string title = string.Format(LocalizedText.S("VoicebankCheck.Title", "Voicebank check: {0}"), singer.LocalizedName);
        string root = singer.Location;
        if (singer.SingerType != USingerType.Classic || string.IsNullOrEmpty(root) || !Directory.Exists(root))
        {
            return new(title, LocalizedText.S("VoicebankCheck.Unsupported",
                "This check only works for UTAU-style voicebanks stored in a folder."));
        }

        List<string> problems = [];
        List<string> tips = [];
        try
        {
            VoicebankErrorChecker checker = new(root, singer.BasePath);
            checker.Check();
            AddGroups(checker.Errors, problems, root);
            AddGroups(checker.Infos, tips, root);
            AddCharacterPathProblems(singer, root, problems);
            AddFileFindings(singer, root, tips);
        }
        catch (Exception exception)
        {
            Log.Error(exception, "Voicebank check failed.");
            return new(title, string.Format(LocalizedText.S("VoicebankCheck.Failed",
                "The check could not be completed: {0}"), exception.Message));
        }

        StringBuilder text = new();
        if (problems.Count == 0 && tips.Count == 0)
        {
            text.Append(LocalizedText.S("VoicebankCheck.AllGood", "No problems found. This voicebank looks healthy."));
        }
        else
        {
            text.AppendLine(string.Format(LocalizedText.S("VoicebankCheck.Summary",
                "Found {0} problem(s) and {1} tip(s)."), problems.Count, tips.Count));
            AppendSection(text, LocalizedText.S("VoicebankCheck.ProblemsHeader", "Problems to fix"), problems);
            AppendSection(text, LocalizedText.S("VoicebankCheck.TipsHeader", "Tips (optional)"), tips);
        }
        return new(title, text.ToString().TrimEnd());
    }

    private static void AppendSection(StringBuilder text, string header, List<string> lines)
    {
        if (lines.Count == 0) return;
        text.AppendLine().AppendLine(header);
        foreach (string line in lines)
        {
            text.Append("• ").AppendLine(line);
        }
    }

    /// <summary>把检查器的原始记录按分组合并，并附上最多几个例子。</summary>
    private static void AddGroups(List<VoicebankError> errors, List<string> lines, string root)
    {
        foreach (IGrouping<string, VoicebankError> group in errors.GroupBy(error =>
                     GroupByKey.TryGetValue(error.messageKey, out string? id) ? id : "other"))
        {
            int count = group.Count();
            List<string> examples = group
                .Select(error => DescribeExample(error, root))
                .Where(example => !string.IsNullOrEmpty(example))
                .Distinct()
                .Take(MaxExamples)
                .ToList();
            if (group.Key == "duplicate")
            {
                // 重复别名由检查器合并成一条多行文本，这里拆开取前几行作为例子。
                string[] duplicateLines = group.SelectMany(error => error.strings)
                    .SelectMany(item => item.Split('\n'))
                    .Select(item => item.Trim())
                    .Where(item => item.Length > 0)
                    .ToArray();
                count = duplicateLines.Length;
                examples = duplicateLines.Take(MaxExamples).ToList();
            }
            string line = string.Format(GroupText(group.Key), count);
            if (examples.Count > 0)
            {
                line += " " + string.Format(LocalizedText.S("VoicebankCheck.Examples", "For example: {0}"),
                    string.Join("; ", examples));
            }
            lines.Add(line);
        }
    }

    private static string DescribeExample(VoicebankError error, string root)
    {
        if (!string.IsNullOrEmpty(error.soundFile))
        {
            return Path.GetRelativePath(root, error.soundFile);
        }
        if (error.trace != null && !string.IsNullOrEmpty(error.trace.file))
        {
            return $"{Path.GetFileName(error.trace.file)} #{error.trace.lineNumber}";
        }
        return error.e?.Message ?? string.Empty;
    }

    private static string GroupText(string group) => group switch
    {
        "noCharacterTxt" => LocalizedText.S("VoicebankCheck.NoCharacterTxt",
            "character.txt is missing, so this folder is not recognized as a voicebank."),
        "loadFailed" => LocalizedText.S("VoicebankCheck.LoadFailed",
            "The voicebank could not be opened ({0} error)."),
        "noYaml" => LocalizedText.S("VoicebankCheck.NoYaml",
            "There is no character.yaml. It is optional and only adds extra info such as a default phonemizer."),
        "brokenLine" => LocalizedText.S("VoicebankCheck.BrokenLine",
            "{0} line(s) in oto.ini are broken: a number is missing or cannot be read."),
        "missingFile" => LocalizedText.S("VoicebankCheck.MissingFile",
            "{0} setting(s) point to sound files that do not exist, so those sounds will fail."),
        "cannotOpen" => LocalizedText.S("VoicebankCheck.CannotOpen",
            "{0} sound file(s) are damaged or empty and cannot be opened."),
        "soundFormat" => LocalizedText.S("VoicebankCheck.SoundFormat",
            "{0} sound file(s) are not 44.1 kHz and 16-bit. They may sound wrong or fail."),
        "stereo" => LocalizedText.S("VoicebankCheck.Stereo",
            "{0} sound file(s) are not mono. Mono sounds work better and take less space."),
        "duplicate" => LocalizedText.S("VoicebankCheck.Duplicate",
            "{0} sound name(s) are listed more than once. Only one of each is used."),
        "oddNames" => LocalizedText.S("VoicebankCheck.OddNames",
            "{0} sound name(s) contain odd spaces or special characters that can stop them matching lyrics."),
        "fileNameForm" => LocalizedText.S("VoicebankCheck.FileNameForm",
            "{0} sound file name(s) use a special character form (common on macOS) and may not be found on this device."),
        "caseMismatch" => LocalizedText.S("VoicebankCheck.CaseMismatch",
            "{0} file name(s) use different upper/lower case than oto.ini says. This works on Windows but fails on Android and Linux."),
        "caseDuplicate" => LocalizedText.S("VoicebankCheck.CaseDuplicate",
            "{0} file(s) have names that differ only by upper/lower case, which breaks on some systems."),
        "timing" => LocalizedText.S("VoicebankCheck.Timing",
            "{0} sound(s) have timing numbers (offset, preutterance, overlap, consonant or cutoff) that do not fit inside the sound file. Fix them in an oto editor."),
        _ => LocalizedText.S("VoicebankCheck.Other", "{0} other issue(s) were found."),
    };

    /// <summary>检查 character.txt / character.yaml 里的图片、示例音频路径是否指向不存在的文件。</summary>
    private static void AddCharacterPathProblems(USinger singer, string root, List<string> problems)
    {
        foreach (string fileName in new[] { VoicebankLoader.kCharTxt, VoicebankLoader.kCharYaml })
        {
            string path = Path.Combine(root, fileName);
            if (!File.Exists(path)) continue;
            foreach (string line in File.ReadLines(path, singer.TextFileEncoding))
            {
                if (!TryReadPathEntry(line, fileName.EndsWith(".yaml", StringComparison.OrdinalIgnoreCase), out string key, out string value))
                {
                    continue;
                }
                if (!File.Exists(Path.Combine(root, value)))
                {
                    problems.Add(string.Format(LocalizedText.S("VoicebankCheck.CharacterPath",
                        "{0} points to {1} \"{2}\", but that file is not there."), fileName, key, value));
                }
            }
        }
    }

    private static bool TryReadPathEntry(string line, bool yaml, out string key, out string value)
    {
        key = string.Empty;
        value = string.Empty;
        int split = line.IndexOf(yaml ? ':' : '=');
        if (split <= 0) return false;
        string name = line[..split].Trim().ToLowerInvariant();
        if (name is not ("image" or "sample" or "portrait")) return false;
        value = line[(split + 1)..].Trim().Trim('"', '\'');
        key = name;
        return value.Length > 0;
    }

    /// <summary>统计没被任何设置用到的音频，以及可以安全删除的缓存和垃圾文件。</summary>
    private static void AddFileFindings(USinger singer, string root, List<string> tips)
    {
        singer.EnsureLoaded();
        HashSet<string> referenced = new(StringComparer.OrdinalIgnoreCase);
        foreach (UOto oto in singer.Otos)
        {
            if (!string.IsNullOrEmpty(oto.File))
            {
                referenced.Add(Path.GetFullPath(oto.File));
            }
        }

        EnumerationOptions options = new() { RecurseSubdirectories = true, IgnoreInaccessible = true };
        List<string> unused = [];
        long unusedBytes = 0;
        int cacheCount = 0;
        long cacheBytes = 0;
        foreach (string file in Directory.EnumerateFiles(root, "*", options))
        {
            string name = Path.GetFileName(file).ToLowerInvariant();
            string extension = Path.GetExtension(file).ToLowerInvariant();
            long length = new FileInfo(file).Length;
            if (CacheExtensions.Contains(extension) || JunkFileNames.Contains(name) || file.Contains("__MACOSX", StringComparison.Ordinal))
            {
                cacheCount++;
                cacheBytes += length;
            }
            else if (extension == ".wav" && referenced.Count > 0 && !referenced.Contains(Path.GetFullPath(file)))
            {
                // 没有任何设置时无法判断哪些音频没被用到，因此要求 referenced 非空。
                unused.Add(Path.GetRelativePath(root, file));
                unusedBytes += length;
            }
        }

        if (unused.Count > 0)
        {
            tips.Add(string.Format(LocalizedText.S("VoicebankCheck.Unused",
                "{0} sound file(s) ({1}) are not used by any setting in oto.ini, so you can delete them to save space. For example: {2}"),
                unused.Count, FormatSize(unusedBytes), string.Join("; ", unused.Take(MaxExamples))));
        }
        if (cacheCount > 0)
        {
            tips.Add(string.Format(LocalizedText.S("VoicebankCheck.CacheFiles",
                "{0} helper file(s) ({1}) were made automatically by other programs (pitch caches, thumbnails). They are safe to delete and are rebuilt if needed."),
                cacheCount, FormatSize(cacheBytes)));
        }
    }

    private static string FormatSize(long bytes) => bytes switch
    {
        >= 1024 * 1024 => $"{bytes / 1024.0 / 1024.0:0.#} MB",
        >= 1024 => $"{bytes / 1024.0:0.#} KB",
        _ => $"{bytes} B",
    };
}
