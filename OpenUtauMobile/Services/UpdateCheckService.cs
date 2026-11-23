using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using OpenUtau.Core.Util;
using OpenUtauMobile.Helpers;
using OpenUtauMobile.Services.Dialogs;
using OpenUtauMobile.Services.Platform;
using Serilog;

namespace OpenUtauMobile.Services;

/// <summary>
/// 检查 GitHub 上是否有新版本，并把该版本的更新说明展示给用户。
/// 自动检查每个版本只提醒一次；手动检查总会给出结果。
/// </summary>
public static class UpdateCheckService
{
    /// <summary>发布版本所在的 GitHub 仓库（owner/name）。</summary>
    private const string Repository = "sparky-commits1637/OpenUtauMobile";

    private const int MaxNotesChars = 1500;

    private static readonly HttpClient Client = CreateClient();
    private static readonly Regex BuildRegex = new(@"build[\s_\-]*(\d+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex VersionRegex = new(@"\d+(\.\d+){1,3}", RegexOptions.Compiled);
    private static int _checking;

    private sealed record ReleaseInfo(string Tag, string Name, string Notes, string Url);

    private static HttpClient CreateClient()
    {
        HttpClient client = new() { Timeout = TimeSpan.FromSeconds(15) };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("OpenUtauMobile-UpdateCheck");
        client.DefaultRequestHeaders.Accept.ParseAdd("application/vnd.github+json");
        return client;
    }

    /// <summary>启动后延迟检查一次更新，失败时静默。</summary>
    public static async Task CheckOnStartupAsync()
    {
        await Task.Delay(TimeSpan.FromSeconds(3));
        await CheckAsync(manual: false);
    }

    /// <summary>检查更新；<paramref name="manual"/> 为 true 时无论结果都会给出提示。</summary>
    public static async Task CheckAsync(bool manual)
    {
        if (Interlocked.Exchange(ref _checking, 1) == 1) return;
        try
        {
            ReleaseInfo? release = await FetchLatestAsync();
            if (release == null)
            {
                if (manual) ToastService.Enqueue(LocalizedText.S("UpdateCheck.NoRelease", "No releases were found."));
                return;
            }

            bool? newer = IsNewer(release);
            if (newer == false)
            {
                if (manual) ToastService.Enqueue(LocalizedText.S("UpdateCheck.UpToDate", "You are on the latest version."));
                return;
            }
            // 无法判断新旧时只在手动检查中展示最新版本，避免自动弹窗误报。
            if (newer == null && !manual) return;
            if (!manual && ReadNotifiedTag() == release.Tag) return;

            await ShowReleaseAsync(release, newer == true);
            WriteNotifiedTag(release.Tag);
        }
        catch (Exception exception)
        {
            Log.Warning(exception, "Update check failed.");
            if (manual) ToastService.Enqueue(LocalizedText.S("UpdateCheck.Failed", "Could not check for updates. Check your connection."));
        }
        finally
        {
            Volatile.Write(ref _checking, 0);
        }
    }

    private static async Task<ReleaseInfo?> FetchLatestAsync()
    {
        // 使用列表接口以便包含标为预发布的版本；草稿不会返回。
        using HttpResponseMessage response = await Client.GetAsync($"https://api.github.com/repos/{Repository}/releases?per_page=1");
        response.EnsureSuccessStatusCode();
        string json = await response.Content.ReadAsStringAsync();
        using JsonDocument document = JsonDocument.Parse(json);
        if (document.RootElement.ValueKind != JsonValueKind.Array || document.RootElement.GetArrayLength() == 0)
        {
            return null;
        }
        JsonElement release = document.RootElement[0];
        string tag = ReadString(release, "tag_name");
        if (tag.Length == 0) return null;
        return new ReleaseInfo(tag, ReadString(release, "name"), ReadString(release, "body"), ReadString(release, "html_url"));
    }

    private static string ReadString(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : string.Empty;

    /// <summary>优先按构建号比较，其次按版本号比较；都无法解析时返回 null。</summary>
    private static bool? IsNewer(ReleaseInfo release)
    {
        Assembly assembly = typeof(UpdateCheckService).Assembly;
        string? localBuildText = assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
            .FirstOrDefault(attribute => attribute.Key == "BuildNumber")?.Value;
        Match remoteBuild = BuildRegex.Match($"{release.Tag} {release.Name}");
        if (remoteBuild.Success
            && long.TryParse(localBuildText, out long localBuild) && localBuild > 0
            && long.TryParse(remoteBuild.Groups[1].Value, out long remoteBuildNumber))
        {
            return remoteBuildNumber > localBuild;
        }

        Match remoteVersion = VersionRegex.Match(release.Tag);
        Match localVersion = VersionRegex.Match(CurrentVersionText());
        if (remoteVersion.Success && localVersion.Success
            && Version.TryParse(remoteVersion.Value, out Version? remote)
            && Version.TryParse(localVersion.Value, out Version? local))
        {
            return remote > local;
        }
        return null;
    }

    private static string CurrentVersionText()
    {
        string? version = typeof(UpdateCheckService).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        if (string.IsNullOrWhiteSpace(version)) return string.Empty;
        int plus = version.IndexOf('+');
        return plus >= 0 ? version[..plus] : version;
    }

    private static async Task ShowReleaseAsync(ReleaseInfo release, bool isNewer)
    {
        string title = string.Format(isNewer
            ? LocalizedText.S("UpdateCheck.NewTitle", "New version available: {0}")
            : LocalizedText.S("UpdateCheck.LatestTitle", "Latest release: {0}"), release.Tag);
        string current = CurrentVersionText();
        string content = string.Format(LocalizedText.S("UpdateCheck.Current", "You have: {0}"),
            current.Length == 0 ? L.S("Common.Unknown") : current)
            + Environment.NewLine + Environment.NewLine + CleanNotes(release.Notes);
        List<OptionConfirmOption> options =
        [
            new(LocalizedText.S("UpdateCheck.Open", "Open release page"), "open", isPrimary: true),
            new(LocalizedText.S("UpdateCheck.Later", "Later"), "later", isDefault: true),
        ];

        string? answer = await Dispatcher.UIThread.InvokeAsync(() => OptionConfirmPopupService.ShowAsync(title, content, options));
        if (answer == "open" && !string.IsNullOrEmpty(release.Url))
        {
            ExternalUrlLaunchResult result = await ExternalUrlService.OpenAsync(release.Url);
            if (!result.Succeeded)
            {
                ToastService.Enqueue(L.S("About.Toast.OpenLinkFailed"));
            }
        }
    }

    /// <summary>去掉 Markdown 标题和强调符号，并限制长度，让更新说明在弹窗里更易读。</summary>
    private static string CleanNotes(string notes)
    {
        if (string.IsNullOrWhiteSpace(notes))
        {
            return LocalizedText.S("UpdateCheck.NoNotes", "(this release has no patch notes)");
        }
        StringBuilder builder = new();
        foreach (string rawLine in notes.Replace("\r", string.Empty).Split('\n'))
        {
            string line = rawLine.TrimEnd();
            if (line.StartsWith('#')) line = line.TrimStart('#', ' ');
            builder.AppendLine(line.Replace("**", string.Empty).Replace("`", string.Empty));
        }
        string text = builder.ToString().Trim();
        return text.Length > MaxNotesChars ? text[..MaxNotesChars] + "…" : text;
    }

    private static string NotifiedTagPath => Path.Combine(PathManager.Inst.DataPath, "last-update-notified.txt");

    private static string ReadNotifiedTag()
    {
        try
        {
            return File.Exists(NotifiedTagPath) ? File.ReadAllText(NotifiedTagPath).Trim() : string.Empty;
        }
        catch (Exception exception)
        {
            Log.Debug(exception, "Failed to read last notified update tag.");
            return string.Empty;
        }
    }

    private static void WriteNotifiedTag(string tag)
    {
        try
        {
            File.WriteAllText(NotifiedTagPath, tag);
        }
        catch (Exception exception)
        {
            Log.Debug(exception, "Failed to save last notified update tag.");
        }
    }
}
