using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using OpenUtau.Classic;
using OpenUtau.Core;
using Serilog;

namespace OpenUtauMobile.Services.Resamplers;

/// <summary>在线目录中的单个文件（通常是 .sh 脚本或其附属文件）。</summary>
public sealed class CatalogFile
{
    /// <summary>安装后相对于该重采样器目录的路径。</summary>
    [JsonPropertyName("path")] public string Path { get; set; } = string.Empty;

    /// <summary>下载地址；相对地址将基于 index.json 所在目录解析。</summary>
    [JsonPropertyName("url")] public string Url { get; set; } = string.Empty;

    /// <summary>必填的 SHA-256（十六进制），下载后必须匹配。</summary>
    [JsonPropertyName("sha256")] public string? Sha256 { get; set; }

    /// <summary>是否需要在类 Unix 系统上授予可执行权限。</summary>
    [JsonPropertyName("executable")] public bool Executable { get; set; }
}

/// <summary>在线目录中的一个重采样器条目。</summary>
public sealed class CatalogResampler
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("name")] public string Name { get; set; } = string.Empty;
    [JsonPropertyName("description")] public string Description { get; set; } = string.Empty;
    [JsonPropertyName("author")] public string Author { get; set; } = string.Empty;

    /// <summary>该重采样器自身的源码仓库或主页。</summary>
    [JsonPropertyName("sourceUrl")] public string SourceUrl { get; set; } = string.Empty;
    [JsonPropertyName("version")] public string Version { get; set; } = "1";
    [JsonPropertyName("files")] public List<CatalogFile> Files { get; set; } = [];
}

/// <summary>目录索引文件（index.json）。</summary>
public sealed class CatalogIndex
{
    [JsonPropertyName("schema")] public int Schema { get; set; } = 1;
    [JsonPropertyName("resamplers")] public List<CatalogResampler> Resamplers { get; set; } = [];
}

/// <summary>已通过目录安装的重采样器记录，保存在其目录下的 .catalog.json。</summary>
public sealed class InstalledCatalogRecord
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("version")] public string Version { get; set; } = string.Empty;
}

/// <summary>
/// 在线重采样器目录：拉取开发者发布的 index.json，下载脚本到 Resamplers 目录并刷新列表。
/// 目录中的条目全部由开发者自己发布；用户自行添加的重采样器不受影响。
/// </summary>
public static class ResamplerCatalogService
{
    /// <summary>默认索引地址，开发者在该仓库中发布 index.json 与 .sh 文件。</summary>
    public const string DefaultIndexUrl =
        "https://raw.githubusercontent.com/sparky-commits1637/OpenUtauMobile-Resamplers/main/index.json";

    private const string RecordFileName = ".catalog.json";
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(30) };

    /// <summary>
    /// 目录签名公钥（PEM，ECDSA P-256）。私钥只保存在开发者手里，应用只能验证、无法签发。
    /// 为空时目录功能关闭，避免未签名内容被执行。
    /// </summary>
    public const string PublicKeyPem =
        "-----BEGIN PUBLIC KEY-----\n" +
        "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEgC1IISIJUxESADCozlURp02ChtxV\n" +
        "0s7s3I9Vnpt9w5sFgIJ2bJc/vNETvq3TBk1W19FlE2TzMeN6hhazrgSxLA==\n" +
        "-----END PUBLIC KEY-----\n";

    /// <summary>拉取并解析目录索引；签名无效或缺失时拒绝。</summary>
    public static async Task<CatalogIndex> FetchIndexAsync(string? indexUrl = null, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(PublicKeyPem))
            throw new InvalidOperationException("Catalog signing key is not configured.");
        string url = indexUrl ?? DefaultIndexUrl;
        byte[] data = await Http.GetByteArrayAsync(url, ct);
        byte[] signature = await Http.GetByteArrayAsync(url + ".sig", ct);
        using ECDsa ecdsa = ECDsa.Create();
        ecdsa.ImportFromPem(PublicKeyPem);
        if (!ecdsa.VerifyData(data, signature, HashAlgorithmName.SHA256, DSASignatureFormat.Rfc3279DerSequence))
            throw new InvalidDataException("Catalog signature is invalid.");
        CatalogIndex index = JsonSerializer.Deserialize<CatalogIndex>(data) ?? new CatalogIndex();
        // 每个文件都必须带 SHA-256，且只允许 https 或相对地址。
        index.Resamplers = [.. index.Resamplers.Where(r => IsSafeId(r.Id) && r.Files.Count > 0
            && r.Files.All(f => !string.IsNullOrWhiteSpace(f.Sha256) && IsAllowedUrl(f.Url)))];
        return index;
    }

    private static bool IsAllowedUrl(string url) =>
        !url.Contains("://") || url.StartsWith("https://", StringComparison.OrdinalIgnoreCase);

    /// <summary>读取某条目当前已安装的版本；未通过目录安装则返回 null。</summary>
    public static string? GetInstalledVersion(string id)
    {
        try
        {
            string file = Path.Combine(GetInstallDir(id), RecordFileName);
            if (!File.Exists(file)) return null;
            return JsonSerializer.Deserialize<InstalledCatalogRecord>(File.ReadAllText(file))?.Version;
        }
        catch (Exception e)
        {
            Log.Warning(e, "读取重采样器安装记录失败: {Id}", id);
            return null;
        }
    }

    /// <summary>下载并安装（或更新）一个条目，随后刷新重采样器列表。</summary>
    public static async Task InstallAsync(CatalogResampler entry, string? indexUrl = null,
        IProgress<double>? progress = null, CancellationToken ct = default)
    {
        if (!IsSafeId(entry.Id)) throw new InvalidOperationException("Invalid resampler id.");
        Uri baseUri = new(indexUrl ?? DefaultIndexUrl);
        string target = GetInstallDir(entry.Id);
        string staging = target + ".tmp";
        if (Directory.Exists(staging)) Directory.Delete(staging, true);
        Directory.CreateDirectory(staging);
        try
        {
            for (int i = 0; i < entry.Files.Count; i++)
            {
                CatalogFile file = entry.Files[i];
                string destination = ResolveInside(staging, file.Path);
                Directory.CreateDirectory(System.IO.Path.GetDirectoryName(destination)!);
                Uri source = new(baseUri, file.Url);
                byte[] data = await Http.GetByteArrayAsync(source, ct);
                string actual = Convert.ToHexString(SHA256.HashData(data));
                if (!actual.Equals(file.Sha256?.Trim(), StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException($"Checksum mismatch: {file.Path}");
                await File.WriteAllBytesAsync(destination, data, ct);
                if (file.Executable && !OperatingSystem.IsWindows())
                {
                    File.SetUnixFileMode(destination,
                        UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute |
                        UnixFileMode.GroupRead | UnixFileMode.GroupExecute |
                        UnixFileMode.OtherRead | UnixFileMode.OtherExecute);
                }
                progress?.Report((i + 1) / (double)entry.Files.Count);
            }
            File.WriteAllText(System.IO.Path.Combine(staging, RecordFileName),
                JsonSerializer.Serialize(new InstalledCatalogRecord { Id = entry.Id, Version = entry.Version }));
            // 全部下载并校验成功后才替换旧版本，失败不会破坏已安装的内容。
            if (Directory.Exists(target)) Directory.Delete(target, true);
            Directory.Move(staging, target);
        }
        catch
        {
            if (Directory.Exists(staging)) Directory.Delete(staging, true);
            throw;
        }
        await Task.Run(() => ToolsManager.Inst.Initialize(), ct);
    }

    /// <summary>卸载通过目录安装的条目。</summary>
    public static async Task UninstallAsync(string id)
    {
        if (!IsSafeId(id)) return;
        string dir = GetInstallDir(id);
        if (Directory.Exists(dir)) Directory.Delete(dir, true);
        await Task.Run(() => ToolsManager.Inst.Initialize());
    }

    /// <summary>列出用户自行添加的重采样器（不含目录安装的条目和内置项）。</summary>
    public static List<string> ListOwnResamplers()
    {
        string root = PathManager.Inst.ResamplersPath;
        List<string> result = [];
        foreach (IResampler resampler in ToolsManager.Inst.Resamplers)
        {
            string name = resampler.ToString() ?? string.Empty;
            string full = System.IO.Path.Combine(root, name);
            if (!File.Exists(full)) continue; // 内置重采样器没有对应文件
            string? dir = System.IO.Path.GetDirectoryName(full);
            if (dir != null && File.Exists(System.IO.Path.Combine(dir, RecordFileName))) continue;
            result.Add(name);
        }
        return result;
    }

    private static string GetInstallDir(string id) =>
        System.IO.Path.Combine(PathManager.Inst.ResamplersPath, id);

    private static bool IsSafeId(string id) =>
        !string.IsNullOrWhiteSpace(id) && id.All(c => char.IsLetterOrDigit(c) || c is '-' or '_' or '.') && !id.StartsWith('.');

    /// <summary>把相对路径解析到根目录之内，拒绝越界路径。</summary>
    private static string ResolveInside(string root, string relative)
    {
        string full = System.IO.Path.GetFullPath(System.IO.Path.Combine(root, relative));
        string normalizedRoot = System.IO.Path.GetFullPath(root) + System.IO.Path.DirectorySeparatorChar;
        if (!full.StartsWith(normalizedRoot, StringComparison.Ordinal))
            throw new InvalidOperationException($"Unsafe path: {relative}");
        return full;
    }
}
