using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenUtau.Classic;
using OpenUtau.Core;
using OpenUtau.Core.Util;
using OpenUtauMobile.Helpers;
using Serilog;

namespace OpenUtauMobile.Services;

/// <summary>重采样器测试结果。</summary>
/// <param name="Success">是否成功生成了测试音频。</param>
/// <param name="Summary">给普通用户看的一句话结论。</param>
/// <param name="Details">退出码与程序输出，用于排查问题。</param>
public sealed record ResamplerTestResult(bool Success, string Summary, string Details);

/// <summary>
/// 携带测试详情的异常，仅用于把详情交给错误弹窗展示；<see cref="ToString"/> 只返回详情文本。
/// </summary>
public sealed class ResamplerTestException : Exception
{
    public ResamplerTestException(string details) : base(details)
    {
    }

    public override string ToString() => Message;
}

/// <summary>
/// 对外部重采样器做一次真实调用：生成一小段测试音频，按真实渲染的参数格式运行程序，
/// 并把退出码和程序输出整理成容易读懂的结果。
/// </summary>
public static class ResamplerTestService
{
    private const int TimeoutMs = 30000;
    private const int MaxOutputChars = 1200;

    /// <summary>在后台线程测试指定名称的重采样器。</summary>
    public static Task<ResamplerTestResult> RunAsync(string resamplerName) => Task.Run(() => Run(resamplerName));

    private static ResamplerTestResult Run(string resamplerName)
    {
        IResampler? resampler = ToolsManager.Inst.Resamplers.FirstOrDefault(tool => tool.ToString() == resamplerName);
        if (resampler == null)
        {
            return new(false,
                string.Format(LocalizedText.S("ResamplerTest.NotFound", "Resampler \"{0}\" was not found."), resamplerName),
                string.Empty);
        }

        string? filePath = resampler.FilePath;
        if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
        {
            // 内置重采样器没有外部程序文件，无需测试。
            return new(true,
                string.Format(LocalizedText.S("ResamplerTest.BuiltIn", "{0} is built into the app, so there is nothing to test."), resamplerName),
                string.Empty);
        }

        string workDir = Path.Combine(PathManager.Inst.CachePath, "resampler-test");
        string inputFile = Path.Combine(workDir, "input.wav");
        string outputFile = Path.Combine(workDir, "output.wav");
        try
        {
            Directory.CreateDirectory(workDir);
            File.Delete(outputFile);
            WriteTestWave(inputFile);
            try
            {
                resampler.CheckPermissions();
            }
            catch (Exception exception)
            {
                Log.Warning(exception, "Failed to set resampler permissions for test.");
            }
            return Execute(resampler, filePath, inputFile, outputFile);
        }
        catch (Exception exception)
        {
            Log.Error(exception, "Resampler test failed unexpectedly.");
            return new(false,
                LocalizedText.S("ResamplerTest.Unexpected", "The test could not be completed."),
                exception.Message);
        }
        finally
        {
            TryDelete(inputFile);
            TryDelete(outputFile);
        }
    }

    private static ResamplerTestResult Execute(IResampler resampler, string filePath, string inputFile, string outputFile)
    {
        // 参数顺序与真实渲染相同：输入、输出、音高、力度、flags、偏移、时长、辅音、截断、音量、调制、速度、音高曲线。
        int[] pitches = new int[40];
        string arguments = FormattableString.Invariant(
            $"\"{inputFile}\" \"{outputFile}\" {MusicMath.GetToneName(60)} 100 \"\" 0 300 0 0 100 0 !120 {Base64.Base64EncodeInt12(pitches)}");

        // 与真实渲染一致：设置了 Wine 且程序是 Windows 可执行文件时，通过 Wine 启动。
        string winePath = Preferences.Default.WinePath;
        string extension = Path.GetExtension(filePath).ToLowerInvariant();
        bool useWine = !string.IsNullOrEmpty(winePath) && (extension == ".exe" || extension == ".bat");
        ProcessStartInfo startInfo = new(useWine ? winePath : filePath, useWine ? $"{filePath} {arguments}" : arguments)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        StringBuilder output = new();
        object outputLock = new();
        using Process process = new() { StartInfo = startInfo };
        process.OutputDataReceived += (_, e) => AppendLine(output, outputLock, e.Data);
        process.ErrorDataReceived += (_, e) => AppendLine(output, outputLock, e.Data);

        try
        {
            process.Start();
        }
        catch (Exception exception) when (exception is Win32Exception or IOException or UnauthorizedAccessException or InvalidOperationException)
        {
            return new(false,
                LocalizedText.S("ResamplerTest.StartFailed",
                    "The system could not start this resampler. It may not be made for this device, or it was blocked. Windows .exe files usually cannot run on Android."),
                exception.Message);
        }

        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        if (!process.WaitForExit(TimeoutMs))
        {
            try
            {
                process.Kill(entireProcessTree: true);
            }
            catch (Exception exception)
            {
                Log.Warning(exception, "Failed to kill resampler test process.");
            }
            return new(false,
                LocalizedText.S("ResamplerTest.Timeout", "The resampler did not finish within 30 seconds, so the test was stopped."),
                FormatDetails(null, output, outputLock));
        }
        // 无参数的 WaitForExit 会等待异步输出读取完成。
        process.WaitForExit();

        int exitCode = process.ExitCode;
        bool produced = File.Exists(outputFile) && new FileInfo(outputFile).Length > 44;
        string details = FormatDetails(exitCode, output, outputLock);
        if (produced)
        {
            string summary = exitCode == 0
                ? LocalizedText.S("ResamplerTest.Success", "The resampler works: it made a test sound.")
                : string.Format(LocalizedText.S("ResamplerTest.SuccessWithCode",
                    "The resampler made a test sound, but it reported exit code {0}. It should still work."), exitCode);
            return new(true, summary, details);
        }

        string failure = exitCode == 0
            ? LocalizedText.S("ResamplerTest.NoOutput", "The resampler finished without errors but did not create a sound file.")
            : string.Format(LocalizedText.S("ResamplerTest.ExitError", "The resampler started but stopped with an error (exit code {0})."), exitCode);
        return new(false, failure, details);
    }

    private static void AppendLine(StringBuilder output, object outputLock, string? line)
    {
        if (line == null) return;
        lock (outputLock)
        {
            output.AppendLine(line);
        }
    }

    private static string FormatDetails(int? exitCode, StringBuilder output, object outputLock)
    {
        string text;
        lock (outputLock)
        {
            text = output.ToString().Trim();
        }
        if (text.Length > MaxOutputChars)
        {
            text = "…" + text[^MaxOutputChars..];
        }
        if (text.Length == 0)
        {
            text = LocalizedText.S("ResamplerTest.NoText", "(the program printed nothing)");
        }
        string code = exitCode.HasValue
            ? string.Format(LocalizedText.S("ResamplerTest.ExitCodeLine", "Exit code: {0}"), exitCode.Value)
            : LocalizedText.S("ResamplerTest.NoExitCode", "Exit code: none (stopped)");
        return code + Environment.NewLine + LocalizedText.S("ResamplerTest.OutputLabel", "Program output:") + Environment.NewLine + text;
    }

    /// <summary>写入一段约半秒、44.1 kHz、16 位单声道的 220 Hz 正弦波作为测试输入。</summary>
    private static void WriteTestWave(string path)
    {
        const int sampleRate = 44100;
        const int sampleCount = sampleRate / 2;
        const int dataSize = sampleCount * 2;
        using FileStream stream = File.Create(path);
        using BinaryWriter writer = new(stream);
        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(36 + dataSize);
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));
        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2);
        writer.Write((short)2);
        writer.Write((short)16);
        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(dataSize);
        for (int i = 0; i < sampleCount; i++)
        {
            double envelope = Math.Min(1.0, Math.Min(i, sampleCount - i) / 2000.0);
            writer.Write((short)(Math.Sin(2 * Math.PI * 220 * i / sampleRate) * 12000 * envelope));
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (Exception exception)
        {
            Log.Debug(exception, "Failed to delete resampler test file.");
        }
    }
}
