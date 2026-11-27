using System;

namespace OpenUtauMobile.Helpers;

/// <summary>
/// 读取本地化文本；当前语言缺少该键时回退为调用处给出的英文原文，避免界面显示键名。
/// </summary>
public static class LocalizedText
{
    /// <summary>返回本地化文本，缺失时返回 <paramref name="english"/>。</summary>
    public static string S(string key, string english)
    {
        string value = L.S(key);
        return string.Equals(value, key, StringComparison.Ordinal) ? english : value;
    }
}
