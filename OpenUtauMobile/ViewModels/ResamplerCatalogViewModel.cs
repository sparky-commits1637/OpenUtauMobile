using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Reactive;
using System.Threading.Tasks;
using Avalonia.Threading;
using Classic;
using OpenUtauMobile.Helpers;
using OpenUtauMobile.Services.Dialogs;
using OpenUtauMobile.Services.Resamplers;
using OpenUtauMobile.Storage;
using ReactiveUI;
using ReactiveUI.Fody.Helpers;
using Serilog;

namespace OpenUtauMobile.ViewModels;

/// <summary>目录中单个重采样器的列表项，负责安装、更新与卸载。</summary>
public class CatalogResamplerItemViewModel : ViewModelBase
{
    private readonly CatalogResampler _entry;
    private readonly ResamplerCatalogViewModel _owner;

    public string Name => string.IsNullOrWhiteSpace(_entry.Name) ? _entry.Id : _entry.Name;
    public string Description => _entry.Description;
    public string Author => _entry.Author;
    public string SourceUrl => _entry.SourceUrl;
    public string Version => _entry.Version;

    /// <summary>当前已安装版本；未安装为空。</summary>
    [Reactive] public string? InstalledVersion { get; private set; }
    [Reactive] public bool IsBusy { get; private set; }
    [Reactive] public double Progress { get; private set; }

    public bool IsInstalled => InstalledVersion != null;
    public bool HasUpdate => InstalledVersion != null && InstalledVersion != _entry.Version;

    public ReactiveCommand<Unit, Unit> InstallCommand { get; }
    public ReactiveCommand<Unit, Unit> RemoveCommand { get; }

    public CatalogResamplerItemViewModel(CatalogResampler entry, ResamplerCatalogViewModel owner)
    {
        _entry = entry;
        _owner = owner;
        InstalledVersion = ResamplerCatalogService.GetInstalledVersion(entry.Id);
        InstallCommand = ReactiveCommand.CreateFromTask(InstallAsync);
        RemoveCommand = ReactiveCommand.CreateFromTask(RemoveAsync);
    }

    private async Task InstallAsync()
    {
        IsBusy = true;
        Progress = 0;
        try
        {
            Progress<double> progress = new(p => Dispatcher.UIThread.Post(() => Progress = p));
            await ResamplerCatalogService.InstallAsync(_entry, ResamplerCatalogService.DefaultIndexUrl, progress);
            InstalledVersion = ResamplerCatalogService.GetInstalledVersion(_entry.Id);
            this.RaisePropertyChanged(nameof(IsInstalled));
            this.RaisePropertyChanged(nameof(HasUpdate));
            ToastService.Enqueue(L.S("ResamplerCatalog.Toast.Installed"));
            _owner.RefreshOwn();
        }
        catch (Exception e)
        {
            Log.Error(e, "安装重采样器失败: {Id}", _entry.Id);
            ToastService.Enqueue(L.S("ResamplerCatalog.Toast.Failed") + " " + e.Message);
        }
        finally { IsBusy = false; }
    }

    private async Task RemoveAsync()
    {
        IsBusy = true;
        try
        {
            await ResamplerCatalogService.UninstallAsync(_entry.Id);
            InstalledVersion = null;
            this.RaisePropertyChanged(nameof(IsInstalled));
            this.RaisePropertyChanged(nameof(HasUpdate));
        }
        catch (Exception e)
        {
            Log.Error(e, "卸载重采样器失败: {Id}", _entry.Id);
            ToastService.Enqueue(L.S("ResamplerCatalog.Toast.Failed") + " " + e.Message);
        }
        finally { IsBusy = false; }
    }
}

/// <summary>重采样器页面：「在线目录」与「我自己的」两个标签页。</summary>
public class ResamplerCatalogViewModel : NavigateViewModelBase
{
    /// <summary>当前标签：0=在线目录，1=我自己的。</summary>
    [Reactive] public int SelectedTabIndex { get; set; }
    [Reactive] public bool IsLoading { get; private set; }
    [Reactive] public string Error { get; private set; } = string.Empty;

    public ObservableCollection<CatalogResamplerItemViewModel> Items { get; } = [];
    public ObservableCollection<string> OwnResamplers { get; } = [];

    public ReactiveCommand<Unit, Unit> BackCommand { get; }
    public ReactiveCommand<Unit, Unit> RefreshCommand { get; }
    public ReactiveCommand<Unit, Unit> AddFromFileCommand { get; }

    public ResamplerCatalogViewModel(MainViewModel navigator) : base(navigator)
    {
        BackCommand = ReactiveCommand.Create(() => Navigator.NavigateBack(this));
        RefreshCommand = ReactiveCommand.CreateFromTask(LoadAsync);
        AddFromFileCommand = ReactiveCommand.CreateFromTask(AddFromFileAsync);
        RefreshOwn();
        _ = LoadAsync();
    }

    /// <summary>重新读取用户自行添加的重采样器列表。</summary>
    public void RefreshOwn()
    {
        OwnResamplers.Clear();
        foreach (string name in ResamplerCatalogService.ListOwnResamplers()) OwnResamplers.Add(name);
    }

    private async Task LoadAsync()
    {
        IsLoading = true;
        Error = string.Empty;
        try
        {
            CatalogIndex index = await ResamplerCatalogService.FetchIndexAsync();
            Items.Clear();
            foreach (CatalogResampler entry in index.Resamplers)
                Items.Add(new CatalogResamplerItemViewModel(entry, this));
        }
        catch (Exception e)
        {
            Log.Warning(e, "加载重采样器目录失败");
            Error = L.S("ResamplerCatalog.Error.Load");
        }
        finally { IsLoading = false; }
    }

    /// <summary>用户自己下载的重采样器：选择 .sh 文件并复制到 Resamplers 目录。</summary>
    private async Task AddFromFileAsync()
    {
        try
        {
            string path = await FilePicker.PickSingleFileAsync(L.S("ResamplerCatalog.FilePicker.Title"), ["*.sh", "*.exe", "*.bat"]);
            if (string.IsNullOrEmpty(path)) return;
            await Task.Run(() => ExeInstaller.Install(path, ExeType.resampler));
            await Task.Run(() => OpenUtau.Classic.ToolsManager.Inst.Initialize());
            RefreshOwn();
            ToastService.Enqueue(L.S("ResamplerCatalog.Toast.Installed"));
        }
        catch (Exception e)
        {
            Log.Error(e, "从文件添加重采样器失败");
            ToastService.Enqueue(L.S("ResamplerCatalog.Toast.Failed") + " " + e.Message);
        }
    }
}
