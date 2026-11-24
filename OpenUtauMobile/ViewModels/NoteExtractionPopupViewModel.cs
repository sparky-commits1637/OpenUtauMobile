using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using OpenUtau.Core.Analysis;
using OpenUtauMobile.Helpers;
using OpenUtauMobile.Services.NoteExtraction;
using ReactiveUI;
using ReactiveUI.Fody.Helpers;

namespace OpenUtauMobile.ViewModels;

public sealed record NoteExtractionLanguage(string? Code, string Label);

public sealed record NoteExtractionPitchModel(NotePitchModel Model, string Label);

public sealed class NoteExtractionPopupViewModel : PopupViewModelBase
{
    private readonly GameDependency onnx;
    private readonly GameDependency ggml;
    public bool RmvpeAvailable { get; }
    public bool TalkloidAvailable { get; }
    public bool MoonshineAvailable { get; }
    /// <summary>至少有一个音高模型可用时才允许勾选提取音高。</summary>
    public bool CanExtractPitch => RmvpeAvailable || TalkloidAvailable;
    private bool SelectedPitchAvailable => PitchModel?.Model == NotePitchModel.Talkloid ? TalkloidAvailable : RmvpeAvailable;
    public string RmvpeStatus => L.S(RmvpeAvailable ? "NoteExtraction.RmvpeInstalled" : "NoteExtraction.RmvpeMissing");
    public string PitchStatus => PitchModel?.Model == NotePitchModel.Talkloid
        ? L.S(TalkloidAvailable ? "NoteExtraction.TalkloidReady" : "NoteExtraction.TalkloidMissing")
        : RmvpeStatus;
    public string MoonshineStatus => L.S(MoonshineAvailable ? "NoteExtraction.MoonshineInstalled" : "NoteExtraction.MoonshineMissing");
    public IReadOnlyList<NoteExtractionPitchModel> PitchModels { get; }
    public IReadOnlyList<NoteExtractionLanguage> Languages { get; }
    public IReadOnlyList<int> SamplingStepOptions { get; } = [1, 2, 4, 8, 16];
    [Reactive] public NoteExtractionLanguage? Language { get; set; }
    [Reactive] public bool ExtractPitch { get; set; }
    [Reactive] public NoteExtractionPitchModel? PitchModel { get; set; }
    [Reactive] public bool ExtractLyrics { get; set; }
    [Reactive] public int SamplingSteps { get; set; } = 8;
    [Reactive] public decimal? BoundaryThreshold { get; set; } = 0.2m;
    [Reactive] public decimal? BoundaryRadius { get; set; } = 2;
    [Reactive] public decimal? ScoreThreshold { get; set; } = 0.2m;
    [Reactive] public decimal? BatchSize { get; set; } = 1;
    [Reactive] public decimal? MaxBatchDuration { get; set; } = 60;

    private bool Valid => Language != null && SamplingStepOptions.Contains(SamplingSteps)
        && BoundaryThreshold is >= 0 and <= 1 && ScoreThreshold is >= 0 and <= 1
        && BoundaryRadius is >= 0 and <= 10 && BoundaryRadius == decimal.Truncate(BoundaryRadius.Value)
        && BatchSize is >= 1 and <= 32 && BatchSize == decimal.Truncate(BatchSize.Value)
        && MaxBatchDuration is >= 0 and <= 600 && (!ExtractPitch || SelectedPitchAvailable) && (!ExtractLyrics || MoonshineAvailable);
    public bool HasInvalidSettings => !Valid;
    public bool CanUseOnnx => GameBackendProvider.IsSupported(NoteExtractionBackend.Onnx) && Valid && onnx.IsInstalled && onnx.SupportsLanguage(Language?.Code);
    public bool CanUseGgml => GameBackendProvider.IsSupported(NoteExtractionBackend.Ggml) && Valid && ggml.IsInstalled && ggml.SupportsLanguage(Language?.Code);
    public string OnnxStatus => Status("ONNX", onnx, NoteExtractionBackend.Onnx);
    public string GgmlStatus => Status("GGML", ggml, NoteExtractionBackend.Ggml);
    public ReactiveCommand<Unit, Unit> OnnxCommand { get; }
    public ReactiveCommand<Unit, Unit> GgmlCommand { get; }
    public ReactiveCommand<Unit, Unit> CancelCommand { get; }

    public NoteExtractionPopupViewModel() : this(GameBackendProvider.Inspect(NoteExtractionBackend.Onnx),
        GameBackendProvider.Inspect(NoteExtractionBackend.Ggml), RmvpeTranscriber.IsInstalled(),
        TalkloidPitchTranscriber.IsAvailable(), MoonshineTranscriber.IsInstalled()) { }

    internal NoteExtractionPopupViewModel(GameDependency onnx, GameDependency ggml, bool rmvpeAvailable,
        bool talkloidAvailable = false, bool moonshineAvailable = false)
    {
        this.onnx = onnx;
        this.ggml = ggml;
        RmvpeAvailable = rmvpeAvailable;
        TalkloidAvailable = talkloidAvailable;
        MoonshineAvailable = moonshineAvailable;
        PitchModels = [new(NotePitchModel.Rmvpe, "RMVPE"), new(NotePitchModel.Talkloid, "TalkLoid")];
        // 默认选第一个可用的模型，避免勾选提取音高后因默认项不可用而无法开始。
        PitchModel = PitchModels.FirstOrDefault(m => m.Model == NotePitchModel.Rmvpe ? rmvpeAvailable : talkloidAvailable) ?? PitchModels[0];
        Languages = new[] { new NoteExtractionLanguage(null, L.S("NoteExtraction.Universal")) }
            .Concat(new[] { onnx, ggml }.Where(d => d.IsInstalled)
                .SelectMany(d => d.Config.Languages?.Keys.AsEnumerable() ?? Enumerable.Empty<string>())
                .Distinct().OrderBy(code => code).Select(code => new NoteExtractionLanguage(code, code))).ToArray();
        Language = Languages[0];
        OnnxCommand = ReactiveCommand.Create(() => Confirm(NoteExtractionBackend.Onnx), this.WhenAnyValue(vm => vm.CanUseOnnx));
        GgmlCommand = ReactiveCommand.Create(() => Confirm(NoteExtractionBackend.Ggml), this.WhenAnyValue(vm => vm.CanUseGgml));
        CancelCommand = ReactiveCommand.Create(() => RaiseClose(null));
        PropertyChanged += (_, args) =>
        {
            if (args.PropertyName is nameof(PitchModel))
            {
                this.RaisePropertyChanged(nameof(PitchStatus));
            }
            if (args.PropertyName is nameof(Language) or nameof(ExtractPitch) or nameof(PitchModel) or nameof(ExtractLyrics) or nameof(SamplingSteps)
                or nameof(BoundaryThreshold) or nameof(BoundaryRadius) or nameof(ScoreThreshold)
                or nameof(BatchSize) or nameof(MaxBatchDuration))
            {
                this.RaisePropertyChanged(nameof(HasInvalidSettings));
                this.RaisePropertyChanged(nameof(CanUseOnnx));
                this.RaisePropertyChanged(nameof(CanUseGgml));
                this.RaisePropertyChanged(nameof(OnnxStatus));
                this.RaisePropertyChanged(nameof(GgmlStatus));
            }
        };
    }

    private string Status(string name, GameDependency dependency, NoteExtractionBackend backend) => string.Format(L.S("NoteExtraction.BackendStatus"),
        name, dependency.PackageId, L.S(!GameBackendProvider.IsSupported(backend) ? "NoteExtraction.Unsupported"
            : !dependency.IsInstalled ? "NoteExtraction.NotInstalled"
            : dependency.SupportsLanguage(Language?.Code) ? "Common.Installed" : "NoteExtraction.LanguageUnavailable"));

    private void Confirm(NoteExtractionBackend backend)
    {
        if (backend == NoteExtractionBackend.Onnx ? !CanUseOnnx : !CanUseGgml) return;
        NoteExtractionSettings settings = new(ExtractPitch, Language?.Code, SamplingSteps,
            (float)BoundaryThreshold!.Value, (int)BoundaryRadius!.Value, (float)ScoreThreshold!.Value,
            (int)BatchSize!.Value, (float)MaxBatchDuration!.Value,
            (PitchModel ?? PitchModels[0]).Model, ExtractLyrics);
        settings.Validate();
        RaiseClose(new NoteExtractionRequest(backend, settings));
    }

    public override void RequestBack() => RaiseClose(null);
}
