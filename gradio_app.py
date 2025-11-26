import os
import shutil
import tempfile
from typing import Optional, Tuple
import gradio as gr


# ---------------- Utilities ----------------
def _ensure_file_exists(path: str, name: str):
    if not path or not os.path.isfile(path):
        raise gr.Error(f"{name} not found: {path}")


def _maybe_default_bs_roformer_paths() -> Tuple[str, str]:
    cfg = os.path.join("accom_separation","ckpt", "bs_roformer", "config_bd_roformer.yaml")
    ckpt = os.path.join("accom_separation","ckpt", "bs_roformer", "bs_roformer.ckpt")
    return (cfg if os.path.isfile(cfg) else "", ckpt if os.path.isfile(ckpt) else "")


def _default_svc_cfg() -> str:
    candidate = os.path.join("configs", "YingMusic-SVC.yml")
    return candidate if os.path.isfile(candidate) else ""


def _locate_stem(out_dir: str, name_candidates: Tuple[str, ...]) -> Optional[str]:
    for stem in name_candidates:
        wav_path = os.path.join(out_dir, f"{stem}.wav")
        flac_path = os.path.join(out_dir, f"{stem}.flac")
        if os.path.isfile(wav_path):
            return wav_path
        if os.path.isfile(flac_path):
            return flac_path
    return None

 
# ---------------- Separation ----------------
def run_separation(
    mix_file: str,
    sep_config_path: str,
    sep_checkpoint_path: str,
    sep_device_id: int,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (lead, backing, accompany)
    """
    if not mix_file:
        raise gr.Error("Please upload an accompanied audio file.")
    _ensure_file_exists(mix_file, "Input audio")
    _ensure_file_exists(sep_config_path, "Separation config")
    _ensure_file_exists(sep_checkpoint_path, "Separation checkpoint")

    input_dir = tempfile.mkdtemp(prefix="ams_in_")
    output_dir = tempfile.mkdtemp(prefix="ams_out_")
    mix_basename = os.path.basename(mix_file)
    mix_copy_path = os.path.join(input_dir, mix_basename)
    shutil.copy2(mix_file, mix_copy_path)

    from accom_separation.inference import proc_folder

    args = {
        "model_type": "bs_roformer",
        "config_path": sep_config_path,
        "start_check_point": sep_checkpoint_path,
        "input_folder": input_dir,
        "store_dir": output_dir,
        "extract_instrumental": True,
        "extract_other": False,
        "device_ids": [int(sep_device_id)],
        "disable_detailed_pbar": True,
        "force_cpu": False,
        "flac_file": False,
        "use_tta": False,
    }
    proc_folder(args)

    song_name = os.path.splitext(mix_basename)[0]
    song_out_dir = os.path.join(output_dir, song_name)

    lead = _locate_stem(song_out_dir, ("vocals", "lead", "leading_vocals"))
    backing = _locate_stem(song_out_dir, ("backing_vocal", "backing", "back_vocals", "background"))
    accompany = _locate_stem(song_out_dir, ("instrumental", "accompaniment"))

    if not any([lead, backing, accompany]):
        raise gr.Error("No stems found. Please check separation config and checkpoint.")
    return lead, backing, accompany


# ---------------- SVC ----------------
def run_svc(
    source_vocal: str,
    target_timbre: str,
    accompany_file: Optional[str],
    svc_checkpoint: str,
    svc_config: str,
    svc_steps: int,
    svc_cuda_index: int,
    svc_fp16: bool,
) -> Tuple[str, Optional[str]]:
    """
    Return (converted vocal, mixed with accompaniment when provided)
    """
    if not source_vocal:
        raise gr.Error("Please upload the source vocal.")
    if not target_timbre:
        raise gr.Error("Please upload the reference timbre.")
    _ensure_file_exists(source_vocal, "Source vocal")
    _ensure_file_exists(target_timbre, "Reference timbre")
    _ensure_file_exists(svc_checkpoint, "SVC checkpoint")
    _ensure_file_exists(svc_config, "SVC config")
    if accompany_file:
        _ensure_file_exists(accompany_file, "Accompaniment file")

    import types
    import torch
    from Remix.auger import echo_then_reverb_save
    from my_inference import load_models_api, run_inference

    args = types.SimpleNamespace()
    args.source = source_vocal
    args.target = target_timbre
    args.diffusion_steps = int(svc_steps)
    args.checkpoint = svc_checkpoint
    args.expname = "gradio"
    args.cuda = torch.device(f"cuda:{int(svc_cuda_index)}") if torch.cuda.is_available() else torch.device("cpu")
    args.fp16 = bool(svc_fp16)
    args.accompany = accompany_file if accompany_file else None
    args.config = svc_config
    args.length_adjust = 1.0
    args.inference_cfg_rate = 0.7
    args.f0_condition = True
    args.semi_tone_shift = None
    args.output = "./outputs"
    os.makedirs(args.output, exist_ok=True)

    model_bundle = load_models_api(args, device=args.cuda)
    vc_path = run_inference(args, model_bundle, device=args.cuda)

    mixed_out = None
    if accompany_file:
        base_dir = os.path.dirname(vc_path)
        fname = os.path.basename(vc_path)
        acc_dir = os.path.join(base_dir, "accompany")
        os.makedirs(acc_dir, exist_ok=True)
        mixed_out = os.path.join(acc_dir, fname)
        echo_then_reverb_save(vc_path, mixed_out, accompany_file)
    return vc_path, mixed_out


# ---------------- One-click Pipeline ----------------
def run_pipeline(
    mix_file: str,
    target_timbre: str,
    # separation settings
    sep_config_path: str,
    sep_checkpoint_path: str,
    sep_device_id: int,
    # svc settings
    svc_checkpoint: str,
    svc_config: str,
    svc_steps: int,
    svc_cuda_index: int,
    svc_fp16: bool,
) -> Tuple[Optional[str], Optional[str], Optional[str], str, Optional[str]]:
    lead, backing, accompany = run_separation(
        mix_file=mix_file,
        sep_config_path=sep_config_path,
        sep_checkpoint_path=sep_checkpoint_path,
        sep_device_id=sep_device_id,
    )
    vc_path, mixed_out = run_svc(
        source_vocal=lead or "",
        target_timbre=target_timbre,
        accompany_file=accompany,
        svc_checkpoint=svc_checkpoint,
        svc_config=svc_config,
        svc_steps=svc_steps,
        svc_cuda_index=svc_cuda_index,
        svc_fp16=svc_fp16,
    )
    return lead, backing, accompany, vc_path, mixed_out


# ---------------- UI ----------------
def build_ui():
    default_sep_cfg, default_sep_ckpt = _maybe_default_bs_roformer_paths()
    default_svc_cfg = _default_svc_cfg()
    header_img = os.path.join("figs", "svc_main.jpg")
    has_header_img = os.path.isfile(header_img)

    # discover sample timbres
    prompt_dir = "prompts"
    audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    sample_timbres = []
    if os.path.isdir(prompt_dir):
        for fname in sorted(os.listdir(prompt_dir)):
            if fname.lower().endswith(audio_exts):
                sample_timbres.append(os.path.join(prompt_dir, fname))
            if len(sample_timbres) >= 2:
                break

    with gr.Blocks(
        title="YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion with Flow-GRPO and Singing-Specific Inductive Biases",
        theme=gr.themes.Soft(),
    ) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion with Flow-GRPO and Singing-Specific Inductive Biases")
            if has_header_img:
                with gr.Column(scale=1):
                    gr.Image(header_img, show_label=False, height=140)

        with gr.Tab("🧩 Global Settings"):
            gr.Markdown("Set shared parameters here. All three modes reuse these values.")
            with gr.Group():
                gr.Markdown("**Separation Settings**")
                sep_cfg_inp = gr.Textbox(value=default_sep_cfg, label="Separation Config (config_path)", info="Path to bs_roformer YAML config")
                sep_ckpt_inp = gr.Textbox(value=default_sep_ckpt, label="Separation Checkpoint (start_check_point)", info="Path to bs_roformer ckpt")
                sep_device_inp = gr.Number(value=0, label="Separation GPU (device id)", precision=0, info="Use 0 for default; on CPU-only, it will fallback automatically")
            with gr.Group():
                gr.Markdown("**SVC Settings**")
                svc_ckpt_inp = gr.Textbox(value="", label="SVC Checkpoint", info="Path to SVC .pt weights")
                svc_cfg_inp = gr.Textbox(value=default_svc_cfg, label="SVC Config", info="YAML config path (defaults to configs/YingMusic-SVC.yml if present)")
                svc_steps_inp = gr.Number(value=100, label="Diffusion Steps", precision=0, info="Higher is steadier but slower; 100 is a good balance")
                svc_cuda_inp = gr.Number(value=0, label="SVC GPU (cuda)", precision=0, info="GPU index when multiple GPUs are available")
                svc_fp16_inp = gr.Checkbox(value=True, label="FP16", info="Saves VRAM and speeds up inference on supported GPUs")

        with gr.Tab("🎚️ Accompaniment Separation"):
            gr.Markdown("Upload an accompanied track to obtain Lead (vocals), Backing vocals, and pure Accompaniment.")
            with gr.Row():
                with gr.Column(scale=1):
                    mix_file = gr.Audio(label="Accompanied Audio", type="filepath")
                    # example if available
                    example_path = os.path.join("accom_separation", "samples", "raw", "All I Want For Christmas Is You01.MP3")
                    if os.path.isfile(example_path):
                        gr.Examples(
                            examples=[[example_path]],
                            inputs=[mix_file],
                            label="Examples",
                        )
                    btn_sep = gr.Button("Start Separation", variant="primary")
                with gr.Column(scale=1):
                    out_lead = gr.Audio(label="Lead", type="filepath")
                    out_back = gr.Audio(label="Backing", type="filepath")
                    out_acc = gr.Audio(label="Accompaniment", type="filepath")
            btn_sep.click(
                fn=run_separation,
                inputs=[mix_file, sep_cfg_inp, sep_ckpt_inp, sep_device_inp],
                outputs=[out_lead, out_back, out_acc],
            )

        with gr.Tab("🎤 Standalone SVC"):
            gr.Markdown("Upload source vocal and reference timbre (optionally upload accompaniment to auto-mix).")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        src_vocal = gr.Audio(label="Source Vocal (leading)", type="filepath")
                        ref_timbre = gr.Audio(label="Reference Timbre", type="filepath")
                    # sample timbre buttons
                    if len(sample_timbres) >= 1:
                        btn_t1 = gr.Button("Use Sample Timbre 1")
                    if len(sample_timbres) >= 2:
                        btn_t2 = gr.Button("Use Sample Timbre 2")
                    accompany_opt = gr.Audio(label="Accompaniment (optional)", type="filepath")
                    btn_svc = gr.Button("Convert", variant="primary")
                with gr.Column(scale=1):
                    out_vc = gr.Audio(label="Converted Vocal", type="filepath")
                    out_mixed = gr.Audio(label="Mixed with Accompaniment (if provided)", type="filepath")

            # wire sample buttons for SVC
            def _sample_1():
                return sample_timbres[0] if len(sample_timbres) >= 1 else None

            def _sample_2():
                return sample_timbres[1] if len(sample_timbres) >= 2 else None

            if len(sample_timbres) >= 1:
                btn_t1.click(fn=_sample_1, inputs=[], outputs=[ref_timbre])
            if len(sample_timbres) >= 2:
                btn_t2.click(fn=_sample_2, inputs=[], outputs=[ref_timbre])

            btn_svc.click(
                fn=run_svc,
                inputs=[
                    src_vocal,
                    ref_timbre,
                    accompany_opt,
                    svc_ckpt_inp,
                    svc_cfg_inp,
                    svc_steps_inp,
                    svc_cuda_inp,
                    svc_fp16_inp,
                ],
                outputs=[out_vc, out_mixed],
            )

        with gr.Tab("⚡ One-Click Separation + SVC"):
            gr.Markdown("Upload accompanied audio and reference timbre, then run separation and conversion in one go.")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        mix_in = gr.Audio(label="Accompanied Audio", type="filepath")
                        ref_in = gr.Audio(label="Reference Timbre", type="filepath")
                    # example if available
                    example_path2 = os.path.join("accom_separation", "samples", "raw", "All I Want For Christmas Is You01.MP3")
                    if os.path.isfile(example_path2):
                        gr.Examples(
                            examples=[[example_path2]],
                            inputs=[mix_in],
                            label="Examples",
                        )
                    # sample timbre buttons for pipeline
                    if len(sample_timbres) >= 1:
                        btn_pt1 = gr.Button("Use Sample Timbre 1")
                    if len(sample_timbres) >= 2:
                        btn_pt2 = gr.Button("Use Sample Timbre 2")
                    btn_all = gr.Button("Process All", variant="primary")
                with gr.Column(scale=1):
                    oc_lead = gr.Audio(label="Lead", type="filepath")
                    oc_back = gr.Audio(label="Backing", type="filepath")
                    oc_acc = gr.Audio(label="Accompaniment", type="filepath")
                    oc_vc = gr.Audio(label="Converted Vocal", type="filepath")
                    oc_mixed = gr.Audio(label="Mixed with Accompaniment", type="filepath")

            # wire sample buttons for pipeline
            if len(sample_timbres) >= 1:
                btn_pt1.click(fn=_sample_1, inputs=[], outputs=[ref_in])
            if len(sample_timbres) >= 2:
                btn_pt2.click(fn=_sample_2, inputs=[], outputs=[ref_in])

            btn_all.click(
                fn=run_pipeline,
                inputs=[
                    mix_in,
                    ref_in,
                    sep_cfg_inp,
                    sep_ckpt_inp,
                    sep_device_inp,
                    svc_ckpt_inp,
                    svc_cfg_inp,
                    svc_steps_inp,
                    svc_cuda_inp,
                    svc_fp16_inp,
                ],
                outputs=[oc_lead, oc_back, oc_acc, oc_vc, oc_mixed],
            )

        gr.Markdown("— Enjoy —")
        return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_port=30911,inbrowser=True,server_name='0.0.0.0')

