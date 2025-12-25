import os

import numpy as np

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import shutil
import warnings
import argparse
import torch
import yaml
warnings.simplefilter('ignore')

# load packages
import random
from modules.commons import *
import time
from Remix.auger import echo_then_reverb_save
import torchaudio
import librosa
from modules.commons import str2bool
from mm4 import preprocess_voice_conversion
from hf_utils import load_custom_model_from_hf

########## tools ##########
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

def load_models_api(args, device=torch.device("cuda")):
    dit_checkpoint_path = args.checkpoint
    print(f'load model from {dit_checkpoint_path}')
    dit_config_path = args.config
    print(f'load config from {dit_config_path}')
    # f0 extractor
    from modules.rmvpe import RMVPE

    model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)

    f0_extractor = RMVPE(model_path, is_half=False, device=device)
    f0_fn = f0_extractor.infer_from_audio

    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )

    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )

    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name

        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)

        #bigvgan_cache is not defined, and hence causing not defined error
        #bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_cache, use_cuda_kernel=False) 

        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name

        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        f0_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )


@torch.no_grad()
def run_inference(args, model_bundle, device=torch.device("cuda")):

    dit_config_path = args.config
    config = yaml.safe_load(open(dit_config_path, "r"))

    # use_style_residual
    use_style_residual = config['model_params']['length_regulator'].get('use_style_residual',False)

    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = model_bundle
    fp16 = args.fp16
    sr = mel_fn_args['sampling_rate']
    f0_condition = args.f0_condition
    forch_pitch_shift = args.semi_tone_shift

    source = args.source

    target_name = args.target
    print(source, target_name)
    diffusion_steps = args.diffusion_steps
    length_adjust = args.length_adjust
    inference_cfg_rate = args.inference_cfg_rate
    exp_path = os.path.join(args.output , args.expname)
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target_name, sr=sr)[0]


    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length


    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    time_vc_start = time.time()
    # Resample
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    # if source audio less than 30 seconds, whisper can handle in one forward
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1)
            S_chunk = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_chunk)
            else:
                S_alt_list.append(S_chunk[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)


    mel = mel_fn(source_audio.float())
    mel2 = mel_fn(ref_audio.float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        shifted_log_f0_alt = log_f0_alt.clone()
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)

        # automatic f0 adjust
        shifted_f0_alt, pitch_shift = preprocess_voice_conversion(
            voiced_f0_ori=voiced_F0_ori,
            voiced_f0_alt=voiced_F0_alt,
            shifted_f0_alt=shifted_f0_alt,
            enable_adaptive=True,
            max_shift_semitones=24,
            forch_pitch_shift = forch_pitch_shift,
        )
        print(f'automatic pitch shift {pitch_shift} semi tones')


    else:
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None


    # Length regulation
    cond, _, codes, commitment_loss, codebook_loss, style_cond = model.length_regulator(S_alt, ylens=target_lengths,
                                                                                       n_quantizers=3,
                                                                                       f0=shifted_f0_alt, style=style2, return_style_residual=True)
    prompt_condition, _, codes, commitment_loss, codebook_loss, style_prompt = model.length_regulator(S_ori,
                                                                                       ylens=target2_lengths,
                                                                                       n_quantizers=3,
                                                                                       f0=F0_ori, style=style2, return_style_residual=True)

    max_source_window = max_context_window - mel2.size(2)
    # split source condition (cond) into chunks
    processed_frames = 0
    generated_wave_chunks = []
    # generate chunk by chunk and stream the output
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        # use_style_residual
        if use_style_residual:
            chunk_style_cond = style_cond[:, processed_frames:processed_frames + max_source_window]
            cat_style_cond = torch.cat([style_prompt, chunk_style_cond], dim=1)
        else:
            cat_style_cond=None
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            # Voice Conversion
            vc_target = model.cfm.inference(cat_condition,
                                                       torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                       mel2, style2, None, diffusion_steps,
                                                       inference_cfg_rate=inference_cfg_rate, style_r=cat_style_cond)

            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()
        vc_wave = vc_wave[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                                    overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()

    time_vc_end = time.time()
    print(f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")
    os.makedirs(exp_path, exist_ok=True)
    src_name = os.path.basename(source).split(".")[0]
    tgt_name = os.path.basename(target_name).split(".")[0]
    if hasattr(args, "uuid"):
        vc_name = f'{src_name}_{tgt_name}_' + args.uuid + '.wav'
    else:
        vc_name = f"{tgt_name}_{src_name}_{pitch_shift}.wav"
    output_path = os.path.join(exp_path, vc_name)
    torchaudio.save(output_path, vc_wave.cpu(), sr)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("--expname", type=str)
    parser.add_argument("--cuda", type=str)
    parser.add_argument("--fp16", type=str)
    parser.add_argument("--accompany", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    args.cuda = torch.device(f"cuda:{args.cuda}")
    args.fp16 = str2bool(args.fp16)
    if args.fp16:
        print('Start fp16 to accelerate inference！')

    args.length_adjust = 1.0
    args.inference_cfg_rate = 0.7
    args.f0_condition = True
    args.semi_tone_shift = None    # If None, the tone is automatically sandhi

    args.output = './outputs'
    os.makedirs(args.output, exist_ok=True)

    models = load_models_api(args, device=args.cuda)
    vc = run_inference(args, models, device=args.cuda)
    if args.accompany:
        vc_t = vc.split('/')
        a,b = '/'.join(vc_t[:-1]), vc_t[-1]
        os.makedirs(a + '/accompany', exist_ok=True)
        op = a + '/accompany/'+b
        echo_then_reverb_save(vc,op,args.accompany)
