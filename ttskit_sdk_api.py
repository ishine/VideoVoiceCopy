# author: kuangdd
# date: 2021/4/23
"""
### sdk_api
语音合成SDK接口。
本地函数式地调用语音合成。

+ 简单使用
```python
from ttskit import sdk_api

wav = sdk_api.tts_sdk('文本', audio='1')
```
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import os
import argparse
import json
import tempfile
import base64
import numpy as np
import torch
import aukit
import tqdm
import requests

from waveglow import inference as waveglow
from mellotron import inference as mellotron
from mellotron.layers import TacotronSTFT
from mellotron.hparams import create_hparams

_home_dir = os.path.dirname(os.path.abspath(__file__))

# 用griffinlim声码器
_hparams = create_hparams()
_stft = TacotronSTFT(
    _hparams.filter_length, _hparams.hop_length, _hparams.win_length,
    _hparams.n_mel_channels, _hparams.sampling_rate, _hparams.mel_fmin,
    _hparams.mel_fmax)

_use_waveglow = 0
_device = 'cuda' if torch.cuda.is_available() else 'cpu'

_dataloader = None
_reference_audio_list = []

def load_models(resource_path=_home_dir, **kwargs):
    """加载模型，如果使用默认参数，则判断文件是否已经下载，如果未下载，则先下载文件。"""
    global _use_waveglow
    global _dataloader
    global _reference_audio_list
    global _reference_audio_dict

    mellotron_path = os.path.join(resource_path, 'resource', 'model', 'mellotron.kuangdd-rtvc.pt')
    waveglow_path = os.path.join(resource_path, 'resource', 'model', 'waveglow.kuangdd.pt')
    ge2e_path = os.path.join(resource_path, 'resource', 'model', 'ge2e.kuangdd.pt')
    mellotron_hparams_path = os.path.join(resource_path, 'resource', 'model', 'mellotron_hparams.json')
    _reference_audio_tar_path = os.path.join(resource_path, 'resource', 'reference_audio.tar')
    _audio_tar_path = os.path.join(resource_path, 'resource', 'audio.tar')

    tmp = os.path.splitext(_audio_tar_path)[0]
    if os.path.isdir(tmp):
        tmp = list(sorted([*Path(tmp).glob('*.wav'), *Path(tmp).glob('*.mp3')]))
        _reference_audio_list.extend(tmp)

    tmp = os.path.splitext(_reference_audio_tar_path)[0]
    if os.path.isdir(tmp):
        tmp = list(sorted([*Path(tmp).glob('*.wav'), *Path(tmp).glob('*.mp3')]))
        _reference_audio_list.extend(tmp)

    _reference_audio_list = [w.__str__() for w in _reference_audio_list]
    _reference_audio_dict = {os.path.basename(w).split('-')[1]: w for w in _reference_audio_list}

    if _dataloader is not None:
        return
    if waveglow_path and waveglow_path not in {'_', 'gf', 'griffinlim'}:
        waveglow.load_waveglow_torch(waveglow_path)
        _use_waveglow = 1

    if mellotron_path:
        mellotron.load_mellotron_torch(mellotron_path)

    mellotron_hparams = mellotron.create_hparams(open(mellotron_hparams_path, encoding='utf8').read())
    mellotron_hparams.encoder_model_fpath = ge2e_path
    _dataloader = mellotron.TextMelLoader(audiopaths_and_text='',
                                          hparams=mellotron_hparams,
                                          speaker_ids=None,
                                          mode='test')
    return _dataloader


def transform_mellotron_input_data(dataloader, text, speaker='', audio='', device=''):
    """输入数据转换为模型输入的数据格式。"""
    if not device:
        device = _device

    text_data, mel_data, speaker_data, f0_data = dataloader.get_data_train_v2([audio, text, speaker])
    text_data = text_data[None, :].long().to(device)
    style_data = 0
    speaker_data = speaker_data.to(device)
    f0_data = f0_data
    mel_data = mel_data[None].to(device)

    return text_data, style_data, speaker_data, f0_data, mel_data


def tts_sdk(text, speaker='biaobei', audio='0', output='',resource_path=_home_dir,  **kwargs):
    """语音合成函数式SDK接口。 
    text为待合成的文本。
    speaker可设置为内置的发音人名称，可选名称见_reference_audio_dict；默认的发音人名称列表见resource/reference_audio/__init__.py。
    audio如果是数字，则调用内置的语音作为发音人参考音频；如果是语音路径，则调用audio路径的语音作为发音人参考音频。
    output如果以.wav结尾，则为保存语音文件的路径；如果以play开头，则合成语音后自动播放语音。
    """
    global _dataloader
    if _dataloader is None:
        load_models(resource_path, **kwargs)

    if str(audio).isdigit():
        audio = _reference_audio_list[(int(audio) - 1) % len(_reference_audio_list)]
    elif os.path.isfile(audio):
        audio = str(audio)
    elif isinstance(audio, bytes):
        tmp_audio = tempfile.TemporaryFile(suffix='.wav')
        tmp_audio.write(audio)
        audio = tmp_audio.name
    elif isinstance(audio, str) and len(audio) >= 256:
        tmp_audio = tempfile.TemporaryFile(suffix='.wav')
        tmp_audio.write(base64.standard_b64decode(audio))
        audio = tmp_audio.name
    elif speaker in _reference_audio_dict:
        audio = _reference_audio_dict[speaker]
    else:
        raise AssertionError
    text_data, style_data, speaker_data, f0_data, mel_data = transform_mellotron_input_data(
        dataloader=_dataloader, text=text, speaker=speaker, audio=audio, device=_device)

    mels, mels_postnet, gates, alignments = mellotron.generate_mel(text_data, style_data, speaker_data, f0_data)

    out_gate = gates.cpu().numpy()[0]
    end_idx = np.argmax(out_gate > kwargs.get('gate_threshold', 0.2)) or np.argmax(out_gate) or out_gate.shape[0]

    mels_postnet = mels_postnet[:, :, :end_idx]
    if _use_waveglow:
        wavs = waveglow.generate_wave(mel=mels_postnet, **kwargs)
    else:
        wavs = _stft.griffin_lim(mels_postnet, n_iters=5)

    wav_output = wavs.squeeze(0).cpu().numpy()

    if output.startswith('play'):
        aukit.play_sound(wav_output, sr=_stft.sampling_rate)
    if output.endswith('.wav'):
        aukit.save_wav(wav_output, output, sr=_stft.sampling_rate)
    wav_output = aukit.anything2bytes(wav_output, sr=_stft.sampling_rate)
    return wav_output
