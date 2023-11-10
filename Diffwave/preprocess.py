# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import os
from glob import glob
from tqdm import tqdm


parser = ArgumentParser(description='prepares a dataset to train DiffWave')
parser.add_argument('dir',
    help='directory containing .wav files for training')
parser.add_argument('--sr', type=int, default=16000,
    help='sample rate')
parser.add_argument('--hop', type=int, default=256,
    help='number of overlap')
parser.add_argument('--nfft', type=int, default=1024,
    help='number of fft')
parser.add_argument('--n_mels', type=int, default=80,
    help='number of mel bins')
args = parser.parse_args()



def transform(filename):
    audio, sr = T.load(filename)
    audio = torch.clamp(audio[0], -1.0, 1.0)
  
    if args.sr != sr:
      raise ValueError(f'Invalid sample rate {sr}.')
    mel_args = {
        'sample_rate': sr,
        'win_length': args.hop * 4,
        'hop_length': args.hop,
        'n_fft': args.nfft,
        'f_min': 20.0,
        'f_max': sr / 2.0,
        'n_mels': args.n_mels,
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args)
  
    with torch.no_grad():
      spectrogram = mel_spec_transform(audio)
      spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
      spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)

      np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())
      


def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))

if __name__ == '__main__':
    main(args)