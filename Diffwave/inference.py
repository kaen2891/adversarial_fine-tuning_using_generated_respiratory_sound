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
import os
import torch
import torchaudio
from glob import glob
from argparse import ArgumentParser
import json

#from params import AttrDict, params as base_params
from model import DiffWave


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
  
    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


def get_params(args):
    return AttrDict(
        # Data params
        sample_rate=args['sample_rate'],
        n_mels=args['n_mels'],
        n_fft=args['n_fft'],
        hop_samples=args['hop_samples'],
        crop_mel_frames=args['crop_mel_frames'],
        # Model params
        residual_layers=30,
        residual_channels=64,
        dilation_cycle_length=10,
        unconditional = args['uncondition'],
        class_condition = args['class_condition'],
        noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
        inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
    
        # unconditional sample len
        audio_len = int(args['sample_rate'] * 5), # unconditional_synthesis_samples
        # audio_len = 16000*5, # unconditional_synthesis_samples
    )


models = {}

def predict(spectrogram=None, model_dir=None, params=None, attrdict=None, class_condition=None, device=torch.device('cuda'), fast_sampling=False):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    #model = DiffWave(AttrDict(params)).to(device)
    model = DiffWave(attrdict(params)).to(device)    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model

  model = models[model_dir]
  model.params.override(params)
  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)


    if not model.params.unconditional:
      if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
        spectrogram = spectrogram.unsqueeze(0)
      spectrogram = spectrogram.to(device)
      audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    else:
      audio = torch.randn(1, params.audio_len, device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      #audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
      if model.params.class_condition:
        audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram, class_condition.to(device)).squeeze(1))
      else:
        audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
      
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, model.params.sample_rate


import random
import torch.backends.cudnn as cudnn

def main(args):
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
    
    ckpt_dir, ckpt_name = os.path.split(args.model_dir)
    _ckpt_dir, model_name = os.path.split(ckpt_dir)
    ckpt_name_tmp = ckpt_name.split('.')[0]+'_seed={}'.format(args.seed)
    _, exp_name = os.path.split(args.model_dir)
    json_dir = os.path.join(ckpt_dir, ckpt_name, 'train_args.json')
    with open(json_dir, "r") as json_infor:
        params_infor = json.load(json_infor)
    
    base_params = get_params(params_infor)
    output_dir_for_ckpt = os.path.join(args.output, model_name, ckpt_name_tmp)
    if not os.path.isdir(output_dir_for_ckpt):
        os.makedirs(output_dir_for_ckpt, exist_ok=True)
    
    
    if args.spectrogram_path:
        spectrogram_list = sorted(glob(args.spectrogram_path+'/*.npy'))
        if args.iter_for_generate > 1:
            for idx in range(1, args.iter_for_generate + 1):
                chosen_index = random.randrange(len(spectrogram_list))
                chosen_spectrogram = spectrogram_list[chosen_index]
                _, spec_id = os.path.split(chosen_spectrogram)
                spec_id_tmp = spec_id.split('.')[0]
                
                
                '''
                # for visualization
                plot_spectrogram_to_numpy(spectrogram.numpy(), '{}'.format(idx), 'Original Spectrogram')
                audio, sr = predict(spectrogram, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params, attrdict=AttrDict, class_condition=torch.tensor(args.class_condition))
                generated_spectrogram = mel_spec_transform(audio.cpu())
                generated_spectrogram = 20 * torch.log10(torch.clamp(generated_spectrogram, min=1e-5)) - 20
                generated_spectrogram = torch.clamp((generated_spectrogram + 100) / 100, 0.0, 1.0)
                plot_spectrogram_to_numpy(generated_spectrogram.squeeze(0).numpy(), '{}+_index={}'.format(spec_id_tmp, idx), 'Generated audio with no augmentation')
                torchaudio.save(os.path.join(output_dir_for_ckpt, spec_id_tmp+'_index={}.wav'.format(idx)), audio.cpu(), sample_rate=sr)
                '''                
                
                spectrogram = torch.from_numpy(np.load(chosen_spectrogram))
                aug_audio, sr = predict(spectrogram, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params, attrdict=AttrDict, class_condition=torch.tensor(args.class_condition)) # note that class_condition is not working
                torchaudio.save(os.path.join(output_dir_for_ckpt, spec_id_tmp+'_index={}.wav'.format(idx)), aug_audio.cpu(), sample_rate=sr)
                
                print('{}-index {} saved at {}'.format(idx, spec_id_tmp, os.path.join(output_dir_for_ckpt, spec_id_tmp+'_index={}.wav'.format(idx))))
        else:
            for chosen_spectrogram in spectrogram_list:
                spectrogram = torch.from_numpy(np.load(chosen_spectrogram))
                _, spec_id = os.path.split(chosen_spectrogram)
                spec_id_tmp = spec_id.split('.')[0]
                audio, sr = predict(spectrogram, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params, attrdict=AttrDict, class_condition=torch.tensor(args.class_condition))
                torchaudio.save(os.path.join(output_dir_for_ckpt, spec_id_tmp+'.wav'), audio.cpu(), sample_rate=sr)
                print('spec_id_tmp {} done'.format(spec_id_tmp))

    
    
    


if __name__ == '__main__':    
    parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
    parser.add_argument('model_dir', help='directory containing a trained model (or full path to weights.pt file)')
    parser.add_argument('--spectrogram_path', '-s', help='path to a spectrogram file generated by diffwave.preprocess')
    parser.add_argument('--output', '-o', default='output.wav', help='output file name')
    parser.add_argument('--fast', '-f', action='store_true', help='fast sampling procedure')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iter_for_generate', type=int, default=0)
    parser.add_argument('--class_condition', type=int, default=0)
    main(parser.parse_args())
