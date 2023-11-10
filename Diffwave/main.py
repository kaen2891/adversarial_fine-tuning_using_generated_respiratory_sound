
import os
import json
import argparse
import torch.backends.cudnn as cudnn
import torch

from torch.cuda import device_count
from torch.multiprocessing import spawn
torch.multiprocessing.set_sharing_strategy('file_system')

from learner import train, train_distributed

import random
import numpy as np

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

def parse_args():
    parser = argparse.ArgumentParser('argument for training DiffWave model')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./save', help='directory in which to store model checkpoints and training logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    

    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--tag', type=str, default='gpu2')
    parser.add_argument('--data_dirs', nargs='+', default='./data/', help='space separated list of directories from which to read .wav files for training')
    parser.add_argument('--max_steps', type=int, default=None, help='maximum number of training steps')
    parser.add_argument('--uncondition', action='store_true', default=False, help='condition or uncondition')
    parser.add_argument('--class_condition', action='store_true', default=False, help='class condition')
    parser.add_argument('--max_step', type=int, default=50, help='diffusion steps')
    parser.add_argument('--fp16', action='store_true', default=False, help='space separated list of directories from which to read .wav files for training')
    
    parser.add_argument('--label', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_grad_norm', action='store_true', default=False)
    parser.add_argument('--sample_rate', type=int,  default=16000, help='sampling rate when load audio data')
    parser.add_argument('--desired_length', type=int, default=5, help='wav length')
    parser.add_argument('--n_mels', type=int, default=80, help='the number of mel filter banks')
    parser.add_argument('--n_fft', type=int, default=1024, help='the number of fft coefficients')
    parser.add_argument('--hop_samples', type=int, default=256, help='the number of hop length (overlap)')
    parser.add_argument('--crop_mel_frames', type=int, default=62, help='the number of cropping mel frames') # Probably an error in paper.
    parser.add_argument('--num_workers', type=int, default=16)
    
    
    args = parser.parse_args()
    

    condition_check = 'uncondition' if args.uncondition else 'condition'
    args.model_name = '{}_{}_bs_{}_sr_{}_len_{}_label_{}_seed_{}_cc_{}_step_{}_tag_{}'.format(condition_check, args.dataset, args.batch_size, args.sample_rate, args.desired_length, args.label, args.seed, args.class_condition, args.max_step, args.tag)
    print(args.model_name)
    
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    
    params = AttrDict(
        # Training params
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
    
        # Data params
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_samples=args.hop_samples,
        crop_mel_frames=args.crop_mel_frames,  # Probably an error in paper.
    
        # Model params
        residual_layers=30,
        residual_channels=64,
        dilation_cycle_length=10,
        unconditional = args.uncondition,
        class_condition = args.class_condition,
        #noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
        noise_schedule=np.linspace(1e-4, 0.05, args.max_step).tolist(),
        inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
    
        # unconditional sample len
        audio_len = int(args.sample_rate * args.desired_length), # unconditional_synthesis_samples
        # audio_len = 16000*5, # unconditional_synthesis_samples
    )

    
    return args, params

def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]




def main():
    args, params = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    replica_count = device_count()
    if replica_count > 1:
        print('DDP Training')
        if args.batch_size % replica_count != 0:
            raise ValueError(f'Batch size {args.batch_size} is not evenly divisble by # GPUs {replica_count}.')
        args.batch_size = args.batch_size // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
    else:
        print('Not DDP Training')
        train(args, params)



if __name__ == '__main__':
    main()
