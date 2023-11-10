from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio
import torchaudio


def get_meta_infor(stetho, real_or_gen, args):
    
    if args.meta_mode == 'none':
        meta_label = -1
    
    
    elif args.meta_mode == 'mixed':
        if real_or_gen == 'real':
            meta_label = 0
        else:
            meta_label = 1
    
                   
    return meta_label


class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, real, print_flag=True):
        train_data_folder = os.path.join(args.data_folder, 'training', 'real') if real else os.path.join(args.data_folder, 'training', args.real_gen_dir)
        test_data_folder = os.path.join(args.data_folder, 'test', 'real')
        self.train_flag = train_flag
        
        if self.train_flag:
            self.data_folder = train_data_folder
        else:
            self.data_folder = test_data_folder
        self.transform = transform
        self.args = args
        
        # parameters for spectrograms
        self.sample_rate = self.args.sample_rate
        self.n_mels = self.args.n_mels        
        
        self.data_glob = sorted(glob(self.data_folder+'/*/*.wav')) if not real else sorted(glob(self.data_folder+'/*.wav'))
        
        print('Total length of dataset is', len(self.data_glob))
                                    
        # ==========================================================================
        """ convert fbank """
        self.audio_images = []
        for index in self.data_glob: #for the training set, 4142
            _, file_id = os.path.split(index)
            
            file_id_tmp = file_id.split('.wav')[0]
            stetho = file_id_tmp.split('_')[4]
            if 'index' in file_id_tmp:
                label = file_id_tmp.split('_')[-2]
                real_or_gen = 'gen'
                meta_label = get_meta_infor(stetho, real_or_gen, self.args)
            else:
                label = file_id_tmp.split('_')[-1]
                real_or_gen = 'real'
            meta_label = get_meta_infor(stetho, real_or_gen, self.args)
            audio, sr = torchaudio.load(index)
            
            audio_image = []
            for aug_idx in range(self.args.raw_augment+1):
                if aug_idx > 0:
                    if self.train_flag:
                        audio = augment_raw_audio(np.asarray(audio.squeeze(0)), self.sample_rate, self.args)
                        audio = cut_pad_sample_torchaudio(torch.tensor(audio), self.args)                
                    
                    image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels)
                    audio_image.append(image)
                else:
                    image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels) 
                    audio_image.append(image)
            self.audio_images.append((audio_image, int(label))) if self.args.meta_mode == 'none' else self.audio_images.append((audio_image, int(label), meta_label))
            
            
        '''
        self.class_nums = np.zeros(args.n_cls)
        self.class_nums[sample[1]] += 1
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        if print_flag:
            print('total number of audio data: {}'.format(len(self.data_glob)))
            print('*' * 25)
            print('For the Label Distribution')
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))
        '''
        # ==========================================================================

    def __getitem__(self, index):
        if self.args.meta_mode == 'none':
            audio_images, label = self.audio_images[index][0], self.audio_images[index][1]
        else:
            audio_images, label, meta_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2]

        if self.args.raw_augment and self.train_flag:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        
        if self.train_flag:
            if self.args.meta_mode == 'none':
                return audio_image, torch.tensor(label)
            else:
                return audio_image, (torch.tensor(label), torch.tensor(meta_label))
        else:
            return audio_image, torch.tensor(label)

    def __len__(self):
        return len(self.data_glob)