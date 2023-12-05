# Generate respiratory samples with audio diffusion model

## Generate Sound Data


### 1. Download ICBHI lung Sound Data
1) To download the ICBHI respiratory sound dataset pre-processed as 4 seconds, use this URL:

https://drive.google.com/file/d/1ULLuqNBacPkBQZqVYhSn5TKkPBYfEDhZ/view?usp=sharing

2) Unzip the file and put it in the ```/dataset/``` directory. 

This dataset should be located in
```
Adversarial-Adaptation-Synthetic-Respiratory-Sound-Data/Diffwave/dataset/
```
it might be like this.
```
Adversarial-Adaptation-Synthetic-Respiratory-Sound-Data/Diffwave/dataset/wav_4secs_16000/label*/*.wav
```
 
### Preprocess 
Note that if you download our pre-processed ICBHI dataset, you can skip this procedure. If not, please follow as below:

This procedure involves pre-processing waveform to Mel-spectrogram for training conditional DiffWave.  
```
python preprocess.py dir [YOUR_SOUND_DATA_DIR]  
# [YOUR_SOUND_DATA_DIR] should be mentioned above  
```

### Run Diffwave training
```
cd Diffwave
sh ./scripts/train_1msteps_icbhi.sh
```
Note that the ```train_1msteps_icbhi.sh``` file is set up for DDP training. 
If you want to train with a single GPU, please change ```CUDA_VISIBLE_DEVICES=0```


### Model
The pretrained weights used in our paper can be downloaded as below:

https://drive.google.com/file/d/1Y2xxZTOmMHCkvEGPDyD2kBkbjQy7YWJb/view?usp=sharing

For inference, this pretrained weight should be located in
```
Adversarial-Adaptation-Synthetic-Respiratory-Sound-Data/Diffwave/save/1msteps_icbhi_Diffwave/
```

### Inference


If you want to use a folder containing Mel-spectrograms to generate respiratory sound samples, do

```
python inference.py [YOUR_MODEL_PATH] --spectrogram_path [Spectrograms_PATH]
# [YOUR_MODEL_PATH] directory should contain train_args.json
# [Spectrograms_PATH] directory should contain the preprocessed spectorgrams you want to synthesize 
```

Or, if you have followed our guidelines (moved to weights.pt into ./save/1msteps_icbhi_Diffwave/), you can use our script files.

If you want to generate the samples in a specific folder (here, ./samples/test_set_samples/), do
```
sh ./scripts/eval_1msteps_for_generate_direct.sh
```

You can obtain the generated 10 samples of the ICBHI test set (event level) in the ``` ./samples/generated_test_samples/save/1msteps_icbhi_Diffwave_seed=0/```

If you want to generate 2,000 samples from label3's test data, do
```
sh ./scripts/eval_1msteps_for_generate_label3.sh
```

If you want to generate 10,000 samples from label2's test data, you can modify the ```iter_for_generate=10000``` of the  ```./scripts/eval_1msteps_for_generate_label2.sh``` file.

In script files, you can modify the hyperparameters.

```iter_for_generate```: how many samples do you want to generate





### References
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf)
