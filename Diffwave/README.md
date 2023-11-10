# Generate respiratory samples with audio diffusion model

## Generate Sound Data


### 1. Download ICBHI lung Sound Data
(Note: At the end of the anonymity rules term, we will provide an organized and pre-processed dataset of our progress.)  

Download the ICBHI respiratory sound dataset  
unzip the file and put it in the certain directory. 

Sound Data should be located in
```
Adversarial-Adaptation-Synthetic-Respiratory-Sound-Data/Diffwave/dataset/
```
it might be like this.
```
Adversarial-Adaptation-Synthetic-Respiratory-Sound-Data/Diffwave/dataset/[train_or_test]/[wav_files]
```
 
### Preprocess 
Pre-process waveform to Mel-spectrogram.  
```
python preprocess.py dir [YOUR_SOUND_DATA_DIR]  
# [YOUR_SOUND_DATA_DIR] should be mentioned above  
```

### Run Diffwave training
```
cd Diffwave
sh ./scripts/train_1msteps_icbhi.sh
```

### Model
(Note: Due to anonymity rules, we cannot provide the 1M step model at this time.
But we will provide it when the term is over.)  

Model should be located in
```
Adversarial-Adaptation-Synthetic-Respiratory-Sound-Data/Diffwave/save/1msteps_icbhi_Diffwave/
```
for inference.

### Inference


If you want to generate some of the Mel-spectrograms in the folder using command lines, do

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

If you want to generate 2,000 samples from label3's training data, do the following
```
sh ./scripts/eval_1msteps_for_generate_label3.sh
```

If you want to generate 10,000 samples from label2's training data, you can modify the ```iter_for_generate=10000``` of the  ```./scripts/eval_1msteps_for_generate_label2.sh``` file.

In script files, you can modify the hyperparameters.

```iter_for_generate```: how many samples do you want to generate





### References
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf)
