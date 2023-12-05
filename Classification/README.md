# Adversarial Fine-tuning using Generated Respiratory Sound to Address Class Imbalance
Training Respiratory Sound Classification Model with Adversarial FT


### Environments
`Ubuntu xx.xx`  
`Python 3.8.xx`


## Datasets

### Generated Training Samples

We set the dataloader as event-level loader. (i.e., each waveform sample has various events).
Here, following link includes original ICBHI dataset, Mixed-500, Mixed-2k:


However, you can move data via download from [ICBHI dataset](https://paperswithcode.com/dataset/icbhi-respiratory-sound-database), and generated from our pretrained Diffwave.

unzip the ```icbhi_dataset.zip``` into `./data/`.

For the real training samples, move into



<pre>
data/training
├── real
│    ├── (4,142 samples)
│    ├── ...
│    ├── ...
│    └── 226_1b1_Pl_sc_LittC2SE_event_9_label_1.wav
│
├── generated_from_1msteps
│   ├── mixed500
│    |   ├── class0
│    |    |    └── (500 samples of class0)
│    |   ├── class1
│    |    |    └── (500 samples of class1)
│    |   ├── class2
│    |    |    └── (500 samples of class2)
│    |   ├── class3
│    |    |    └── (500 samples of class3)
│   ├── mixed2k
│    |   ├── class0
│    |    |    └── (2,000 samples of class0)
│    |   ├── class1
│    |    |    └── (2,000 samples of class1)
│    |   ├── class2
│    |    |    └── (2,000 samples of class2)
│    |   ├── class3
│    |    |    └── (2,000 samples of class3)
</pre>

<pre>
data/test
├──test sample1
├──test sample2
├──...
├──test sample2,756
</pre>

## Run

### AST FT (No aug.)
```
$ ./scripts/icbhi_ce.sh
```

### AST FT (Mixed500)
```
$ ./scripts/icbhi_ce_1msteps_500.sh
```

### AST FT (Mixed2k)
```
$ ./scripts/icbhi_ce_1msteps_2k.sh
```

### Aderversarial FT (Mixed500)
```
$ ./scripts/icbhi_aft_1msteps_500_mixed.sh
```

## Evaluation

### Pretrained AFT weights
We will provide the AFT pretrained on Mixed500, which has the performance of around 62 Score.

https://drive.google.com/file/d/15Vfy9RAaAZZTiOEI1mn92qcqE-wn7mm2/view?usp=sharing

