# Adversarial Fine-tuning using Generated Respiratory Sound to Address Class Imbalance
Training Respiratory Sound Classification Model with Adversarial FT


### Environments
`Ubuntu xx.xx`  
`Python 3.8.xx`


## Datasets

### Generated Training Samples (Shared will be the final version)

We set the dataloader as event-level loader

Owing to anonymity, we will provide the original ICBHI dataset, Mixed-500, Mixed-2k in the final version

However, you can move data via download from ICBHI dataset, and generated from our pretrained Diffwave.

For the real training samples, move into
```
./data/real/
```

For the Mixed500, move into
```
./data/generated_from_1msteps/mixed500/
```

For the Mixed2k, move into
```
./data/generated_from_1msteps/mixed2k/
```


### Test samples
Move all the test sets (event-level) into 
```
./data/test/real/
```

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
We will provide the AFT pretrained on Mixed500, which has the performance of around 62 Score in the final version