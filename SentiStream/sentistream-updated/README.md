## TODO

- [x] Semi-Supervised Models
    - [x] ANN module
    - [x] HAN module
- [ ] Unsupervised Models
    - [x] PLStream module (updated one)
- [ ] SentiStream Module
    - [ ] Run with n without Flink
    - [ ] Compatible with all datasets
- [ ] Evaluation Module
    - [ ] Compatible with all ANN models (Last layer SoftMax or Sigmoid)
    - [ ] Compatible with all word vector algorithms (W2V, FastText, Sense2Vec)
- [ ] Batch Inference Module
- [x] Classifier Module
- [ ] Supervised Module
    - [x] Combine Initial Train Module
- [ ] Utils Module
    - [x] Make all utility functions suitable for batch processing and stream processing
    - [x] Define most accurate preprocessing strategy for all works.
- [ ] Requirements (Install only required ones)



## CHECK WORD EMBEDDINGS FOR INCREMENTAL TRAIINNG
## CONTINOUS TRAIN
## DONT SINK PSEUDO LABELS...
## SHOULD I PUT ALL DATA TO GPU INITIALLY? - to reduce communication overhead in each batch since lost of VRAM is idle.
## CHECK OUTPUT OF STEMMER WHEN USING REF TABLE......
## PLSTREAM WORD VECOTR AVG IS WRONG ONE
## CONVERRT REF TABLE TO SET 

## WHAT ABT MAX SENT, WORD LEN???

## CHECK FOR LAST BATCH IN STREAM

# TO REM

## Plstream neg <-> pos count -- wrong one



# UPGRADE NUMPY AFTER INSTALLING REQUIREMENTS