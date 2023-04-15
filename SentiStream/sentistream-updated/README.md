## TODO

- [x] Semi-Supervised Models
    - [x] ANN module
    - [x] HAN module
- [x] Unsupervised Models
    - [x] PLStream module (updated one)
- [ ] SentiStream Module
    - [ ] Run with n without Flink
    - [ ] Compatible with all datasets (Not now)
- [x] Evaluation Module
    - [x] Compatible with all ANN models (Last layer SoftMax or Sigmoid)
    - [x] Compatible with all word vector algorithms (W2V, FastText, Sense2Vec)
- [x] Batch Inference Module
- [x] Classifier Module
- [x] Supervised Module
    - [x] Combine Initial Train Module
- [ ] Utils Module
    - [x] Make all utility functions suitable for batch processing and stream processing
    - [x] Define most accurate preprocessing strategy for all works.
- [x] Requirements (Install only required ones)
- [ ] Create new README



## CHECK WORD EMBEDDINGS FOR INCREMENTAL TRAIINNG
## CONTINOUS TRAIN
## DONT SINK PSEUDO LABELS...
## SHOULD I PUT ALL DATA TO GPU INITIALLY? - to reduce communication overhead in each batch since lost of VRAM is idle.
## CHECK OUTPUT OF STEMMER WHEN USING REF TABLE......
## PLSTREAM WORD VECOTR AVG IS WRONG ONE

### PLSTREAM REF WORD NNED TO STEM

## VECTORIZE IF POSSIBLE - PLSTREAM COSINE SIM

## WHY DO WE HAVE BATCH INF?, EVEN IF 90% ACC, IT MAY HAVE SOME NEW VOCABS TO LEARN???
## CONVERRT REF TABLE TO SET 

## WHAT ABT MAX SENT, WORD LEN???

## CHECK FOR LAST BATCH IN STREAM

# TO REM

## Plstream neg <-> pos count -- wrong one


### PSEUDO_LABEL -> THRESHOLDDDDS


# UPGRADE NUMPY AFTER INSTALLING REQUIREMENTS