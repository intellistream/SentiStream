## TODO

- [x] Semi-Supervised Models
    - [x] ANN module
    - [x] HAN module
- [ ] Unsupervised Models
    - [ ] PLStream module (updated one)
- [ ] SentiStream Module
    - [ ] Run with n without Flink
    - [ ] Compatible with all datasets
- [ ] Evaluation Module
    - [ ] Compatible with all ANN models (Last layer SoftMax or Sigmoid)
    - [ ] Compatible with all word vector algorithms (W2V, FastText, Sense2Vec)
- [ ] Batch Inference Module
- [ ] Classifier Module
- [ ] Supervised Module
    - [x] Combine Initial Train Module
- [ ] Utils Module
    - [ ] Make all utility functions suitable for batch processing and stream processing
    - [ ] Define most accurate preprocessing strategy for all works.
- [ ] Requirements (Install only required ones)



## CHECK WORD EMBEDDINGS FOR INCREMENTAL TRAIINNG
## CONTINOUS TRAIN
## DONT SINK PSEUDO LABELS...
## SHOULD I PUT ALL DATA TO GPU INITIALLY? - to reduce communication overhead in each batch since lost of VRAM is idle.