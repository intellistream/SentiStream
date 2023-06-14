from dataclasses import dataclass


@dataclass
class Word2VecConfig:
    sg: int = 0
    hs: int = 0
    cbow_mean: int = 1
    window: int = 5
    seed: int = 1
    sample: float = 1e-3
    shrink_windows: bool = True
    negative: int = 5
    vector_size: int = 20
    epochs: int = 5
    alpha: float = 0.025
    min_alpha: float = 0.0001
    min_count: int = 5
    confidence: float = 0.5
    ttd: bool = True
    batch_size: int = 100

    def __post_init__(self):
        self.ns_exponent = 0.75
        self.workers = 1
        self.layer1_size = self.vector_size
        self.compute_loss = False
        self.domain = 2**31-1
        self.running_training_loss = 0.0
        self.ref_pos = [
            'love', 'best', 'beautiful', 'great',
            'cool', 'awesome', 'wonderful',
            'brilliant', 'excellent', 'fantastic']
        self.ref_neg = [
            'bad', 'worst', 'stupid', 'disappointing',
            'terrible', 'rubbish', 'boring', 'awful',
            'unwatchable', 'awkward']
        self.batching_flag = 'BATCHING'
