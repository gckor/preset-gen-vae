"""
Allows easy modification of all configuration parameters required to perform a series of models evaluations.
This script is not intended to be run, it only describes parameters.
"""


import datetime
from utils.config import EvalConfig


eval = EvalConfig()  # (shadows unused built-in name)
eval.start_datetime = datetime.datetime.now().isoformat()

# Names must be include experiment folder and run name (_kf suffix must be omitted is all_k_folds is True)
eval.models_names = [  # - - - 30k samples full dataset ('b' suffix means 'big') - - -
                     'FlFl/kfold0-s2',
                     ]
eval.dataset = 'test'  # Do not use 'test' dataset during models development
eval.override_previous_eval = False  # If True, all models be re-evaluated (might be very long)

eval.minibatch_size = 1  # Reduced mini-batch size not to reserve too much GPU RAM. 1 <=> per-preset metrics
eval.device = 'cuda'
# Don't use too many cores, numpy uses multi-threaded MKL (in each process)
eval.multiprocess_cores_ratio = 0.1  # ratio of CPU cores to be used (if 1.0: use all os.cpu_count() cores)
eval.verbosity = 2

eval.sampling_rate = 22050 # Same value with the trained model config
eval.logs_root_dir = '/exp_logs/preset-gen-vae'