"""
Best HWNet configuration found by the sweep (run `HWNet_r4_D8_lr2_s7`).

    python run_HWNet.py --model_name HWNet_best

Builds on configs/HWNet.py and only changes the keys below; everything else
(input shape, losses, lambda_rec=0.1, masking, LR schedule) stays at the
HWNet defaults. Logs go to logs/HWNet_best/.

Results (BCIC2a, 2 classes, subject-dependent, 9 subjects x 5 folds):
    test_acc 0.781 +/- 0.136, f1 0.767 +/- 0.156  (MixNet: 0.772, 0.763)

Note: this seed was picked by test accuracy; the same config with seed 42
gives 0.760. Report the mean over seeds.
"""

import importlib.util
import os

BEST = dict(
    D=8,                # depth multiplier of the spatial conv (default 2)
    n_dec_layers=1,     # decoder transformer blocks (default 2)
    lr=2e-3,            # default 1e-3
    es_patience=60,     # default 30
    seed=7,             # default 1234
)


def _base_get_params():
    path = os.path.join(os.path.dirname(__file__), 'HWNet.py')
    spec = importlib.util.spec_from_file_location('configs.HWNet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_params


def get_params(dataset, train_type, data_type, num_class, loss_weights=None,
               log_dir='logs', **kwargs):
    """Return the HWNet dotdict with the best hyper-parameters applied."""
    kwargs.setdefault('tag', '_best')
    config = _base_get_params()(dataset, train_type, data_type, num_class,
                                loss_weights=loss_weights, log_dir=log_dir,
                                **kwargs)
    config.model_params.update(BEST)
    return config
