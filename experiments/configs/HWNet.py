"""
=============================================================================
 EXPERIMENT CONFIG FOR HWNet  (BCIC2a, subject-dependent)
=============================================================================

`experiments/run_HWNet.py` loads this file through
`configs/exp_config.get_params()` and uses the returned dotdict to build the
model, the DataLoader and the log directory.

You will come back to this file constantly -- it is where you tune the
training model without touching model code, because every key of
`model_params` is forwarded to `HWNet.__init__` and re-applied as
`self.<key>` at the end of `HWNet._config()`.

-----------------------------------------------------------------------------
 THE ONE THING THAT BREAKS FIRST: SHAPES
-----------------------------------------------------------------------------
mixnet.utils.DataLoader is the class that loads your data and passes it 
to your model. 

Every array your Part 1 pipeline saved is passed through
`mixnet.utils.DataLoader._change_data_format()` on its way to your model --
`load_train_set()`, `load_val_set()` and `load_test_set()` all call it. It is
about twenty lines; open it and read it alongside this section.

The letters are axis names: N = trials, C = channels, T = time, D = depth
(the singleton axis a Conv2D needs), S = sub-bands/freqs, H/W = height/width
of a 2-D spatial map.

The below table shows the five valid values for `data_format` and what they do to 
the shape of your data. The result is what your model will see, so make sure it 
matches the input shape you defined in `HWNet.__init__()`.

  data_format | what the method does                      | result
  ------------+-------------------------------------------+--------------
  'NCTD'      | reshape(n, C, T, 1)                       | (n, C, T, 1)
  'NDCT'      | reshape(n, 1, C, T)                       | (n, 1, C, T)
  'NTCD'      | reshape(n, C, T, 1), then swapaxes(1, 3)  | (n, 1, T, C)
  'NSHWD'     | reshape(n, S, H, W, 1)                    | (n, S, H, W, 1)
   None       | nothing at all                            | (n, C, T)

Anything other than these five values raises:
    Value Error: data_format requires None, 'NCTD', 'NDCT', 'NTCD' or 'NSHWD'
    
Note: You can add your custom data_format to the list above, but you must also implement it in
`mixnet.utils.DataLoader._change_data_format()`
-----------------------------------------------------------------------------
"""

import json
import os

from tensorflow.keras.optimizers import Adam

from mixnet.loss import SparseCategoricalCrossentropy
from mixnet.models import HWNet
from mixnet.models.HWNet import MaskedMSE
from mixnet.utils import dotdict

# Ablations for H1 (masked autoencoding vs. masking as augmentation) and H2
# (attention depth). Documentation only: switch by editing values + `tag`, or
# set HWNET_ABLATION=<key> in the environment (applies the entry and logs
# under logs/HWNet_<key>/). `lambda_rec` becomes loss_weights[0]; every other
# key goes into model_params.
# Choose a final configuration from VALIDATION classification only
# (val_crossentropy_loss / val_acc in S*_fold*_out_log.csv), never test_acc.
ABLATIONS = {
    # H1: does the masked-autoencoding task help, beyond masking as augmentation?
    'clf_only':  dict(p_band=0.0, n_tmask=0, lambda_rec=0.0),  # plain classifier, no mask, no recon
    'aug_only':  dict(lambda_rec=0.0),                         # masking as augmentation, no recon
    'ae_nomask': dict(p_band=0.0, n_tmask=0),                  # aux plain autoencoder (full-input MSE)
    'mae':       dict(),                                       # default: masked autoencoding + clf
    # which mask matters for reconstruction
    'mae_band':  dict(n_tmask=0),
    'mae_time':  dict(p_band=0.0),
    # reconstruction weight (kept small; choose by validation classification only)
    'lam0.03': dict(lambda_rec=0.03), 'lam0.3': dict(lambda_rec=0.3),
    # H2: attention depth
    'L0': dict(n_layers=0), 'L1': dict(n_layers=1), 'L3': dict(n_layers=3),
}


def get_params(dataset, train_type, data_type, num_class, loss_weights=None,
               log_dir='logs', **kwargs):
    """Return a dotdict of parameters for `run_HWNet.py`."""
    model_name = 'HWNet'
    tag = kwargs.get('tag', '')
    lambda_rec = kwargs.get('lambda_rec', 0.1)
    overrides = {}
    ablation = os.environ.get('HWNET_ABLATION', '')
    if ablation:
        if ablation not in ABLATIONS:
            raise ValueError('HWNET_ABLATION={} not in {}'.format(
                ablation, list(ABLATIONS)))
        overrides = dict(ABLATIONS[ablation])
        lambda_rec = overrides.pop('lambda_rec', lambda_rec)
        tag = '_' + ablation
        print('HWNet ablation {!r}: {}, lambda_rec={}'.format(
            ablation, overrides, lambda_rec))
    # Hyper-parameter search hook: HWNET_OVERRIDES='{"lr": 3e-3, ...}' (JSON)
    # overrides model_params (and lambda_rec); HWNET_TAG sets the log suffix.
    # Configurations must be compared on validation metrics only.
    extra = json.loads(os.environ.get('HWNET_OVERRIDES', '') or '{}')
    if extra:
        if 'input_shape' in extra:
            extra['input_shape'] = tuple(extra['input_shape'])
        lambda_rec = extra.pop('lambda_rec', lambda_rec)
        overrides = dict(overrides, **extra)
        print('HWNet overrides: {}, lambda_rec={}'.format(extra, lambda_rec))
    tag = os.environ.get('HWNET_TAG', tag)

    # [recon, crossentropy]: classification keeps weight 1.0, recon is a
    # small auxiliary regulariser.
    loss_weights = [lambda_rec, 1.0] if loss_weights is None else list(loss_weights)
    assert len(loss_weights) == 2 and loss_weights[-1] == 1.0, (
        'loss_weights must be [lambda_rec, 1.0], got {}'.format(loss_weights))
    log_path = '{}/{}{}/{}_{}_classes_{}'.format(
        log_dir, model_name, tag, train_type, num_class, dataset)

    arch = dict(D=2, d_model=32, pool=16, n_layers=2, n_dec_layers=2,
                n_heads=4, key_dim=8, ffn_mult=2, dropout_rate=0.3,
                norm_rate=0.25,
                p_band=0.2, n_tmask=1, t_mask_len=40)
    arch.update(overrides)

    model_params = dotdict({
        'model_name': model_name,
        'num_class': num_class,
        'seed': 1234,
        'input_shape': (9, 8, 400),         # (bands, CSP comps = n_csp in prep_HW.py, samples)
        'data_format': 'channels_last',     # Trainer sets this as the global Keras image format
        'loss': [MaskedMSE(), SparseCategoricalCrossentropy()],
        'loss_names': ['recon', 'crossentropy'],   # keep 'crossentropy' (class balancing)
        'loss_weights': loss_weights,
        'adaptive_gradient': False,
        # checkpoint / LR schedule / early stopping on validation classification only
        'monitor': 'val_crossentropy_loss',
        'mode': 'min',
        'class_balancing': True,
        'f1_average': 'macro',
        'optimizer': Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        'lr': 1e-3,
        'min_lr': 1e-4,
        'factor': 0.5,
        'patience': 5,
        'epochs': 300,
        'batch_size': 16,
        'es_patience': 30,
        'min_epochs': 20,
        'verbose': 1,
        'log_path': log_path,
    })
    model_params.update(arch)

    return dotdict({
        'model': HWNet,
        'model_params': model_params,
        'data_params': dotdict({
            'dataset': dataset,
            'train_type': train_type,
            'data_format': None,            # arrays are already (n, 9, 8, 400)
            'data_type': data_type,
            'num_class': num_class,
            'dataset_path': 'datasets',
            'n_subjects': 9,
            'n_folds': 5,
            'load_path': 'datasets/{}/{}/{}_class/'.format(
                dataset, data_type, num_class),
        }),
        'log_path': log_path,
    })
