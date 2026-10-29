"""
=============================================================================
 HOMEWORK PART 1 -- BUILD YOUR OWN PREPROCESSING PIPELINE  (BCIC2a)
=============================================================================

Your job in this file is to turn *raw* BCI Competition IV-2a MI-EEG trials
into the feature tensors that your model in `mixnet/models/HWNet.py`
will be trained on.

WHAT IS ALREADY DONE FOR YOU (do not change unless you know why):
  * reading + cropping the raw .mat files              -> `_load_bcic2a()`
  * the 5-fold subject-dependent cross-validation loop -> `subject_dependent_setting()`
  * saving the .npy files with the exact file names    -> `_save_data_with_valset()`
    that `mixnet.utils.DataLoader` expects

WHAT YOU MUST IMPLEMENT (the two `TODO` functions below):
  1. `fit_transform()` -- learn any data-driven parameters on the TRAINING
     fold only, and return the transformed training features.
  2. `transform()`     -- apply the *already learned* parameters to the
     validation / test folds.

-----------------------------------------------------------------------------
THE GOLDEN RULE OF THIS ASSIGNMENT
-----------------------------------------------------------------------------
 Anything that *LEARNS* from data (CSP filters, PCA, a mean/std scaler, a
 whitening matrix, a channel-selection criterion, ...) must be fitted on the
 TRAINING fold ONLY, then merely *applied* to validation and test data.
 Fitting on all the data first and splitting afterwards is data leakage: it
 silently inflates your accuracy and makes the benchmark against MixNet
 meaningless. Marks are deducted for leakage.

 Stateless operations (band-pass filtering, cropping, resampling, log of the
 variance, ...) are safe to apply to every split independently.
-----------------------------------------------------------------------------

Data you receive in `fit_transform` / `transform`
    X : np.ndarray, shape (n_trials, n_channels, n_samples)
        n_channels = len(sel_chs)              (20 by default, see config.py)
        n_samples  = MI_len * pick_smp_freq    (4 s * 100 Hz = 400 by default)
    y : np.ndarray, shape (n_trials,)  with values in {0, 1}
        0 = left hand, 1 = right hand

Data you must return
    X_out : np.ndarray, shape (n_trials, ...anything you like...)
        Keep the trial axis first. Whatever trailing shape you choose decides
        the `input_shape` of your model: `DataLoader` reshapes it according to
        `data_format`, and the result minus the trial axis is what your
        network receives. The driver below prints the shape after every fold
        -- note it down, you need it in `experiments/configs/HWNet.py`. See
        the data_format table in that file.
"""

import os

import numpy as np
from scipy import signal
from sklearn.model_selection import StratifiedKFold

from mixnet.preprocessing.BCIC2a import raw
from mixnet.preprocessing.BCIC2a.HW_prep_helpers import apply_csp, fit_csp
from mixnet.preprocessing.config import CONSTANT

CONSTANT = CONSTANT['BCIC2a']
raw_path = CONSTANT['raw_path']
n_subjs = CONSTANT['n_subjs']
orig_smp_freq = CONSTANT['orig_smp_freq']
MI_len = CONSTANT['MI']['len']

# Name of the folder your features are written to. It must match the
# `--data_type` argument of run_HWNet.py. Keep it as is.
DATA_TYPE = 'HW_prep'


# =============================================================================
#  HWNet pipeline: 9-band filter bank -> per-band CSP projection ->
#                  per-(band, component) standardization
#
#  Stateless : zero-phase Butterworth band-pass (SOS form) per band;
#              projection onto stored CSP filters.
#  Fitted    : CSP filters (supervised, uses y) and mean / std per
#              (band, component) -- on the TRAINING fold only.
#  Output    : (n, Nb, n_csp, T) time series, i.e. the time axis is kept.
#  Random band/time masking is NOT done here (it lives in the model and is
#  active only during training), so this output is deterministic.
#  n_csp=None skips CSP (plain filter bank on all channels, (n, Nb, C, T)).
# =============================================================================
DEFAULT_BANDS = [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24],
                 [24, 28], [28, 32], [32, 36], [36, 40]]
DEFAULT_ORDER = 4
DEFAULT_N_CSP = 8
EPS = 1e-8


def filter_bank(X, pick_smp_freq, bands=DEFAULT_BANDS,
                filter_order=DEFAULT_ORDER):
    """(n, C, T) -> (n, Nb, C, T) float32. Band-pass each band along time.

    Uses second-order sections + `sosfiltfilt` (zero-phase) rather than
    `mixnet.utils.butter_bandpass_filter`, whose (b, a) form is numerically
    fragile for narrow 4 Hz bands at 100 Hz (e.g. 36-40 Hz near Nyquist).
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.empty((X.shape[0], len(bands)) + X.shape[1:], dtype=np.float32)
    for i, (lo, hi) in enumerate(bands):
        sos = signal.butter(filter_order, [lo, hi], btype='band',
                            fs=pick_smp_freq, output='sos')
        out[:, i] = signal.sosfiltfilt(sos, X, axis=-1)
    if not np.all(np.isfinite(out)):
        raise ValueError('filter_bank produced NaN/Inf; check bands {} and '
                         'order {} at fs={} Hz'.format(bands, filter_order,
                                                       pick_smp_freq))
    return out


def fit_transform(X, y, pick_smp_freq, bands=None, filter_order=None,
                  n_csp=DEFAULT_N_CSP, **kwargs):
    """Filter-bank the TRAINING fold, fit per-band CSP and a scaler on it.

    Returns:
        X_out (np.ndarray): (n_trials, Nb, n_csp, T) float32, standardized
            ((n_trials, Nb, C, T) when n_csp is None).
        state (dict): {'bands', 'filter_order', 'W', 'mu', 'sd'}; W is
            (Nb, n_csp, C) or None, mu/sd are (Nb, n_csp). All of them are
            computed from this training fold only.
    """
    bands = DEFAULT_BANDS if bands is None else [list(b) for b in bands]
    order = DEFAULT_ORDER if filter_order is None else int(filter_order)

    Xf = filter_bank(X, pick_smp_freq, bands, order)
    W = None if n_csp is None else fit_csp(Xf, y, int(n_csp))
    Xs = Xf if W is None else apply_csp(Xf, W)
    mu = Xs.mean(axis=(0, 3), dtype=np.float64)        # (Nb, K)
    sd = Xs.std(axis=(0, 3), dtype=np.float64)         # (Nb, K)
    state = {'bands': bands, 'filter_order': order, 'W': W,
             'mu': mu, 'sd': sd}
    return _standardize(Xs, mu, sd), state


def transform(X, state, pick_smp_freq, **kwargs):
    """Apply filter bank + stored CSP filters + stored scaler. Fits nothing."""
    Xf = filter_bank(X, pick_smp_freq, state['bands'], state['filter_order'])
    Xs = Xf if state['W'] is None else apply_csp(Xf, state['W'])
    return _standardize(Xs, state['mu'], state['sd'])


def _standardize(Xf, mu, sd):
    return ((Xf - mu[None, :, :, None]) /
            (sd[None, :, :, None] + EPS)).astype(np.float32)


# =============================================================================
#  Provided for you -- the subject-dependent 5-fold CV driver
#  
#  NOTE: you are allowed to add any extra arguments to `fit_transform()`, 
# `transform()`, and `subject_dependent_setting()`
#  but you must NOT change the signature of `subject_dependent_setting()`.
# =============================================================================
def subject_dependent_setting(k_folds, pick_smp_freq, save_path, num_class=2,
                              sel_chs=None, subjects=None, **kwargs):
    """Run the whole pipeline for every subject in the subject-dependent setting.

    Subject-dependent means: for each of the 9 subjects we train, validate and
    test on that subject alone. Session T is split into train/validation with
    stratified k-fold CV; session E is the untouched test set of every fold.

    Args:
        k_folds (int): number of CV folds (5 to match the MixNet baseline).
        pick_smp_freq (int): target sampling rate in Hz (100 in the baseline).
        save_path (str): dataset root, e.g. 'datasets'.
        num_class (int): 2 = left hand vs right hand.
        sel_chs (list[str] | None): channel names; None = all 20 from config.py.
        subjects (list[int] | None): 1-based subject ids; None = all 9.
        **kwargs: forwarded verbatim to `fit_transform()` / `transform()`.
    """
    sel_chs = CONSTANT['sel_chs'] if sel_chs is None else sel_chs
    save_path = os.path.join(
        save_path, 'BCIC2a', DATA_TYPE, '{}_class'.format(num_class),
        'subject_dependent')
    os.makedirs(save_path, exist_ok=True)

    id_chosen_chs = raw.chanel_selection(sel_chs)
    subjects = range(1, n_subjs + 1) if subjects is None else subjects

    for subject in subjects:
        X_tr, y_tr, X_te, y_te = _load_bcic2a(
            subject, pick_smp_freq, num_class, id_chosen_chs)

        if X_tr.ndim != 3:
            raise Exception('Dimension Error, must have 3 dimensions, '
                            'found {}'.format(X_tr.shape))

        skf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
        for fold, (train_index, val_index) in enumerate(skf.split(X_tr, y_tr)):
            print('SUBJECT:', subject, 'FOLD:', fold + 1,
                  'TRAIN:', len(train_index), 'VALIDATION:', len(val_index))

            X_tr_cv, X_val_cv = X_tr[train_index], X_tr[val_index]
            y_tr_cv, y_val_cv = y_tr[train_index], y_tr[val_index]

            # ---- your pipeline: fit on train only, then apply everywhere ----
            X_tr_out, state = fit_transform(
                X_tr_cv, y_tr_cv, pick_smp_freq=pick_smp_freq, **kwargs)
            X_val_out = transform(
                X_val_cv, state, pick_smp_freq=pick_smp_freq, **kwargs)
            X_te_out = transform(
                X_te, state, pick_smp_freq=pick_smp_freq, **kwargs)
            # -----------------------------------------------------------------

            _check_shapes(X_tr_out, X_val_out, X_te_out,
                          y_tr_cv, y_val_cv, y_te)
            print('Feature shapes -- train {}, val {}, test {}'.format(
                X_tr_out.shape, X_val_out.shape, X_te_out.shape))

            save_name = 'S{:03d}_fold{:03d}'.format(subject, fold + 1)
            _save_data_with_valset(save_path, save_name,
                                   X_tr_out, y_tr_cv,
                                   X_val_out, y_val_cv,
                                   X_te_out, y_te)
            print('The preprocessing of subject {} from fold {} is DONE!!!'
                  .format(subject, fold + 1))


def _load_bcic2a(subject, pick_smp_freq, num_class, id_chosen_chs):
    """Read session T (train) and session E (test), crop to the MI window."""
    start = CONSTANT['MI']['start']   # 2 s after the cue
    stop = CONSTANT['MI']['stop']     # 6 s after the cue
    return raw.load_crop_data(PATH=raw_path, subject=subject,
                              start=start, stop=stop,
                              new_smp_freq=pick_smp_freq,
                              num_class=num_class,
                              id_chosen_chs=id_chosen_chs)


def _check_shapes(X_tr, X_val, X_te, y_tr, y_val, y_te):
    """Fail early and loudly on the mistakes that cost the most debugging time."""
    for name, X, y in [('train', X_tr, y_tr), ('val', X_val, y_val),
                       ('test', X_te, y_te)]:
        X = np.asarray(X)
        if X.shape[0] != len(y):
            raise Exception(
                'Trial-axis mismatch on the {} split: X has {} trials but y '
                'has {}. Your transform must keep the trial axis first and '
                'must not drop or reorder trials.'.format(
                    name, X.shape[0], len(y)))
        if not np.all(np.isfinite(X)):
            raise Exception(
                'The {} split contains NaN or Inf. This usually means an '
                'unstable filter (check the order and the band edges against '
                'the Nyquist frequency) or a division by a zero variance.'
                .format(name))
    if X_tr.shape[1:] != X_val.shape[1:] or X_tr.shape[1:] != X_te.shape[1:]:
        raise Exception(
            'Feature shapes differ across splits: train {}, val {}, test {}. '
            'transform() must produce the same trailing shape as '
            'fit_transform().'.format(
                X_tr.shape[1:], X_val.shape[1:], X_te.shape[1:]))


def _save_data_with_valset(save_path, NAME, X_train, y_train,
                           X_val, y_val, X_test, y_test):
    """Write the six .npy files under the names `DataLoader` looks for."""
    np.save(save_path + '/X_train_' + NAME + '.npy', X_train)
    np.save(save_path + '/X_val_' + NAME + '.npy', X_val)
    np.save(save_path + '/X_test_' + NAME + '.npy', X_test)
    np.save(save_path + '/y_train_' + NAME + '.npy', y_train)
    np.save(save_path + '/y_val_' + NAME + '.npy', y_val)
    np.save(save_path + '/y_test_' + NAME + '.npy', y_test)
    print('save DONE')
