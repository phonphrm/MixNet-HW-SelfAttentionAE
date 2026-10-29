# TASK: Implement HWNet (masked spectro-temporal autoencoding + filter-bank transformer classifier) in the MixNet repository

You are working in the repository `xydxdy/MixNet`.
Read `INSTRUCTION.md` at the repo root **before writing any code**, and read the header docstring of every file you edit — they define the interfaces you must satisfy.

---

## 1. Goal

Build a **multi-task** MI-EEG classifier for **BCIC IV-2a, subject-dependent, 2-class (left vs right hand)**. It combines classification with a **masked spectro-temporal autoencoding** auxiliary task:

1. **Preprocessing** (`HW_prep.py`): a 9-band filter-bank decomposition + per-band/channel standardization fitted on the training fold only.
2. **Model** (`HWNet.py`): a training-only random **frequency-band + time masking** layer → per-band learnable spatial filters → temporal tokens → **multi-layer self-attention** (shared encoder), feeding two heads:
   - **Reconstruction head (decoder):** reconstructs the *unmasked* filter-bank input from the encoder tokens. The loss is MSE on the **masked** (band, time) cells only, as in MAE/SimMIM.
   - **Classification head:** softmax over the classes.
   Training jointly minimizes `lambda_rec * L_recon + 1.0 * L_crossentropy`.

**Priority: classification is the primary objective. Reconstruction is only an auxiliary regularizer for learning a better representation.** Concretely:
- Success is measured **only** by classification (`test_acc`, `f1-score` vs MixNet). Reconstruction quality is never a goal, a reported result, or a selection criterion.
- The cross-entropy weight stays fixed at `1.0`. The reconstruction weight is small (`lambda_rec = 0.1` by default) so that its gradient never dominates the shared encoder.
- Checkpoint selection, LR schedule, early stopping and hyperparameter choices (including `lambda_rec` and the mask ratios) all use **validation classification** (`val_crossentropy_loss` / `val_acc`) only.
- The decoder is deliberately small, and there is no gradient path from the decoder into the classifier head. Reconstruction can shape the shared encoder only.
- If adding reconstruction lowers validation classification compared with `aug_only` (same masking, `lambda_rec=0`), the auxiliary task is hurting. Lower `lambda_rec`; do not trade classification for reconstruction.
3. **Config** (`configs/HWNet.py`) wiring the two together, with two losses.

This is the multi-task path described in INSTRUCTION §7 ("If You Use a Multi-Task Model"). Every item in that section's table must be handled (see §5).

Hypotheses the user will test (for context, not for you to prove):
- **H1:** reconstructing masked bands/time spans as a low-weight auxiliary task forces the shared encoder to learn cross-band and temporal structure. This improves **classification** accuracy across sessions (train on session `T`, test on session `E`, recorded on a different day) beyond using masking as plain augmentation.
- **H2:** stacked self-attention layers beat the same front-end with plain pooling.

---

## 2. Hard constraints (do not violate)

1. Edit **only** these files:
   - `mixnet/preprocessing/BCIC2a/HW_prep.py` — only `fit_transform()`, `transform()`, and new helper functions. Do **not** change the signature or behaviour of `subject_dependent_setting()`, `_load_bcic2a()`, `_check_shapes()` or `_save_data_with_valset()`.
   - `mixnet/preprocessing/BCIC2a/HW_prep_helpers.py` (new, optional).
   - `mixnet/models/HWNet.py` — `_config()`, `build()`, new custom layer classes, a `MaskedMSE` loss factory, **rewritten** `train_step` / `val_step` / `test_step` / `pred_step` (multi-task), and **overridden** `evaluate()` / `predict()` (INSTRUCTION §7: `base.py` only unpacks multi-output predictions for models named `MixNet`/`MIN2Net`).
   - `experiments/configs/HWNet.py` — `get_params()`.
   - `experiments/prep_HW.py` — only add keyword arguments to the existing call.
   - `env/mixnet.def` (new) and the built `env/mixnet.sif` — environment only, see Step −1.
   Do **not** modify `mixnet/models/base.py`, `mixnet/utils.py`, `mixnet/trainer.py` or any other file.
2. **No data leakage.** Anything data-driven (here: the mean/std) is computed in `fit_transform()` from the training fold only, stored in `state`, and applied in `transform()`. `transform()` must never compute statistics.
3. **Masking is NOT done in preprocessing.** Preprocessing output is deterministic. Masking lives in the model and is active only when `training=True`.
4. Never train on, tune on, or select checkpoints with session `E` (the test set). Do not look at test accuracy to pick hyperparameters.
5. Model name must be `self.model_name` (default `'HWNet'`), never `MixNet` or `MIN2Net`; do not rename the model to reach `base.py`'s multi-output branch. `build()` returns a **list** of outputs with the classifier **last**: `[x_rec, mask, softmax]`. The classifier output is `layers.Activation('softmax', name='softmax')`. Keep the `load_weights` epilogue.
5b. The classification loss must be named `'crossentropy'`, otherwise `Trainer` class balancing breaks. Checkpointing, LR reduction and early stopping must monitor `val_crossentropy_loss`, not the summed `val_loss`. Model selection must depend only on validation classification, not on reconstruction quality.
6. Keep `k_folds=5`, all 9 subjects, and `pick_smp_freq=100`.
7. Target environment: **Python 3.8, TensorFlow 2.7.0**, run inside `env/mixnet.sif` (Step −1). Use only APIs available in TF 2.7 (`tf.keras.layers.MultiHeadAttention` and `LayerNormalization` exist; avoid newer-only APIs).
8. Do not commit or push unless the user asks.
9. Cite external ideas in code comments (INSTRUCTION §10.7): SpecAugment (Park et al., 2019) for the masking layer, MAE (He et al., 2022) and SimMIM (Xie et al., 2022) for masked reconstruction with loss on masked cells only, MixNet (this repo) for the multi-task step/evaluate pattern, EEGNet (Lawhern et al., 2018) for the depthwise spatial filter, and the Transformer (Vaswani et al., 2017) for the attention blocks.
10. Keep runs reproducible (INSTRUCTION §10.9): set an explicit `seed` in the config and do not add any untracked randomness outside `tf.random` ops in the masking layer.

---

## 2b. Step −1 — Environment: build `env/mixnet.sif`

This server has no Docker and no Slurm. It has Apptainer/Singularity (`/usr/bin/apptainer`, `/usr/local/bin/singularity`) and 2× RTX 4090. INSTRUCTION §4.2 uses the `tensorflow/tensorflow:2.7.0-gpu` Docker image (Python 3.8.10, CUDA 11.2). Wrap that same image in a Singularity image, following the convention of `../hemisphere-encoder/env/sphere.def`.

### 2b.1 Create `env/mixnet.def` (repo root: `MixNet/env/mixnet.def`)

```
Bootstrap: docker
From: tensorflow/tensorflow:2.7.0-gpu

# Environment for MixNet + HWNet (INSTRUCTION §4.2: Python 3.8.10, TF 2.7.0 GPU).
# Build (needs --fakeroot):  singularity build --fakeroot env/mixnet.sif env/mixnet.def
# Run (from MixNet/experiments):
#   singularity exec --nv --env PYTHONPATH=$PWD/.. ../env/mixnet.sif python run_HWNet.py ...
# The image is read-only, so `pip install -e .` is replaced by PYTHONPATH pointing at the clone.

%post
    export PIP_NO_CACHE_DIR=1
    # The TF 2.7 image's CUDA apt repo key is stale; skip apt entirely (not needed).
    python -m pip install --upgrade "pip<24"
    # Pins that keep TF 2.7 working: numpy<1.24 (np.object etc. removed), protobuf<4.
    pip install "numpy>=1.21,<1.24" "protobuf==3.20.3" \
        tensorflow-addons==0.16.1 scikit-learn==1.2.2 wget>=3.2 h5py==3.5.0 \
        "pandas>=1.5.3,<2.1" mne==1.6.1 moabb==1.0.0 "ray>=1.11.0" scipy
    python -c "import tensorflow as tf, tensorflow_addons, sklearn, mne, moabb, numpy, scipy, h5py, pandas; print(tf.__version__, numpy.__version__)"

%environment
    export PYTHONUNBUFFERED=1
    export PYTHONNOUSERSITE=1
    export TF_CPP_MIN_LOG_LEVEL=1
    # RTX 4090 (sm_89) is newer than TF 2.7's prebuilt kernels -> PTX JIT on first run; cache it.
    export CUDA_CACHE_MAXSIZE=4294967296

%labels
    Project MixNet-HWNet
    TensorFlow 2.7.0-gpu
```

If a pin fails to resolve under Python 3.8, relax only that pin (for example drop the `moabb` or `ray` version). Record the change in the `.def` comments and in your final summary. Do not change the base image, Python, or TensorFlow versions.

### 2b.2 Build and verify

```bash
cd MixNet
singularity build --fakeroot env/mixnet.sif env/mixnet.def 2>&1 | tee env/build.log
singularity exec --nv --env PYTHONPATH=$PWD env/mixnet.sif python -c \
  "import mixnet, tensorflow as tf; print(mixnet.__file__); print(tf.__version__); \
   print(tf.config.list_physical_devices('GPU')); \
   print(tf.reduce_sum(tf.random.normal((1000,1000)) @ tf.random.normal((1000,1000))))"
```

Expected: `mixnet.__file__` points to `.../MixNet/mixnet/__init__.py` (not `site-packages`, per INSTRUCTION §4.4). TF reports `2.7.0` and lists the 2 GPUs, and the matmul returns a finite number. The first GPU call can take a few minutes because of PTX JIT compilation for sm_89; that is normal. If the GPU op fails with an `sm_89` / `no kernel image` error, report it and fall back to verifying on CPU. Do **not** switch TF versions without asking the user.

### 2b.3 Use the image for every Python command below

Run every `python ...` command in this task (py_compile, unit checks, `prep_HW.py`, `run_HWNet.py`, `benchmark.py`, the MixNet baseline) through the image, from `MixNet/experiments`:

```bash
alias mx='singularity exec --nv --env PYTHONPATH=$(realpath ..) ../env/mixnet.sif'
mx python prep_HW.py
```

Never `pip install` into the host Python or into `~/.local`. `env/mixnet.sif` and `env/build.log` are build artifacts; do not commit them (add `env/*.sif` and `env/build.log` to `.gitignore` only if the user asks you to commit).

---

## 3. Step 0 — Orient yourself (read, don't edit)

- `INSTRUCTION.md`
- `mixnet/preprocessing/BCIC2a/HW_prep.py` (header + driver)
- `mixnet/models/HWNet.py` (header + provided step functions)
- `mixnet/models/EEGNet.py`, `mixnet/models/DeepConvNet.py` (working examples of the same contract)
- `mixnet/models/base.py` (how `build()` is called: 3× per fold, and with `load_weights=True` in `evaluate`; the `MixNet` branch of `evaluate()`/`predict()` is the template for your override)
- `mixnet/models/MixNet.py` — the reference multi-task `train_step`/`val_step`/`test_step`/`pred_step` (unpack outputs, weighted sum of losses, one `*_<name>_loss` log per task, return outputs as a tuple)
- `mixnet/trainer.py`: `testing()`/`prediction()` return `tuple(yp.numpy() for yp in y_pred)`; `self.monitor` drives the checkpoint, `ReduceLROnPlateau` and early stopping; `tf.keras.backend.set_image_data_format(self.data_format)` is applied **globally**.
- `experiments/configs/EEGNet.py` (template for the config)
- `mixnet/utils.py`: check the **exact signature** of `butter_bandpass_filter` (argument order, whether it filters along the last axis, causal vs. zero-phase) and `DataLoader._change_data_format`.
- `mixnet/trainer.py`: confirm which `model_params` keys it reads (`lr`, `min_lr`, `factor`, `patience`, `es_patience`, `min_epochs`, `epochs`, `batch_size`, `seed`, …).

If anything below conflicts with what you find in the code, **follow the code** and note the deviation in your final summary.

---

## 4. Step 1 — Preprocessing (`mixnet/preprocessing/BCIC2a/HW_prep.py`)

Input: `X` with shape `(n, 20, 400)` (float), and `y` in `{0, 1}`.

### 4.1 Defaults (overridable via `**kwargs` from `prep_HW.py`)

```python
DEFAULT_BANDS = [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24],
                 [24, 28], [28, 32], [32, 36], [36, 40]]
DEFAULT_ORDER = 4
EPS = 1e-8
```

### 4.2 Helper (stateless)

```python
def filter_bank(X, pick_smp_freq, bands=DEFAULT_BANDS, filter_order=DEFAULT_ORDER):
    """(n, C, T) -> (n, Nb, C, T) float32. Band-pass each band along the time axis."""
```
- Use zero-phase SOS filtering: `scipy.signal.butter(order, [lo, hi], btype='band', fs=pick_smp_freq, output='sos')` + `scipy.signal.sosfiltfilt(sos, X, axis=-1)`. Do **not** use `mixnet.utils.butter_bandpass_filter`. It is zero-phase and filters along the last axis, but it uses the `(b, a)` form, which can be numerically unstable for narrow 4 Hz bands at 100 Hz, especially 36–40 Hz near the 50 Hz Nyquist frequency.
- Make sure the output is finite.

### 4.3 `fit_transform(X, y, pick_smp_freq, bands=None, filter_order=None, **kwargs)`

1. `Xf = filter_bank(X, ...)` → `(n, 9, 20, 400)`.
2. `mu = Xf.mean(axis=(0, 3))` → `(9, 20)`; `sd = Xf.std(axis=(0, 3))` → `(9, 20)`. Training fold only.
3. `X_out = ((Xf - mu[None, :, :, None]) / (sd[None, :, :, None] + EPS)).astype(np.float32)`.
4. `state = {'bands': bands, 'filter_order': order, 'mu': mu, 'sd': sd}` (plain numpy/Python, picklable).
5. Return `(X_out, state)`. `y` is unused (unsupervised step); keep the argument.

### 4.4 `transform(X, state, pick_smp_freq, **kwargs)`

- Filter with `state['bands']` and `state['filter_order']`, normalize with `state['mu']` and `state['sd']`, cast to float32. **No statistics computed here.**

### 4.5 `experiments/prep_HW.py`

Add to the existing call:
```python
bands=[[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]],
filter_order=4,
```

### 4.6 Verify (run these)

- If the dataset is present: `cd experiments && python prep_HW.py` → **270 files** in `datasets/BCIC2a/HW_prep/2_class/subject_dependent/`, with shapes train `(~115, 9, 20, 400)`, val `(~29, 9, 20, 400)`, test `(144, 9, 20, 400)`.
- If the dataset is **not** present, write and run a throwaway synthetic check (do not commit it), e.g. `X = np.random.randn(115, 20, 400)`:
  - output shape `(115, 9, 20, 400)`, dtype float32, all finite;
  - training output per-(band, channel) mean ≈ 0 and std ≈ 1;
  - `transform` on a different random array uses the stored stats: re-running `transform` twice gives identical output, and its mean/std are **not** forced to 0/1.

---

## 5. Step 2 — Model (`mixnet/models/HWNet.py`)

### 5.1 Custom layer: `SpectroTemporalMask` (define in `HWNet.py`, above the class)

Input: `(batch, Nb, C, T)`. Returns **two tensors**: `[x_masked, mask]`, where `mask` has shape `(batch, Nb, 1, T)`, with 1 = visible and 0 = masked. The decoder loss needs the mask.
- `training` false/None → return `[x, ones]` (`mask = tf.ones([tf.shape(x)[0], Nb, 1, T])`).
- `training` true:
  - **Band mask:** per sample, drop each band independently with probability `p_band`; guarantee at least one band survives (e.g., if all are dropped, keep one random band). Shape `(batch, Nb, 1, 1)`.
  - **Time mask:** per sample, `n_tmask` contiguous spans; each span has length uniform in `[0, t_mask_len]` and a random start, shared across bands and channels. Shape `(batch, 1, 1, T)`. Build it with vectorized `tf.range` comparisons (no Python loops over the batch).
  - `mask = band_mask * time_mask` (broadcast to `(batch, Nb, 1, T)`); `x_masked = x * mask` (masked value = 0, i.e. the training mean after standardization). No rescaling.
- Use only `tf.random.*` ops inside `call` so the masks are re-sampled every step and the layer works inside `@tf.function`.
- Implement `get_config()` returning `p_band`, `n_tmask` and `t_mask_len`.
- `p_band=0` and `n_tmask=0` must give `x_masked == x` and `mask == 1` exactly (needed for the ablations).

### 5.1b Loss: `MaskedMSE()` (define in `HWNet.py`; do not edit `mixnet/loss.py`)

A factory in the style of `mixnet.loss.MeanSquaredError`, returning `inner_masked_mse(x, x_rec, mask)`:
```python
err = tf.square(x - x_rec)                       # (b, Nb, C, T)
w   = (1.0 - mask) * tf.ones_like(err)           # 1 on masked cells
n   = tf.reduce_sum(w)
loss = tf.cond(n > 0,
               lambda: tf.reduce_sum(w * err) / n,   # MAE/SimMIM: masked cells only
               lambda: tf.reduce_mean(err))          # nothing masked (eval, or p_band=n_tmask=0): full-input MSE
```
The reconstruction target is the **unmasked standardized input `x`** that the step function receives.

Because masking is off at `training=False`, `val_recon_loss` and `test_recon_loss` measure plain full-input reconstruction. They are logged for diagnostics only and must never drive model selection (constraint 5b).

### 5.2 `_config(self, **kwargs)`

Set the defaults, derive the shapes, then keep the provided override loop **last**:

```python
self.D = 2                 # spatial filters per band
self.d_model = 32
self.pool = 16             # temporal pooling -> T/pool tokens (400/16 = 25)
self.n_layers = 2          # transformer blocks; 0 = no attention (ablation)
self.n_heads = 4
self.key_dim = 8
self.ffn_mult = 2
self.dropout_rate = 0.3
self.p_band = 0.2
self.n_tmask = 1
self.t_mask_len = 40
self.norm_rate = 0.25
# (lambda_rec is not a model attribute: it is loss_weights[0], set in the config)
# input_shape = (Nb, C, T); data_format=None so no depth axis
self.Nb, self.Chans, self.Samples = self.input_shape
for k in kwargs.keys():
    self.__setattr__(k, kwargs[k])
```
Note: `input_shape` is a named argument of `HWNet.__init__`, so it never appears in `kwargs`. To be safe, still derive `Nb/Chans/Samples` again in `build()`. The input layout is always `(Nb, C, T)`, but `data_format` is **not** irrelevant: `Trainer` calls `tf.keras.backend.set_image_data_format(data_format)` globally. Set the default to `'channels_last'` in `HWNet.__init__` and in the config, and pass `data_format='channels_last'` explicitly to every `Conv2D` / `DepthwiseConv2D` / `AveragePooling2D`.

### 5.3 `build(self, print_summary=True, load_weights=False)`

```
input1 (Nb, C, T)
→ x_masked, mask = SpectroTemporalMask(p_band, n_tmask, t_mask_len)(input1)
── encoder (shared) ─────────────────────────────────────────────
x_masked
→ Permute((2, 3, 1))                                   # (C, T, Nb)
→ DepthwiseConv2D((C, 1), depth_multiplier=D, use_bias=False,
                  depthwise_constraint=max_norm(1.))   # (1, T, Nb*D)
→ BatchNormalization → Activation('elu')
→ Conv2D(d_model, (1, 1), use_bias=False)              # (1, T, d_model)
→ BatchNormalization → Activation('elu')
→ AveragePooling2D((1, pool))                          # (1, T/pool, d_model)
→ Dropout(dropout_rate)
→ Reshape((T // pool, d_model))
→ + learnable positional embedding, shape (1, T//pool, d_model)
   (implement as a small custom layer with add_weight, or an Embedding over tf.range)
→ n_layers × pre-LN transformer block:
     h = LayerNormalization(epsilon=1e-6)(x)
     h = MultiHeadAttention(num_heads=n_heads, key_dim=key_dim, dropout=dropout_rate)(h, h)
     x = Add()([x, Dropout(dropout_rate)(h)])
     h = LayerNormalization(epsilon=1e-6)(x)
     h = Dense(ffn_mult * d_model, activation='elu')(h)
     h = Dense(d_model)(h)
     x = Add()([x, Dropout(dropout_rate)(h)])
→ tokens = LayerNormalization(epsilon=1e-6)            # (T/pool, d_model), shared by both heads
── classification head ──────────────────────────────────────────
tokens
→ GlobalAveragePooling1D
→ Dropout(dropout_rate)
→ Dense(num_class, kernel_constraint=max_norm(norm_rate))
→ softmax = Activation('softmax', name='softmax')
── reconstruction head (lightweight decoder, mirrors the encoder) ──
tokens
→ Dense(pool * Nb * D)                                 # (T/pool, pool*Nb*D)   undo pooling
→ Reshape((T, Nb * D))                                 # (T, Nb*D)  token-major -> time-major
→ Dense(Nb * C)                                        # (T, Nb*C)  undo spatial filtering
→ Reshape((T, Nb, C)) → Permute((2, 3, 1))             # (Nb, C, T)
→ x_rec = Activation('linear', name='recon')
model = Model(inputs=input1, outputs=[x_rec, mask, softmax], name=self.model_name)
(keep the provided summary / load_weights epilogue)
```
Requirements:
- Output order is fixed: `[x_rec, mask, softmax]`, **classifier last** (INSTRUCTION §7). Give `mask` a name too (for example wrap it in `Activation('linear', name='mask')`).
- The decoder reconstructs the **full** `(Nb, C, T)` tensor, but only masked cells contribute to the training loss (§5.1b). Keep the decoder small: it is an auxiliary head, and capacity spent there does not help classification.
- Deterministic graph construction (no randomness at build time apart from weight init); `build()` is called 3× per fold.
- The custom layers must work with `model.load_weights(h5)` (weights-only loading, so `get_config` suffices; no `custom_objects` needed).
- Trainable parameters **including the decoder** should be **< 50k**; report encoder, decoder and total counts.
- `T % pool == 0` must hold (assert with a clear message).

### 5.4 Multi-task step functions (rewrite the provided ones, following `MixNet.py`)

```python
@tf.function
def train_step(self, x, y, loss_weights):
    with tf.GradientTape() as tape:
        x_rec, mask, y_logis = self.model(x, training=True)
        recon_loss        = self.loss.recon(x, x_rec, mask)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses     = [recon_loss, crossentropy_loss]          # same order as loss_names
        train_loss = tf.reduce_sum(loss_weights * tf.stack(losses))
        self.train_acc_metric.update_state(y, y_logis)
    grads = tape.gradient(train_loss, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    logs = {'train_loss': train_loss}
    logs.update(dict(zip(['train_' + n + '_loss' for n in self.loss_names], losses)))
    logs.update({'train_acc': self.train_acc_metric.result()})
    return logs, (x_rec, mask, y_logis)
```
- `val_step` / `test_step`: same unpacking and losses with `training=False`, `val_`/`test_` prefixes, returning `(x_rec, mask, y_logis)`.
- `pred_step`: `return self.model(x, training=False)` (a 3-tuple).
- Use the incoming `loss_weights`; do not hard-code `lambda_rec` in the steps.
- `train_acc` / `val_acc` / `test_acc` must come from the classifier output only, exactly as in the single-task steps.

### 5.5 Override `evaluate()` and `predict()` in `HWNet`

Copy `BaseModel.evaluate()` / `predict()` and replace the name-based branching with a single unpack:
```python
x_rec, mask, y_pred_clf = test_pred            # classifier last
y_pred_argm = np.argmax(y_pred_clf, axis=1)
Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf': y_pred_clf}
```
- Do **not** put `x_rec` in `Y`. `run_HWNet.py` saves `Y` to `S*_prediction_results.npy`, and 144×9×20×400 floats per fold would make that file ~200 MB per subject.
- Keep everything else identical to the base, so `benchmark.py` sees the same columns: `self.build(..., load_weights=True)`, `self.compile(model)`, `self.testing(...)`, timing, GPU memory, `f1-score` with `self.f1_average`, `classification_report`, and the `w_<name>_loss` entries from `self.best_loss_weights`.

### 5.6 Smoke test first

Before the full architecture, temporarily use the instruction's minimal model (Flatten → Dense → softmax) **with the original single-task step functions and a single-loss config** (`[SparseCategoricalCrossentropy()]`, `['crossentropy']`, `[1.0]`). Confirm that `python run_HWNet.py --subjects 1` works end to end (if the data and GPU exist). Then switch to the multi-task `build()`, steps, `evaluate()` and config together.

---

## 6. Step 3 — Config (`experiments/configs/HWNet.py`)

Implement `get_params(dataset, train_type, data_type, num_class, loss_weights=None, log_dir='logs', **kwargs)` following `configs/EEGNet.py`, with:

| Key | Value |
|---|---|
| `model` | `HWNet`. It is already exported in `mixnet/models/__init__.py`, so use `from mixnet.models import HWNet`. Do not edit `__init__`. |
| `model_params.model_name` | `'HWNet'` |
| `model_params.num_class` | `num_class` (argument) |
| `seed` | `1234` (the `Trainer` default, set explicitly for reproducibility) |
| `model_params.input_shape` | `(9, 20, 400)` |
| `model_params.data_format` | `'channels_last'` (required: `Trainer` sets it as the global Keras image format) |
| `loss` | `[MaskedMSE(), SparseCategoricalCrossentropy()]` (`from mixnet.models.HWNet import MaskedMSE`) |
| `loss_names` | `['recon', 'crossentropy']` (same order as the outputs/steps; keep `'crossentropy'` exactly) |
| `loss_weights` | `[lambda_rec, 1.0]` if the `loss_weights` argument is `None`, with `lambda_rec = kwargs.get('lambda_rec', 0.1)` (small on purpose: classification has priority, reconstruction is a regularizer). Assert `loss_weights[-1] == 1.0`; otherwise use the 2 values passed via `run_HWNet.py --loss_weights` |
| `adaptive_gradient` | `False` (fixed weights; MixNet's adaptive blending is out of scope, since val recon loss is unmasked and not comparable to train recon loss) |
| `monitor` | `'val_crossentropy_loss'` (with `mode='min'`), so checkpoint, LR schedule and early stopping depend on validation classification only |
| `class_balancing`, `f1_average` | `True`, `'macro'` |
| `optimizer` | `Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08)` |
| `lr`, `min_lr`, `factor`, `patience` | `1e-3`, `1e-4`, `0.5`, `5` |
| `epochs`, `batch_size`, `es_patience`, `min_epochs` | `300`, `16`, `30`, `20` |
| architecture keys | `D=2, d_model=32, pool=16, n_layers=2, n_heads=4, key_dim=8, ffn_mult=2, dropout_rate=0.3, norm_rate=0.25` |
| mask keys | `p_band=0.2, n_tmask=1, t_mask_len=40` |
| `verbose` | `1` |
| `log_path` | `'{log_dir}/{model_name}{tag}/{train_type}_{num_class}_classes_{dataset}'`, where `tag = kwargs.get('tag', '')`. Set it **both** in `model_params` (the `Trainer` writes checkpoints there) and at the top level of the returned dotdict (`run_HWNet.py` reads `config.log_path`). The tag changes the top-level folder name (for example `logs/HWNet_nomask/`), so `benchmark.py` lists each ablation as a separate run. |
| `data_params.data_format` | `None` (the arrays are already `(n, 9, 20, 400)`; do **not** use `'NSHWD'`, which zero-pads) |
| `data_params` (others) | same as EEGNet: `dataset`, `train_type`, `data_type`, `num_class`, `dataset_path='datasets'`, `n_subjects=9`, `n_folds=5`, `load_path='datasets/{dataset}/{data_type}/{num_class}_class/'` |

Also add a module-level dict `ABLATIONS` (documentation only; the user will switch between entries by editing values / `tag`):
```python
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
```
`lambda_rec` is not a model attribute. `get_params()` must pop it from the ablation entry and turn it into `loss_weights[0]`; every other key goes into `model_params`.

If it is easy and does not require editing `run_HWNet.py`, allow an environment variable `HWNET_ABLATION=<key>` read inside `get_params()` to apply an entry and set `tag='_<key>'`. Otherwise skip it.

Ablations exist to test H1/H2. They are not a way to choose the final configuration by test score. If the user picks a final configuration from among them, the choice must be based on **validation** accuracy/loss only (`val_*` columns in `S*_fold*_out_log.csv`), never on `test_acc` (INSTRUCTION §7, §10.3).

---

## 7. Step 4 — Verification you must run

Run whatever is possible in the local environment and report honestly what was and wasn't run.

1. **Static:** `python -m py_compile` on all edited files.
2. **Model unit check** (throwaway script, not committed; needs TF):
   - Build `HWNet(optimizer=Adam(), input_shape=(9, 20, 400))` and call `build(print_summary=True)`; record the parameter count.
   - Feed `np.random.randn(4, 9, 20, 400).astype('float32')`. Expect 3 outputs: `x_rec` `(4, 9, 20, 400)`, `mask` `(4, 9, 1, 400)`, softmax `(4, 2)` with rows summing to 1.
   - `model(x, training=False)` called twice → identical outputs, and `mask` is all ones. `SpectroTemporalMask(...)(x, training=True)` called twice → different masks, every sample keeps ≥ 1 band, and `x_masked == x * mask`. With `p_band=0, n_tmask=0` and `training=True` → `x_masked == x` and `mask == 1`.
   - `MaskedMSE`: with a hand-made mask, the loss equals the MSE over masked cells only. With an all-ones mask it equals the full MSE. The result is finite in both cases.
   - Instantiate `HWNet` with the config's two-loss setup and run `train_step` once on a random batch. `train_recon_loss` and `train_crossentropy_loss` are both in the logs and finite, and **decoder weights receive non-zero gradients** (so the reconstruction head is wired into the loss).
   - Build twice, save weights from one, `load_weights` into the other → identical outputs.
   - `evaluate()` on a random test set with an already-saved checkpoint returns `Y` with `y_true/y_pred/y_pred_clf` (no `x_rec`), and an `evaluation` dict containing `test_acc`, `f1-score`, `test_recon_loss`, `test_crossentropy_loss`, `w_recon_loss` and `w_crossentropy_loss`.
3. **Preprocessing check:** as in Section 4.6.
4. **End to end** (only if the dataset and TF/GPU are available): `python prep_HW.py`, then `python run_HWNet.py --subjects 1`. Confirm that `logs/HWNet/subject_dependent_2_classes_BCIC2a/S001_all_results.csv` exists with 5 rows. Also confirm that `S001_fold01_out_log.csv` has `train_recon_loss`, `val_recon_loss`, `train_crossentropy_loss` and `val_crossentropy_loss` columns, and that `train_recon_loss` decreases over epochs. Report the first-epoch values of `lambda_rec * train_recon_loss` and `train_crossentropy_loss`. The weighted reconstruction term should be **smaller** than the cross-entropy; if it is not, say so in the final message, because the auxiliary task may be dominating.

Do **not** run the full 9-subject benchmark or tune hyperparameters on test results; that is the user's job.

---

## 8. Deliverable / final message

When done, reply with:
- the files changed (with a one-line summary each);
- the parameter count (encoder / decoder / total) and the input/output shapes of all 3 outputs;
- which checks in Section 7 were run and their results (paste the key output lines), and which could not run and why;
- any deviations from this spec, and why;
- the exact commands the user should run next. The MixNet baseline (INSTRUCTION §5) must exist under `logs/MixNet/` before `benchmark.py` can do the paired comparison:
  ```bash
  cd experiments
  # (once) build the image: see Step −1
  # singularity build --fakeroot ../env/mixnet.sif ../env/mixnet.def
  alias mx='singularity exec --nv --env PYTHONPATH=$(realpath ..) ../env/mixnet.sif'
  # MixNet baseline (once)
  mx python prep_spectral_spatial_signals.py --dataset 'BCIC2a' --setting 'dependent'
  mx python run_MixNet.py --model_name 'MixNet' --dataset 'BCIC2a' \
      --train_type 'subject_dependent' --data_type 'spectral_spatial_signals' \
      --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' \
      --num_class 2 --GPU 0 --margin 1.0 --n_component 2 --warmup 7
  # HWNet
  mx python prep_HW.py
  mx python run_HWNet.py --subjects 1        # smoke test
  mx python run_HWNet.py --GPU 0             # full run
  mx python benchmark.py --baseline MixNet --candidate HWNet
  ```

## 9. Out of scope (do not implement)

- Separate self-supervised **pre-training** followed by fine-tuning (reconstruction is trained **jointly** with classification only), and MAE's drop-the-masked-tokens encoder (this model is SimMIM-style: masked cells are zeroed and the encoder sees all tokens).
- MixNet's adaptive gradient blending, triplet/metric-learning losses, or any third task.
- Subject-independent (LOSO) evaluation.
- Changes to the training loop, `base.py`, `DataLoader`, or `benchmark.py`.
- Writing the report.
- Upgrading TensorFlow/CUDA or changing the base image of `env/mixnet.def`.
