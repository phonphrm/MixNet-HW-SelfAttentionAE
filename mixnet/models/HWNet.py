"""
=============================================================================
 HOMEWORK PART 2 -- BUILD YOUR OWN MODEL  (HWNet)
=============================================================================

`HWNet` plugs into the exact same training machinery as MixNet, EEGNet
and DeepConvNet, so the benchmark in Part 3 is a fair comparison: 
same folds, same subjects, same early stopping, same metrics.

WHAT IS ALREADY DONE FOR YOU (do not change unless you know why):
  * `train_step` / `val_step` / `test_step` / `pred_step` -- the custom
    TensorFlow loops that `mixnet.trainer.Trainer` calls every batch.
    NOTE: they are written for a SINGLE-TASK model -- one output, one loss
    (cross-entropy). If you go multi-task you MUST rewrite them; see
    "IF YOU GO MULTI-TASK" below.
  * inheritance from `BaseModel`, which gives you `.fit()`, `.evaluate()`
    and `.predict()` for free.

WHAT YOU MUST IMPLEMENT:
  1. `_config()` -- declare your architecture hyper-parameters.
  2. `build()`   -- return a `tf.keras.models.Model`.

-----------------------------------------------------------------------------
 CONTRACT YOUR `build()` MUST RESPECT
-----------------------------------------------------------------------------
  * Input : one tensor of shape `self.input_shape` (no batch axis).
  * Output: ONE tensor of shape (batch, num_class) holding probabilities,
            i.e. finish with `layers.Activation('softmax')`.
            Multi-output models take a different code path in
            `mixnet/models/base.py` and need extra work -- see
            "IF YOU GO MULTI-TASK" below before you try one.
  * Name  : pass `name=self.model_name`. Do NOT name it 'MixNet' or
            'MIN2Net' -- `base.py` branches on those names.
  * When `load_weights=True` you must call `model.load_weights(self.weights_dir)`
    (kept in the template below), otherwise `evaluate()` silently scores an
    untrained network.
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------
 NOTE: IF YOU GO MULTI-TASK (optional, advanced -- read this before you start)
-----------------------------------------------------------------------------
 MixNet itself is multi-task: it optimises reconstruction (MSE) + deep metric
 learning (triplet) + classification (cross-entropy) at once, with adaptive
 loss weights. You are allowed to do the same, but the single-task contract
 above then no longer holds and FOUR things must change together:

 1. `build()` returns a LIST of outputs, with the classifier softmax LAST:
        Model(inputs=input1, outputs=[decoder_out, latent, softmax], ...)

 2. The config (`experiments/configs/HWNet.py`) must declare one entry per
    task, in the SAME order, in all three lists:
        'loss':         [MeanSquaredError(), triplet_loss(margin=1.0),
                         SparseCategoricalCrossentropy()],
        'loss_names':   ['mse', 'triplet', 'crossentropy'],
        'loss_weights': [1.0, 1.0, 1.0],
    Keep the name 'crossentropy' for the classification loss -- `Trainer`
    looks it up by that name to apply class balancing.

 3. `train_step` / `val_step` / `test_step` / `pred_step` below must compute
    every loss, combine them with the incoming `loss_weights`, and log one
    `*_<loss_name>_loss` entry per task. `mixnet/models/MixNet.py` is the
    reference implementation -- read its steps side by side with the
    single-task ones below. The shape of a multi-task step is:

        xr, z, y_logis = self.model(x, training=True)
        mse_loss   = self.loss.mse(x, xr)
        trp_loss   = self.loss.triplet(y, z)
        ce_loss    = self.loss.crossentropy(y, y_logis)
        losses     = [mse_loss, trp_loss, ce_loss]     # same order as loss_names
        train_loss = tf.reduce_sum(loss_weights * losses)
        ...
        return logs, (xr, z, y_logis)                  # outputs, classifier last

    The `loss_weights` argument is not decoration: `Trainer` passes the
    adaptive weights in through it every batch. Ignore it and adaptive
    gradient blending silently does nothing.

 4. `evaluate()` / `predict()` in `mixnet/models/base.py` unpack multiple
    outputs ONLY for models whose name starts with 'MixNet' or 'MIN2Net';
    every other model takes the single-output branch, `np.argmax(test_pred,
    axis=1)`, which produces garbage for a list of outputs. Do NOT rename
    your model to sneak into that branch, and do NOT edit `base.py`.
    Override `evaluate()` (and `predict()` if you use it) inside THIS class
    instead -- it is one of the two files you are allowed to change. Copy the
    MixNet branch of `base.py` as your starting point, remembering that
    `Trainer.testing()` hands you a tuple with the classifier output last.

 Multi-task is not required and earns no marks by itself. It earns marks only
 if your report shows what each auxiliary task contributed -- which is an
 ablation: train the same network with and without each extra loss.
-----------------------------------------------------------------------------

Minimal smoke-test architecture -- paste this into `build()` to verify the
plumbing end-to-end BEFORE you design anything clever, then replace it:

        input1  = layers.Input(shape=self.input_shape)
        x       = layers.Flatten()(input1)
        x       = layers.Dense(self.num_class)(x)
        softmax = layers.Activation('softmax', name='softmax')(x)
        model   = Model(inputs=input1, outputs=softmax, name=self.model_name)

Read `mixnet/models/EEGNet.py` and `mixnet/models/DeepConvNet.py` for two
complete, working examples of this same contract.
"""

import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model

from mixnet import models
from mixnet.loss import *


class SpectroTemporalMask(layers.Layer):
    """Training-only random frequency-band + time masking on (batch, Nb, C, T).

    Masking whole sub-bands and contiguous time spans follows SpecAugment
    (Park et al., 2019, "SpecAugment: A Simple Data Augmentation Method for
    Automatic Speech Recognition"). The masked cells are the reconstruction
    targets of the decoder (MAE, He et al., 2022; SimMIM, Xie et al., 2022).

    Returns [x_masked, mask], mask (batch, Nb, 1, T) with 1 = visible, 0 = masked.
    * band mask: each band is dropped independently with prob. `p_band`
      (per sample); at least one band always survives.
    * time mask: `n_tmask` spans per sample, length ~ U{0..t_mask_len},
      random start, shared over bands/channels.
    Masked values are set to 0 (= training mean after standardization), no
    rescaling. At inference, and when p_band=0 and n_tmask=0, it returns
    [x, ones].
    """

    def __init__(self, p_band=0.2, n_tmask=1, t_mask_len=40, **kwargs):
        super().__init__(**kwargs)
        self.p_band = float(p_band)
        self.n_tmask = int(n_tmask)
        self.t_mask_len = int(t_mask_len)

    def _ones(self, x):
        shape = tf.shape(x)
        return tf.ones([shape[0], shape[1], 1, shape[3]], dtype=x.dtype)

    def _sample_mask(self, x):
        shape = tf.shape(x)
        B, Nb, T = shape[0], shape[1], shape[3]
        mask = self._ones(x)
        if self.p_band > 0:
            keep = tf.random.uniform((B, Nb)) >= self.p_band
            # if every band of a sample was dropped, keep one random band
            rescue = tf.cast(tf.one_hot(
                tf.random.uniform((B,), 0, Nb, dtype=tf.int32), Nb), tf.bool)
            keep = tf.where(tf.reduce_any(keep, axis=1, keepdims=True),
                            keep, rescue)
            mask = mask * tf.cast(keep, x.dtype)[:, :, None, None]
        if self.n_tmask > 0 and self.t_mask_len > 0:
            t = tf.range(T)[None, :]                                # (1, T)
            masked = tf.zeros((B, T), dtype=tf.bool)
            for _ in range(self.n_tmask):  # loop over spans, not the batch
                length = tf.random.uniform((B, 1), 0, self.t_mask_len + 1,
                                           dtype=tf.int32)
                length = tf.minimum(length, T)
                start = tf.cast(tf.floor(
                    tf.random.uniform((B, 1)) *
                    tf.cast(T - length + 1, tf.float32)), tf.int32)
                masked = masked | ((t >= start) & (t < start + length))
            mask = mask * (1.0 - tf.cast(masked, x.dtype))[:, None, None, :]
        return mask

    def call(self, x, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        if self.p_band <= 0 and (self.n_tmask <= 0 or self.t_mask_len <= 0):
            return [x, self._ones(x)]

        def masked():
            mask = self._sample_mask(x)
            return x * mask, mask

        def identity():
            return tf.identity(x), self._ones(x)

        if isinstance(training, (bool, int)):
            x_out, mask = masked() if training else identity()
        else:
            x_out, mask = tf.cond(tf.cast(training, tf.bool), masked, identity)
        return [x_out, mask]

    def get_config(self):
        config = super().get_config()
        config.update({'p_band': self.p_band, 'n_tmask': self.n_tmask,
                       't_mask_len': self.t_mask_len})
        return config


class PositionalEmbedding(layers.Layer):
    """Adds a learnable (1, n_tokens, d_model) positional embedding."""

    def __init__(self, n_tokens, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_tokens = int(n_tokens)
        self.d_model = int(d_model)

    def build(self, input_shape):
        self.pos = self.add_weight(
            name='pos_embedding', shape=(1, self.n_tokens, self.d_model),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        return x + self.pos

    def get_config(self):
        config = super().get_config()
        config.update({'n_tokens': self.n_tokens, 'd_model': self.d_model})
        return config


def MaskedMSE():
    """MSE on masked cells only (MAE, He et al., 2022; SimMIM, Xie et al., 2022).

    Returns inner_masked_mse(x, x_rec, mask) with x, x_rec (b, Nb, C, T) and
    mask (b, Nb, 1, T), 1 = visible. If nothing is masked (inference, or
    p_band = n_tmask = 0) it falls back to the full-input MSE.
    """
    def inner_masked_mse(x, x_rec, mask):
        err = tf.square(x - x_rec)                       # (b, Nb, C, T)
        w = (1.0 - mask) * tf.ones_like(err)             # 1 on masked cells
        n = tf.reduce_sum(w)
        return tf.cond(n > 0,
                       lambda: tf.reduce_sum(w * err) / n,
                       lambda: tf.reduce_mean(err))
    return inner_masked_mse


class HWNet(models.base.BaseModel):
    """Multi-task: masked spectro-temporal autoencoding (aux) + classification.

    Outputs [x_rec, mask, softmax] (classifier last), losses
    ['recon', 'crossentropy'] weighted by `loss_weights` = [lambda_rec, 1.0].
    The multi-task step / evaluate pattern follows MixNet (this repository).
    """

    def __init__(self,
                 optimizer,
                 input_shape=(9, 8, 400),
                 num_class=2,
                 loss=[MaskedMSE(), SparseCategoricalCrossentropy()],
                 loss_names=['recon', 'crossentropy'],
                 loss_weights=[0.1, 1.0],
                 model_name='HWNet',
                 data_format='channels_last',
                 **kwargs):
        super().__init__(num_class, loss, loss_names, loss_weights,
                         optimizer, data_format, **kwargs)
        self.input_shape = input_shape
        self.num_class = num_class
        self.model_name = model_name
        self._config(**kwargs)

    # =========================================================================
    #  Architecture hyper-parameters
    # =========================================================================
    def _config(self, **kwargs):
        """HWNet hyper-parameters (all overridable from configs/HWNet.py).

        Input layout is always (Nb, C, T) = (bands, CSP components or
        channels, samples) with `DataLoader` data_format=None. `data_format`
        must still be 'channels_last' because Trainer sets it as the global
        Keras image format; the conv/pool layers also pin it explicitly.
        """
        self.D = 2                 # spatial filters per band
        self.d_model = 32
        self.pool = 16             # temporal pooling -> T/pool tokens (400/16 = 25)
        self.n_layers = 2          # transformer blocks; 0 = no attention (ablation)
        self.n_dec_layers = 2      # decoder transformer blocks (same depth as encoder)
        self.n_heads = 4
        self.key_dim = 8
        self.ffn_mult = 2
        self.dropout_rate = 0.3
        self.p_band = 0.2
        self.n_tmask = 1
        self.t_mask_len = 40
        self.norm_rate = 0.25
        # (lambda_rec is not a model attribute: it is loss_weights[0], set in the config)
        # input_shape = (Nb, C, T), C = CSP components per band (or channels);
        # data_format=None so no depth axis
        self.Nb, self.Chans, self.Samples = self.input_shape

        # Keep this last: it lets the experiment config override the defaults.
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

    # =========================================================================
    #  The network
    # =========================================================================
    def _transformer_block(self, x, prefix=None):
        """One pre-LN Transformer block (Vaswani et al., 2017) on (tokens, d_model).
        Shared by the encoder and the decoder; `prefix` names the layers."""
        name = (lambda n: prefix + n) if prefix else (lambda n: None)
        h = layers.LayerNormalization(epsilon=1e-6, name=name('ln1'))(x)
        h = layers.MultiHeadAttention(num_heads=self.n_heads,
                                      key_dim=self.key_dim,
                                      dropout=self.dropout_rate,
                                      name=name('mha'))(h, h)
        x = layers.Add(name=name('add1'))(
            [x, layers.Dropout(self.dropout_rate, name=name('drop1'))(h)])
        h = layers.LayerNormalization(epsilon=1e-6, name=name('ln2'))(x)
        h = layers.Dense(self.ffn_mult * self.d_model, activation='elu',
                         name=name('ffn1'))(h)
        h = layers.Dense(self.d_model, name=name('ffn2'))(h)
        return layers.Add(name=name('add2'))(
            [x, layers.Dropout(self.dropout_rate, name=name('drop2'))(h)])

    def build(self, print_summary=True, load_weights=False):
        """mask -> shared encoder (per-band spatial filters -> temporal tokens
        -> pre-LN transformer) -> [decoder x_rec, mask, classifier softmax]."""
        self.Nb, self.Chans, self.Samples = self.input_shape
        assert self.Samples % self.pool == 0, (
            'HWNet: n_samples ({}) must be divisible by pool ({})'.format(
                self.Samples, self.pool))
        Nb, C, T = self.Nb, self.Chans, self.Samples
        n_tokens = T // self.pool
        fmt = 'channels_last'  # explicit: Trainer sets the global image format

        input1 = layers.Input(shape=self.input_shape)            # (Nb, C, T)
        x_masked, mask = SpectroTemporalMask(
            self.p_band, self.n_tmask, self.t_mask_len,
            name='spectro_temporal_mask')(input1)

        # ---- shared encoder -------------------------------------------------
        x = layers.Permute((2, 3, 1))(x_masked)                  # (C, T, Nb)
        # Per-band depthwise spatial filter, as in EEGNet
        # (Lawhern et al., 2018, "EEGNet: a compact convolutional neural
        # network for EEG-based brain-computer interfaces").
        x = layers.DepthwiseConv2D((C, 1), use_bias=False,
                                   depth_multiplier=self.D,
                                   depthwise_constraint=max_norm(1.),
                                   data_format=fmt)(x)           # (1, T, Nb*D)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.Conv2D(self.d_model, (1, 1), use_bias=False,
                          data_format=fmt)(x)                    # (1, T, d_model)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.AveragePooling2D((1, self.pool), data_format=fmt)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Reshape((n_tokens, self.d_model))(x)          # (tokens, d_model)
        x = PositionalEmbedding(n_tokens, self.d_model, name='pos_embedding')(x)

        # Pre-LN Transformer encoder blocks (Vaswani et al., 2017, "Attention
        # Is All You Need"); n_layers=0 leaves only pooling (H2 ablation).
        for i in range(self.n_layers):
            x = self._transformer_block(x)
        tokens = layers.LayerNormalization(epsilon=1e-6,
                                           name='encoder_out')(x)

        # ---- classification head --------------------------------------------
        c = layers.GlobalAveragePooling1D()(tokens)
        c = layers.Dropout(self.dropout_rate)(c)
        c = layers.Dense(self.num_class,
                         kernel_constraint=max_norm(self.norm_rate))(c)
        softmax = layers.Activation('softmax', name='softmax')(c)

        # ---- reconstruction head (small decoder, mirrors the encoder) -------
        # Own positional embedding + n_dec_layers pre-LN transformer blocks
        # (as in MAE's lightweight decoder), then undo pooling and spatial
        # filtering with two Dense maps. All decoder layers are named 'dec_*'.
        r = PositionalEmbedding(n_tokens, self.d_model,
                                name='dec_pos_embedding')(tokens)
        for i in range(self.n_dec_layers):
            r = self._transformer_block(r, prefix='dec_blk{}_'.format(i + 1))
        r = layers.LayerNormalization(epsilon=1e-6, name='dec_norm')(r)
        r = layers.Dense(self.pool * Nb * self.D,
                         name='dec_unpool')(r)                   # (tokens, pool*Nb*D)
        r = layers.Reshape((T, Nb * self.D))(r)                  # (T, Nb*D)
        r = layers.Dense(Nb * C, name='dec_unspatial')(r)        # (T, Nb*C)
        r = layers.Reshape((T, Nb, C))(r)
        r = layers.Permute((2, 3, 1))(r)                         # (Nb, C, T)
        x_rec = layers.Activation('linear', name='recon')(r)
        mask = layers.Activation('linear', name='mask')(mask)

        model = Model(inputs=input1, outputs=[x_rec, mask, softmax],
                      name=self.model_name)
        if print_summary:
            model.summary()
        if load_weights:
            print('loading weights from', self.weights_dir)
            model.load_weights(self.weights_dir)
        return model

    # =========================================================================
    #  Multi-task training / evaluation steps (pattern from MixNet.py).
    #  losses follow loss_names = ['recon', 'crossentropy']; accuracies come
    #  from the classifier output only.
    # =========================================================================
    def _losses(self, x, y, x_rec, mask, y_logis):
        recon_loss = self.loss.recon(x, x_rec, mask)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        return [recon_loss, crossentropy_loss]

    @tf.function
    def train_step(self, x, y, loss_weights):
        with tf.GradientTape() as tape:
            x_rec, mask, y_logis = self.model(x, training=True)
            losses = self._losses(x, y, x_rec, mask, y_logis)
            train_loss = tf.reduce_sum(loss_weights * tf.stack(losses))
            self.train_acc_metric.update_state(y, y_logis)
        grads = tape.gradient(train_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        logs = dict({'train_loss': train_loss})
        logs.update(dict(zip(['train_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'train_acc': self.train_acc_metric.result()}))
        return logs, (x_rec, mask, y_logis)

    @tf.function
    def val_step(self, x, y, loss_weights):
        x_rec, mask, y_logis = self.model(x, training=False)
        losses = self._losses(x, y, x_rec, mask, y_logis)
        val_loss = tf.reduce_sum(loss_weights * tf.stack(losses))
        self.val_acc_metric.update_state(y, y_logis)
        logs = dict({'val_loss': val_loss})
        logs.update(dict(zip(['val_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'val_acc': self.val_acc_metric.result()}))
        return logs, (x_rec, mask, y_logis)

    @tf.function
    def test_step(self, x, y, loss_weights):
        x_rec, mask, y_logis = self.model(x, training=False)
        losses = self._losses(x, y, x_rec, mask, y_logis)
        test_loss = tf.reduce_sum(loss_weights * tf.stack(losses))
        self.test_acc_metric.update_state(y, y_logis)
        logs = dict({'test_loss': test_loss})
        logs.update(dict(zip(['test_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'test_acc': self.test_acc_metric.result()}))
        return logs, (x_rec, mask, y_logis)

    @tf.function
    def pred_step(self, x):
        x_rec, mask, y_logis = self.model(x, training=False)
        return (x_rec, mask, y_logis)

    # =========================================================================
    #  evaluate() / predict(): BaseModel unpacks multi-output predictions only
    #  for models named MixNet/MIN2Net, so unpack [x_rec, mask, softmax] here.
    #  x_rec is not stored in Y (it would bloat S*_prediction_results.npy).
    # =========================================================================
    def evaluate(self, X_test, y_test):
        model = self.build(print_summary=self.print_summary, load_weights=True)
        self.compile(model=model)
        start = time.time()
        evaluation, test_pred = self.testing(x=X_test, y=y_test)
        end = time.time()
        x_rec, mask, y_pred_clf = test_pred            # classifier last
        y_pred_argm = np.argmax(y_pred_clf, axis=1)
        Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf': y_pred_clf}

        mem_usage = tf.config.experimental.get_memory_info('GPU:0')['current']
        print("Checking average current GPU memory usage", mem_usage)
        print("F1-score is computed based on {}".format(self.f1_average))
        f1 = f1_score(y_test, y_pred_argm, average=self.f1_average)
        print(classification_report(y_test, y_pred_argm))
        evaluation.update({'f1-score': f1, 'prediction_time': end - start,
                           'memory_usage': mem_usage})
        evaluation.update(dict(zip(['w_' + name + '_loss' for name in self.loss_names],
                                   self.best_loss_weights.numpy())))
        return Y, evaluation

    def predict(self, X_test, y_test):
        model = self.build(print_summary=self.print_summary, load_weights=True)
        self.compile(model=model)
        x_rec, mask, y_pred_clf = self.prediction(x=X_test)
        y_pred_argm = np.argmax(y_pred_clf, axis=1)
        return {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf': y_pred_clf}
