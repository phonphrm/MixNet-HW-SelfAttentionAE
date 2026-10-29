import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = sys.argv[1]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
W, H = 1810, 565
fig = plt.figure(figsize=(13, 13 * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

STY = {  # fill, edge, linestyle
    'data':      ('#FFFFFF', '#374151', '-'),
    'stateless': ('#F3F4F6', '#6B7280', '-'),
    'fitted':    ('#FDEBC8', '#B45309', '-'),
    'learned':   ('#DCE8FB', '#1D4ED8', '-'),
    'trainonly': ('#F3EEFE', '#6D28D9', (0, (4, 2.5))),
    'loss':      ('#FCE4EE', '#BE185D', '-'),
}
INK, MUTED = '#111827', '#374151'
T1, T2 = 10.2, 8.5

def box(x, y, w, h, title, sub=None, kind='learned'):
    fc, ec, ls = STY[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=9',
                                fc=fc, ec=ec, lw=1.4, ls=ls, zorder=2))
    lines = sub if isinstance(sub, list) else ([sub] if sub else [])
    n = 1 + len(lines); step = 21
    y0 = y + h / 2 - (n - 1) * step / 2
    ax.text(x + w / 2, y0, title, ha='center', va='center', fontsize=T1, weight='bold', color=INK, zorder=3)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y0 + (i + 1) * step, s, ha='center', va='center', fontsize=T2, color=MUTED, zorder=3)

def arrow(pts, color='#4B5563', lw=1.4, ls='-'):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle='butt')
    ax.annotate('', xy=pts[-1], xytext=pts[-2], zorder=1,
                arrowprops=dict(arrowstyle='-|>,head_length=0.55,head_width=0.3', color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0))

def label(x, y, s, size=T2, color=MUTED, **kw):
    ax.text(x, y, s, fontsize=size, color=color, zorder=3, **kw)

# ---------------- A: preprocessing ----------------
label(30, 26, 'A   Preprocessing (offline, per subject and fold)', size=12.5, color=INK, weight='bold', va='center')
ya, ha = 48, 84
box(30, ya, 280, ha, 'Raw EEG trial', ['20 channels × 400 samples', '4 s at 100 Hz'], 'data')
box(370, ya, 280, ha, 'Filter bank', ['9 bands, 4–40 Hz (4 Hz wide)', 'Butterworth, zero-phase'], 'stateless')
box(710, ya, 280, ha, 'Per-band CSP', ['8 spatial filters per band', 'fit on training fold only'], 'fitted')
box(1050, ya, 280, ha, 'Standardize', ['mean / std per (band, comp.)', 'fit on training fold only'], 'fitted')
box(1390, ya, 280, ha, r'$\mathbf{X}$  (9 × 8 × 400)', ['bands × CSP comps. × time', 'deterministic, saved'], 'data')
for x0, x1 in [(310, 370), (650, 710), (990, 1050), (1330, 1390)]:
    arrow([(x0, ya + ha / 2), (x1, ya + ha / 2)])

# ---------------- B: model ----------------
label(30, 196, 'B   HWNet: shared encoder, classification head and masked-reconstruction decoder', size=12.5, color=INK, weight='bold', va='center')
yb, hb = 250, 78
cy = yb + hb / 2
# input X -> mask
arrow([(1530, ya + ha), (1530, 168), (140, 168), (140, yb)])

box(30, yb, 220, hb, 'Band + time mask', ['drop bands (p = 0.2)', 'zero a time span (≤ 40)'], 'trainonly')
# encoder group (left -> right)
ax.add_patch(Rectangle((280, 222), 830, 130, fc='none', ec='#93A3B8', lw=1.1, ls=(0, (2, 2)), zorder=0))
label(1102, 238, 'shared encoder', size=T2, color='#475569', ha='right', va='center', style='italic')
enc = [(295, 175, 'Spatial filter', ['depthwise, per band', 'D = 2 → 18 maps']),
       (490, 130, '1×1 Conv', ['d = 32', 'BN, ELU']),
       (640, 130, 'Avg pool', ['÷16 in time', '→ 25 tokens']),
       (790, 130, 'Positional', ['embedding', '(learnable)']),
       (940, 155, 'Transformer', ['2 pre-LN blocks', '4 heads'])]
for x, w, t, sub in enc:
    box(x, yb, w, hb, t, sub, 'learned')
arrow([(250, cy), (295, cy)])
for (x0, w0, *_), (x1, *_r) in zip(enc[:-1], enc[1:]):
    arrow([(x0 + w0, cy), (x1, cy)])
box(1140, yb, 110, hb, 'Tokens', ['25 × 32'], 'data')
arrow([(1095, cy), (1140, cy)])

# classification head (same row)
box(1290, yb, 290, hb, 'Classification head', ['avg-pool → Dense(2) → softmax'], 'learned')
box(1615, yb, 170, hb, 'Cross-entropy', ['weight 1.0'], 'loss')
arrow([(1250, cy), (1290, cy)])
arrow([(1580, cy), (1615, cy)])

# decoder group (right -> left, mirroring the encoder)
yd = 400; dy = yd + hb / 2
ax.add_patch(Rectangle((430, 372), 680, 130, fc='none', ec='#93A3B8', lw=1.1, ls=(0, (2, 2)), zorder=0))
label(1102, 488, 'decoder', size=T2, color='#475569', ha='right', va='center', style='italic')
dec = [(955, 140, 'Positional', ['embedding', '(learnable)']),
       (785, 150, 'Transformer', ['2 pre-LN blocks', '4 heads']),
       (615, 150, 'Un-pool', ['Dense: 25 → 400', 'time steps']),
       (445, 150, 'Un-spatial', ['Dense → 9 × 8', 'bands × comps.'])]
for x, w, t, sub in dec:
    box(x, yd, w, hb, t, sub, 'learned')
arrow([(1195, yb + hb), (1195, dy), (1095, dy)])
for (x0, *_), (x1, w1, *_r) in zip(dec[:-1], dec[1:]):
    arrow([(x0, dy), (x1 + w1, dy)])
box(280, yd, 130, hb, r'$\hat{\mathbf{X}}$', ['9 × 8 × 400'], 'data')
arrow([(445, dy), (410, dy)])
box(30, yd, 220, hb, 'Masked MSE', ['hidden cells only', r'target: clean $\mathbf{X}$, λ = 0.1'], 'loss')
arrow([(280, dy), (250, dy)])
# mask -> masked MSE
arrow([(140, yb + hb), (140, yd)], color='#6D28D9', ls=(0, (4, 2.5)))
label(150, 364, 'mask', color='#6D28D9', va='center')

# objective
label(1290, 420, r'$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda\,\mathcal{L}_{\mathrm{rec}}$', size=12.5, color=INK, va='center')
label(1290, 448, 'checkpoint, LR schedule and early stopping', va='center')
label(1290, 468, 'use validation cross-entropy only', va='center')

# legend
lx, ly = 30, 540
for i, (kind, txt) in enumerate([('stateless', 'stateless'), ('fitted', 'fit on training fold only'),
                                 ('learned', 'learned end to end'), ('trainonly', 'active only in training'),
                                 ('loss', 'loss')]):
    x = lx + [0, 150, 410, 625, 860][i]
    fc, ec, ls = STY[kind]
    ax.add_patch(FancyBboxPatch((x, ly - 9), 30, 18, boxstyle='round,pad=0,rounding_size=4', fc=fc, ec=ec, lw=1.3, ls=ls))
    label(x + 40, ly, txt, va='center')

for ext in ('pdf', 'png', 'svg'):
    fig.savefig('%s.%s' % (OUT, ext), dpi=220 if ext == 'png' else None, facecolor='white')
print('saved')
