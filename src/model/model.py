# -*- coding: utf-8 -*-
"""
HRV 분류 & 전체 그래프 출력 (5-Fold + 8:2 Holdout + Real/Synth 파형)
- 윈도우링 HRV 시퀀스 생성, KDE 합성으로 클래스 밸런스 확보
- 최소 200 epoch 보장 + EarlyStopping(그 이후만 중단)
- Fold/Holdout: Confusion, ROC/PR/Calibration, Loss, YoudenJ 저장
- Real/Synth × High/Low: 3-패널(PPG, Time-HRV, LF/HF) 시계열 저장
"""

import os, re, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from scipy.signal import lombscargle
from scipy import integrate
from scipy.stats import gaussian_kde

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
class Config:
    BASE_DIR = Path(r"C:\Users\hnryu\Desktop\python work")
    DATA_DIR = BASE_DIR / "data"                  # PPG_*.txt (ppg, ibi)
    RESULTS_FILE = BASE_DIR / "nback_results.csv" # filename, accuracy
    OUT = BASE_DIR / "paper_outputs_final"

    # 윈도우링
    WIN_SEC    = 60
    STEP_SEC   = 4
    TIME_STEPS = 12
    MIN_WIN_SAMPLES = 10

    # 합성 목표치(High/Low)
    TARGET_PER_CLASS = 300
    LOW_MIN = 200

    # 학습
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 400
    MIN_EPOCHS = 200   # ⬅ 최소 200 epoch 보장
    LR = 3e-4
    BATCH_SIZE = 64
    DROPOUT = 0.30
    ES_PATIENCE = 40
    WARMUP_EPOCHS = 10
    LABEL_SMOOTH = 0.0

    # 그림 옵션
    CONF_BLUE_SAT = 0.85  # 혼돈행렬 컬러 채도

cfg = Config()
cfg.OUT.mkdir(parents=True, exist_ok=True)

# 재현성
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==================== 유틸 & 파서 ====================
def _norm_name(s: str) -> str:
    s = str(s).strip().lower()
    if s.endswith(".txt"): s = s[:-4]
    return s

def _subject_key(s: str) -> str:
    m = re.search(r"(ppg_sub\d+)", s.lower())
    return m.group(1) if m else s.lower()

def _series_to_float(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip()
    s = s.str.replace('−', '-', regex=False).str.replace(',', '.', regex=False)
    float_pat = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    sub_pat   = rf'^\s*({float_pat})\s*-\s*({float_pat})\s*$'
    num_pat   = rf'^\s*({float_pat})\s*$'
    def parse_cell(x: str) -> float:
        m = re.match(num_pat, x)
        if m: return float(m.group(1))
        if 'e-' not in x.lower():
            m2 = re.match(sub_pat, x)
            if m2: return float(m2.group(1)) - float(m2.group(2))
        return np.nan
    return s.apply(parse_cell).to_numpy(dtype=float)

# ==================== HRV 특징 ====================
def get_all_hrv_features(ibi_ms: np.ndarray) -> np.ndarray:
    """[meanRR, SDNN, RMSSD, LF/HF]"""
    ibi_ms = np.asarray(ibi_ms, dtype=float)
    ibi_ms = ibi_ms[np.isfinite(ibi_ms)]
    ibi_ms = ibi_ms[ibi_ms > 0]
    if len(ibi_ms) < 10:
        return np.full(4, np.nan, np.float32)

    mean_rr = float(np.mean(ibi_ms))
    sdnn    = float(np.std(ibi_ms, ddof=1))
    rmssd   = float(np.sqrt(np.mean(np.diff(ibi_ms)**2)))

    t = np.cumsum(ibi_ms) / 1000.0
    t -= t[0]
    if t[-1] <= 0:
        return np.full(4, np.nan, np.float32)

    freqs_hz = np.linspace(0.01, 0.5, 500)
    ang = 2.0*np.pi*freqs_hz
    pgram = lombscargle(t, ibi_ms - mean_rr, ang, normalize=True)

    lf_mask = (freqs_hz >= 0.04) & (freqs_hz < 0.15)
    hf_mask = (freqs_hz >= 0.15) & (freqs_hz < 0.40)
    lf = float(integrate.simpson(pgram[lf_mask], freqs_hz[lf_mask])) if lf_mask.any() else 0.0
    hf = float(integrate.simpson(pgram[hf_mask], freqs_hz[hf_mask])) if hf_mask.any() else 0.0
    lfhf = (lf / hf) if hf > 1e-6 else 0.0

    return np.array([mean_rr, sdnn, rmssd, lfhf], dtype=np.float32)

def _normalize_accuracy_column(acc_col: pd.Series) -> np.ndarray:
    s = acc_col.astype(str).str.strip().str.replace('%','', regex=False)
    vals = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
    vals = np.where(vals > 1.0, vals/100.0, vals)
    return np.clip(vals, 0.0, 1.0)

def _label_from_accuracy(acc_norm: np.ndarray) -> np.ndarray:
    lbl = (acc_norm >= 0.5).astype(int)
    if lbl.mean() in (0.0, 1.0):  # 단일 클래스 폴백
        thr = float(np.median(acc_norm))
        lbl = (acc_norm >= thr).astype(int)
    if lbl.mean() > 0.6:  # High 과다
        thr = float(np.quantile(acc_norm, 0.6))
        lbl = (acc_norm >= thr).astype(int)
    if (1.0 - lbl.mean()) > 0.6:  # Low 과다
        thr = float(np.quantile(acc_norm, 0.4))
        lbl = (acc_norm >= thr).astype(int)
    return lbl

# ==================== 데이터 로딩 & 합성 ====================
def load_real_data():
    print(f"[*] STM32 데이터 로딩 시작... 경로: '{cfg.DATA_DIR}'")
    if not cfg.RESULTS_FILE.exists():
        raise RuntimeError(f"결과 파일 없음: {cfg.RESULTS_FILE}")

    df_res = pd.read_csv(cfg.RESULTS_FILE, engine="python")
    if "filename" not in df_res.columns:
        df_res = df_res.rename(columns={df_res.columns[0]: "filename"})
    if "accuracy" not in df_res.columns:
        raise RuntimeError("nback_results.csv에 'accuracy' 컬럼이 필요합니다.")

    df_res["key_full"] = df_res["filename"].map(_norm_name)
    df_res["key_subj"] = df_res["filename"].map(_subject_key)
    acc_norm = _normalize_accuracy_column(df_res["accuracy"])
    df_res["acc_norm"] = acc_norm
    df_res["label"] = _label_from_accuracy(acc_norm)  # High=1, Low=0

    txt_files = sorted(cfg.DATA_DIR.glob("PPG_*.txt"))
    all_sequences, all_labels, all_groups = [], [], []

    for f_path in tqdm(txt_files, desc="파일 처리"):
        base_full = _norm_name(f_path.name)
        base_subj = _subject_key(f_path.name)

        hit = df_res[df_res["key_full"] == base_full]
        if hit.empty:
            hit = df_res[df_res["key_subj"] == base_subj]
        if hit.empty:
            continue
        label = int(hit["label"].iloc[0])

        try:
            raw = pd.read_csv(
                f_path, header=None, names=["ppg", "ibi"],
                on_bad_lines="skip", dtype=str, engine="python"
            )
        except Exception:
            continue

        ppg = _series_to_float(raw["ppg"])
        ibi = _series_to_float(raw["ibi"])
        mask = np.isfinite(ppg) & np.isfinite(ibi) & (ibi > 0)
        if mask.sum() < 30:
            continue
        ibi = ibi[mask]

        # 슬라이딩 HRV 시퀀스
        t = np.cumsum(ibi)/1000.0
        t -= t[0]
        feats = []
        start = 0.0
        while start + cfg.WIN_SEC < t[-1]:
            idx = np.where((t >= start) & (t < start + cfg.WIN_SEC))[0]
            if len(idx) >= cfg.MIN_WIN_SAMPLES:
                f = get_all_hrv_features(ibi[idx])
                if np.isfinite(f).all():
                    feats.append(f)
            start += cfg.STEP_SEC

        if len(feats) >= cfg.TIME_STEPS:
            F = np.array(feats, dtype=np.float32)
            for i in range(0, len(F) - cfg.TIME_STEPS + 1):
                seq = F[i:i+cfg.TIME_STEPS]
                if np.isfinite(seq).all():
                    all_sequences.append(seq)
                    m = re.search(r"sub(\d+)", base_subj)
                    gid = int(m.group(1)) if m else -1
                    all_groups.append(gid)
                    all_labels.append(label)

    X = np.array(all_sequences, np.float32)
    y = np.array(all_labels,   np.int32)
    g = np.array(all_groups,   np.int32)
    print(f"[*] 데이터 로딩 완료! 원본 시퀀스: {len(X)}개 / High={int((y==1).sum())}, Low={int((y==0).sum())}")
    return X, y, g

def _kde_fit(samples: np.ndarray):
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if len(samples) < 5:
        mu = samples.mean(axis=0)
        cov = np.cov(samples.T) if len(samples) > 1 else np.eye(samples.shape[1])*1e-3
        return ("gauss", (mu, cov))
    return ("kde", gaussian_kde(samples.T))

def _kde_sample(model, n: int):
    kind, obj = model
    if kind == "kde":
        return obj.resample(n).T.astype(np.float32)
    mu, cov = obj
    return np.random.multivariate_normal(mu, cov + np.eye(len(mu))*1e-6, size=n).astype(np.float32)

def synthesize_balanced(X, y, g, target_per_class, low_min):
    T, F = X.shape[1], X.shape[2]
    X_new, y_new, g_new = [X.copy()], [y.copy()], [g.copy()]
    next_gid = -1

    for cls in (1, 0):
        idx = np.where(y == cls)[0]
        need = target_per_class - len(idx)
        if cls == 0:
            need = max(need, low_min - len(idx))
        if need <= 0: continue

        pool = X[idx].reshape(-1, F)
        model = _kde_fit(pool)

        synth_list, synth_gid = [], []
        for _ in range(int(need)):
            seq = _kde_sample(model, T)
            for t in range(1, T):  # 시간 스무딩
                seq[t] = 0.6*seq[t] + 0.4*seq[t-1]
            # 물리 제약
            seq[:,0] = np.clip(seq[:,0], 300, 1800)
            seq[:,1] = np.clip(seq[:,1],   5,  400)
            seq[:,2] = np.clip(seq[:,2],   5,  400)
            seq[:,3] = np.clip(seq[:,3],   0,    5)
            synth_list.append(seq.astype(np.float32))
            synth_gid.append(next_gid); next_gid -= 1

        if synth_list:
            X_new.append(np.stack(synth_list))
            y_new.append(np.full(len(synth_list), cls, np.int32))
            g_new.append(np.asarray(synth_gid, dtype=np.int32))

    Xb = np.concatenate(X_new, axis=0)
    yb = np.concatenate(y_new, axis=0)
    gb = np.concatenate(g_new, axis=0)
    print(f"[*] 합성 후 개수: High={int((yb==1).sum())}, Low={int((yb==0).sum())}, 총={len(Xb)}")
    return Xb, yb, gb

# ==================== Dataset & Model ====================
class HRVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # (N,1,T,F)
        eps = cfg.LABEL_SMOOTH
        y = y.astype(np.float32).reshape(-1, 1)
        y = y*(1-2*eps) + eps
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class CNN_Transformer(nn.Module):
    def __init__(self, d_model=128, n_heads=8, num_layers=3, drop=cfg.DROPOUT):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding=(1,1)),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, (3,3), padding=(1,1)),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Dropout2d(drop)
        )
        self.patch_len = 2; self.stride = 1; self.conv_out_ch = 64
        patch_dim = self.patch_len * 4 * self.conv_out_ch
        self.patch_embedding = nn.Linear(patch_dim, d_model)
        num_patches = (cfg.TIME_STEPS - self.patch_len)//self.stride + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
        enc_layers = []
        for _ in range(num_layers):
            enc_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=drop, batch_first=True, activation="gelu"))
        self.encoder = nn.Sequential(*enc_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        z = self.conv(x)               # (B,64,T,F)
        B,C,T,F = z.shape
        patches = z.unfold(2, self.patch_len, self.stride)     # (B,C,num_p,patch_len,F)
        patches = patches.permute(0,2,1,3,4).contiguous()      # (B,num_p,C,patch_len,F)
        patches = patches.reshape(B, patches.shape[1], C*self.patch_len*F)
        tokens = self.patch_embedding(patches)                 # (B,num_p,d)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embedding
        h = self.encoder(tokens)
        return self.head(h[:,0])

# ==================== 학습/평가 루틴 ====================
def cosine_with_warmup(optimizer, num_epochs, warmup_epochs):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep+1)/max(1,warmup_epochs)
        prog = (ep - warmup_epochs)/max(1, num_epochs - warmup_epochs)
        return 0.5*(1.0 + math.cos(math.pi*prog))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, loader, opt, scaler, pos_weight, use_amp):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(cfg.DEVICE.type, enabled=use_amp, dtype=(torch.float16 if cfg.DEVICE.type=="cuda" else torch.bfloat16)):
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total += loss.item()*len(xb)
    return total/len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, use_amp, pos_weight=None):
    model.eval()
    all_p, all_t, total = [], [], 0.0
    for xb, yb in loader:
        xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)
        with torch.autocast(cfg.DEVICE.type, enabled=use_amp, dtype=(torch.float16 if cfg.DEVICE.type=="cuda" else torch.bfloat16)):
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()  # 확률
        total += loss.item()*len(xb)
        all_p.append(probs); all_t.append(yb.cpu().numpy().ravel())
    y_p = np.concatenate(all_p); y_t = np.concatenate(all_t)
    return total/len(loader.dataset), y_t, y_p, (y_p>=0.5).astype(int)

# ==================== 시각화 ====================
def _blue_cmap(sat: float = 0.85):
    sat = float(np.clip(sat, 0.0, 1.0))
    base = plt.cm.Blues(np.linspace(0, 1, 256))
    white = np.ones_like(base)
    mixed = (1 - sat)*white + sat*base
    return LinearSegmentedColormap.from_list("BluesSaturated", mixed)

def save_confusion(out_path, title, cm):
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest', cmap=_blue_cmap(cfg.CONF_BLUE_SAT))
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    classes = ["Low group", "High group"]
    tick = np.arange(len(classes))
    plt.xticks(tick, classes); plt.yticks(tick, classes)
    th = cm.max()/2 if cm.size>0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,format(cm[i,j],'d'),ha='center',va='center',
                     color="white" if cm[i,j]>th else "black")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(out_path, dpi=180); plt.close()

def save_roc_pr_cal(out_path, y_true, y_prob, title_suffix=""):
    fpr,tpr,_ = roc_curve(y_true, y_prob); roc_auc = auc(fpr,tpr)
    pr,rc,_ = precision_recall_curve(y_true, y_prob); pr_auc = auc(rc,pr)
    pt,pp = calibration_curve(y_true, y_prob, n_bins=10)
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,3,1); ax1.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}"); ax1.plot([0,1],[0,1],'--'); ax1.set_title("ROC"+title_suffix); ax1.legend(); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax2 = plt.subplot(1,3,2); ax2.plot(rc,pr,label=f"AUC={pr_auc:.3f}"); ax2.set_title("Precision-Recall"+title_suffix); ax2.legend(); ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax3 = plt.subplot(1,3,3); ax3.plot(pp,pt,marker='o'); ax3.plot([0,1],[0,1],'--'); ax3.set_title("Calibration"+title_suffix); ax3.set_xlabel("Predicted"); ax3.set_ylabel("Observed")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()

def save_youden_bar(out_path, y_true, y_prob, acc, f1, roc_auc, fold_name):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    J = tpr - fpr
    j_idx = int(np.argmax(J)) if len(J)>0 else 0
    best_thr = float(thr[j_idx]) if j_idx < len(thr) else 0.5
    best_J   = float(J[j_idx]) if len(J) > 0 else 0.0
    plt.figure(figsize=(6,4)); names = ["AUC","ACC","F1","YoudenJ"]; vals = [roc_auc, acc, f1, best_J]
    plt.bar(names, vals); plt.ylim(0,1.0); plt.title(f"YoudenJ & Metrics ({fold_name})")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()
    return best_thr, best_J

def export_predictions(out_path_csv, fold_name, y_true, y_prob):
    pd.DataFrame({"fold":fold_name,"y_true":y_true.astype(int),"y_prob":y_prob,"y_pred":(y_prob>=0.5).astype(int)}).to_csv(out_path_csv, index=False)

def save_boxplot_real_vs_synth(out_dir, X, y, synth_mask):
    T,F = X.shape[1], X.shape[2]
    feats = ["meanRR","SDNN","RMSSD","LF/HF"]
    Xf = X.reshape(-1, F)
    yf = np.repeat(y, T)
    sf = np.repeat(synth_mask, T)
    for label,name in [(0,"Low"),(1,"High")]:
        data=[]; labels=[]
        for j in range(F):
            real_vals  = Xf[(yf==label)&(sf==0),j]
            synth_vals = Xf[(yf==label)&(sf==1),j]
            data.extend([real_vals, synth_vals]); labels.extend([f"{feats[j]} (real)", f"{feats[j]} (synth)"])
        plt.figure(figsize=(12,6)); plt.boxplot(data, showfliers=False); plt.xticks(np.arange(1,len(labels)+1), labels, rotation=45, ha='right')
        plt.title(f"Feature Distribution — {name} group (real vs synth)")
        plt.tight_layout(); plt.savefig(out_dir/f"boxplot_real_vs_synth_{name}.jpg", dpi=200); plt.close()

# ---------- 보기 쉬운 3-패널 시계열(PPG, Time-HRV, LF/HF) ----------
def _sliding_hrv_from_ibi(ibi_ms: np.ndarray):
    t = np.cumsum(ibi_ms)/1000.0; t -= t[0]
    feats, times = [], []
    start = 0.0
    while start + cfg.WIN_SEC < t[-1]:
        idx = np.where((t >= start) & (t < start + cfg.WIN_SEC))[0]
        if len(idx) >= cfg.MIN_WIN_SAMPLES:
            f = get_all_hrv_features(ibi_ms[idx])
            if np.isfinite(f).all():
                feats.append(f); times.append(start + cfg.WIN_SEC/2)
        start += cfg.STEP_SEC
    return (np.array(times), np.array(feats)) if len(feats)>0 else (None, None)

def save_three_panel_ppg_hrv(out_path: Path, title: str, ppg: np.ndarray, ibi_ms: np.ndarray):
    times, feats = _sliding_hrv_from_ibi(ibi_ms)
    if times is None: return
    # feats: [meanRR, SDNN, RMSSD, LF/HF]
    plt.figure(figsize=(12,7.5))
    # (a) Raw PPG
    n_ppg = len(ppg)
    ax1 = plt.subplot(3,1,1)
    ax1.plot(np.arange(n_ppg), ppg)
    ax1.set_title("(a) Raw PPG Signal"); ax1.set_xlabel("Sample"); ax1.set_ylabel("Amplitude")
    # (b) Time HRV
    ax2 = plt.subplot(3,1,2)
    ax2.plot(times, feats[:,0], label="Mean RR (ms)")
    ax2.plot(times, feats[:,1], label="SDNN (ms)")
    ax2.plot(times, feats[:,2], label="RMSSD (ms)")
    ax2.set_title("(b) Sliding Time-domain HRV"); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("ms"); ax2.legend(loc="upper right")
    # (c) LF/HF
    ax3 = plt.subplot(3,1,3)
    ax3.plot(times, feats[:,3], marker='s', linestyle='--', label="LF/HF Ratio")
    ax3.set_title("(c) Sliding Frequency-domain HRV"); ax3.set_xlabel("Time (s)"); ax3.set_ylabel("LF/HF"); ax3.legend(loc="upper right")
    plt.suptitle(title, y=0.99, fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.97]); plt.savefig(out_path, dpi=200); plt.close()

# ---------- Real/Synth 샘플 그래프 ----------
def _first_valid_by_label(target_label, df_res):
    files = sorted(cfg.DATA_DIR.glob("PPG_*.txt"))
    for f in files:
        base_full = _norm_name(f.name)
        base_subj = _subject_key(f.name)
        hit = df_res[(df_res["key_full"]==base_full) | (df_res["key_subj"]==base_subj)]
        if hit.empty: continue
        if int(hit["label"].iloc[0]) != target_label: continue
        try:
            raw = pd.read_csv(f, header=None, names=["ppg","ibi"], on_bad_lines="skip", dtype=str, engine="python")
            ppg = _series_to_float(raw["ppg"]); ibi = _series_to_float(raw["ibi"])
            mask = np.isfinite(ppg) & np.isfinite(ibi) & (ibi > 0)
            if mask.sum() >= 30:
                return f, ppg[mask], ibi[mask]
        except Exception:
            continue
    return None, None, None

def _synthesize_ibi_from_hrv_step(mean_rr: float, sdnn: float, n_beats: int = 80):
    sd = max(5.0, float(sdnn))
    rr = np.random.normal(loc=float(mean_rr), scale=sd, size=int(max(20, n_beats)))
    rr = np.clip(rr, 200.0, 2000.0)
    return rr

def _synthesize_ppg_from_ibi(ibi_ms: np.ndarray, fs: int = 50):
    if len(ibi_ms) < 5: return np.zeros(1)
    beat_times = np.cumsum(ibi_ms)/1000.0
    T = beat_times[-1]; t = np.arange(0, T, 1.0/fs)
    ppg = np.zeros_like(t); sigma = 0.05
    for bt in beat_times: ppg += np.exp(-0.5*((t - bt)/sigma)**2)
    return ppg

def save_real_synth_timeseries_by_group(out_dir_main: Path, X_bal: np.ndarray, y_bal: np.ndarray, synth_mask: np.ndarray, df_res: pd.DataFrame):
    # Real High/Low
    for lab, name in [(1, "High"), (0, "Low")]:
        f, ppg, ibi = _first_valid_by_label(lab, df_res)
        if f is not None:
            save_three_panel_ppg_hrv(out_dir_main / f"real_{name.lower()}_timeseries.jpg",
                                     f"[Real Subject]  Data Analysis for {f.name}", ppg, ibi)

    # Synth High/Low (HRV 시퀀스 하나를 IBI/PPG로 근사)
    for lab, name in [(1, "High"), (0, "Low")]:
        synth_idx = np.where((synth_mask==1) & (y_bal==lab))[0]
        if len(synth_idx) == 0: continue
        idx = int(synth_idx[0])
        hrv_synth = X_bal[idx]
        mean_rr_ms = float(np.clip(np.nanmean(hrv_synth[:,0]), 300, 1800))
        beats_10s = int(max(20, 10000.0/mean_rr_ms))
        ibi_surr = []
        for t in range(hrv_synth.shape[0]):
            ibi_surr.append(_synthesize_ibi_from_hrv_step(hrv_synth[t,0], hrv_synth[t,1], n_beats=beats_10s))
        ibi_surr = np.concatenate(ibi_surr, axis=0)
        ppg_surr = _synthesize_ppg_from_ibi(ibi_surr, fs=50)
        save_three_panel_ppg_hrv(out_dir_main / f"synth_{name.lower()}_timeseries.jpg",
                                 f"[Synthetic {'High-Performer' if lab==1 else 'Low-Performer'}]  Data Analysis",
                                 ppg_surr, ibi_surr)

# ==================== 메인 학습 파이프라인 ====================
def run_folds_and_holdout(X, y, out_dir: Path):
    # 스케일링(특징 축 기준)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # 5-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
    summary_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_s, y), 1):
        print(f"\n===== FOLD {fold} =====")

        tr_ds, va_ds = HRVDataset(X_s[tr_idx], y[tr_idx]), HRVDataset(X_s[va_idx], y[va_idx])
        tr_dl = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

        model = CNN_Transformer().to(cfg.DEVICE)
        opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
        sched = cosine_with_warmup(opt, cfg.EPOCHS, cfg.WARMUP_EPOCHS)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=(cfg.DEVICE.type=="cuda"))

        # pos_weight
        pos = (y[tr_idx]==1).sum(); neg = (y[tr_idx]==0).sum()
        pw = torch.tensor([neg/max(1,pos)], device=cfg.DEVICE) if (pos>0 and neg>0) else None

        best_auc, best_state, patience = 0.0, None, 0
        tr_losses, va_losses = [], []
        use_amp = True

        for epoch in range(cfg.EPOCHS):
            tl = train_one_epoch(model, tr_dl, opt, scaler_amp, pw, use_amp)
            vl, y_t, y_p, _ = evaluate(model, va_dl, use_amp, pos_weight=pw)
            tr_losses.append(tl); va_losses.append(vl)
            auc_val = roc_auc_score(y_t.astype(int), y_p) if len(np.unique(y_t))>1 else 0.5
            sched.step()

            if auc_val > best_auc:
                best_auc = auc_val; best_state = {k:v.cpu() for k,v in model.state_dict().items()}; patience=0
            else:
                patience += 1

            # ⬇ 최소 epoch 보장 후에만 early stop
            if (epoch + 1) >= cfg.MIN_EPOCHS and patience >= cfg.ES_PATIENCE:
                break

        if best_state is not None:
            model.load_state_dict({k:v.to(cfg.DEVICE) for k,v in best_state.items()})
        vl, y_t, y_p, y_pred = evaluate(model, va_dl, use_amp, pos_weight=pw)
        y_true = y_t.astype(int)
        acc = accuracy_score(y_true, (y_p>=0.5).astype(int))
        f1  = f1_score(y_true, (y_p>=0.5).astype(int), zero_division=0)
        auc_v = roc_auc_score(y_true, y_p) if len(np.unique(y_true))>1 else 0.5
        cm = confusion_matrix(y_true, (y_p>=0.5).astype(int))

        print(f"[Fold {fold}] AUC={auc_v:.3f}  ACC={acc:.3f}  F1={f1:.3f}")
        print(cm)

        # 저장
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_roc_pr_cal(fold_dir / "roc_pr_cal.jpg", y_true, y_p, "")
        save_confusion(fold_dir / "confusion_high_low.jpg", f"Confusion (Fold {fold})", cm)
        best_thr, best_J = save_youden_bar(fold_dir / "youdenJ_metrics_bar.jpg", y_true, y_p, acc, f1, auc_v, f"Fold {fold}")
        export_predictions(fold_dir / "predictions.csv", f"fold_{fold}", y_true, y_p)

        # Loss Curve (최소 200 epoch 軸 보장)
        plt.figure(figsize=(6,4))
        plt.plot(tr_losses, label="train"); plt.plot(va_losses, label="valid")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title(f"Loss (Fold {fold})")
        plt.tight_layout(); plt.savefig(fold_dir / "loss.jpg", dpi=180); plt.close()

        summary_rows.append({
            "fold": fold, "auc": auc_v, "acc": acc, "f1": f1,
            "n_val": int(len(y_true)),
            "pos_val": int((y_true==1).sum()),
            "neg_val": int((y_true==0).sum()),
            "youdenJ": float(best_J), "best_threshold": float(best_thr)
        })

    pd.DataFrame(summary_rows).to_csv(out_dir/"summary_metrics.csv", index=False)

    # -------- 8:2 Holdout Test --------
    print("\n===== HOLDOUT TEST (8:2) =====")
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, stratify=y, random_state=cfg.SEED)
    tr_ds, te_ds = HRVDataset(X_tr, y_tr), HRVDataset(X_te, y_te)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = CNN_Transformer().to(cfg.DEVICE)
    opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    sched = cosine_with_warmup(opt, cfg.EPOCHS, cfg.WARMUP_EPOCHS)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(cfg.DEVICE.type=="cuda"))
    pos = (y_tr==1).sum(); neg = (y_tr==0).sum()
    pw = torch.tensor([neg/max(1,pos)], device=cfg.DEVICE) if (pos>0 and neg>0) else None
    best_auc, best_state, patience = 0.0, None, 0
    tr_losses, te_losses = [], []
    use_amp = True
    for epoch in range(cfg.EPOCHS):
        tl = train_one_epoch(model, tr_dl, opt, scaler_amp, pw, use_amp)
        vl, y_t, y_p, _ = evaluate(model, te_dl, use_amp, pos_weight=pw)
        tr_losses.append(tl); te_losses.append(vl)
        auc_val = roc_auc_score(y_t.astype(int), y_p) if len(np.unique(y_t))>1 else 0.5
        sched.step()
        if auc_val > best_auc:
            best_auc = auc_val; best_state = {k:v.cpu() for k,v in model.state_dict().items()}; patience=0
        else:
            patience += 1
        if (epoch + 1) >= cfg.MIN_EPOCHS and patience >= cfg.ES_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict({k:v.to(cfg.DEVICE) for k,v in best_state.items()})
    vl, y_t, y_p, y_pred = evaluate(model, te_dl, use_amp, pos_weight=pw)
    y_true = y_t.astype(int); y_hat = (y_p>=0.5).astype(int)
    acc = accuracy_score(y_true, y_hat)
    f1  = f1_score(y_true, y_hat, zero_division=0)
    auc_v = roc_auc_score(y_true, y_p) if len(np.unique(y_true))>1 else 0.5
    cm = confusion_matrix(y_true, y_hat)
    print(f"[Test] AUC={auc_v:.3f}  ACC={acc:.3f}  F1={f1:.3f}")
    print(cm)

    test_dir = out_dir / "holdout_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    save_confusion(test_dir / "confusion_high_low.jpg", "Confusion (Holdout 8:2)", cm)
    save_roc_pr_cal(test_dir / "roc_pr_cal.jpg", y_true, y_p, " — Holdout")
    best_thr, best_J = save_youden_bar(test_dir / "youdenJ_metrics_bar.jpg", y_true, y_p, acc, f1, auc_v, "Holdout")
    export_predictions(test_dir / "predictions.csv", "holdout", y_true, y_p)

    plt.figure(figsize=(6,4))
    plt.plot(tr_losses, label="train"); plt.plot(te_losses, label="valid/test")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss (Holdout 8:2)")
    plt.tight_layout(); plt.savefig(test_dir / "loss.jpg", dpi=180); plt.close()

def main():
    # 원본 로딩
    X, y, groups = load_real_data()
    if len(X)==0:
        print("[!] 유효 시퀀스가 없습니다."); return

    # 합성(밸런싱)
    X_bal, y_bal, g_bal = synthesize_balanced(
        X, y, groups,
        target_per_class=cfg.TARGET_PER_CLASS,
        low_min=cfg.LOW_MIN
    )
    # 합성 여부 마스크 (Real=0, Synth=1)
    synth_mask = np.concatenate([np.zeros(len(X), np.int32),
                                 np.ones(len(X_bal)-len(X), np.int32)])

    # Boxplot (Real vs Synth, High/Low)
    save_boxplot_real_vs_synth(cfg.OUT, X_bal, y_bal, synth_mask)

    # Real/Synth PPG·HRV 3-패널 그래프
    df_res = pd.read_csv(cfg.RESULTS_FILE, engine="python")
    if "filename" not in df_res.columns:
        df_res = df_res.rename(columns={df_res.columns[0]: "filename"})
    df_res["key_full"] = df_res["filename"].map(_norm_name)
    df_res["key_subj"] = df_res["filename"].map(_subject_key)
    acc_norm = _normalize_accuracy_column(df_res["accuracy"])
    df_res["label"] = _label_from_accuracy(acc_norm)
    save_real_synth_timeseries_by_group(cfg.OUT, X_bal, y_bal, synth_mask, df_res)

    # 5-Fold + Holdout Test
    run_folds_and_holdout(X_bal, y_bal, cfg.OUT)
    print(f"\n[*] 결과 저장 완료 → {cfg.OUT}")

if __name__ == "__main__":
    main()
