from pathlib import Path
import json
import csv
import numpy as np


# ---------------------------------------------------------------------
# Paths / design
# ---------------------------------------------------------------------

ROOT = Path("/factory")

EMBED_PATH = (
    ROOT
    / "campaigns"
    / "study1_35f300_v1"
    / "analysis"
    / "embeddings_300x6x2048.npy"
)

OUT_DIR = (
    ROOT
    / "campaigns"
    / "study1_35f300_v1"
    / "variance_decomposition_300"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BATCH = 30
N_POSITION = 10
N_EMOTION = 6

EMOTIONS = [
    "normal",
    "happy",
    "sad",
    "calm",
    "angry",
    "surprise",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def pct(x, total):
    return 100.0 * float(x) / float(total)


def fmt(x):
    return f"{x:,.6f}"


def effect_ss(effect, multiplicity=1):
    """
    Multivariate sum of squares:
        multiplicity * sum over effect cells and embedding dimensions of effect^2
    """
    return float(multiplicity * np.sum(effect * effect, dtype=np.float64))


def max_abs(x):
    return float(np.max(np.abs(x)))


# ---------------------------------------------------------------------
# Load / validate
# ---------------------------------------------------------------------

print("=" * 100)
print("STUDY1 35F300 - FULL 300-CANDIDATE VARIANCE DECOMPOSITION")
print("=" * 100)

if not EMBED_PATH.exists():
    raise FileNotFoundError(f"Embedding file not found: {EMBED_PATH}")

X = np.load(EMBED_PATH)

print(f"\nEmbedding file : {EMBED_PATH}")
print(f"Loaded shape   : {X.shape}")
print(f"Loaded dtype   : {X.dtype}")

expected_shape = (
    N_BATCH * N_POSITION,
    N_EMOTION,
    2048,
)

if X.shape != expected_shape:
    raise ValueError(
        f"Unexpected shape: got {X.shape}, expected {expected_shape}"
    )

if not np.all(np.isfinite(X)):
    raise ValueError("Embedding array contains NaN or Inf.")

# Use float64 for the decomposition/accounting itself.
X = X.astype(np.float64, copy=False)

N_CANDIDATE, E, D = X.shape

print(f"Candidates     : {N_CANDIDATE}")
print(f"Emotions       : {E}")
print(f"Dimensions     : {D}")


# =====================================================================
# A. PRIMARY CANDIDATE x EMOTION DECOMPOSITION
#
# X_ce = mu + C_c + E_e + R_ce
#
# R_ce contains:
#   candidate x emotion structure
#   + remaining unseparated cell-level variation
# =====================================================================

mu = X.mean(axis=(0, 1))

C_eff = X.mean(axis=1) - mu
E_eff = X.mean(axis=0) - mu

R = (
    X
    - mu[None, None, :]
    - C_eff[:, None, :]
    - E_eff[None, :, :]
)

centered = X - mu[None, None, :]

SS_total = float(np.sum(centered * centered, dtype=np.float64))

# Candidate effect repeated across all emotions.
SS_C = effect_ss(C_eff, multiplicity=E)

# Emotion effect repeated across all candidates.
SS_E = effect_ss(E_eff, multiplicity=N_CANDIDATE)

SS_R = effect_ss(R)

SS_primary_sum = SS_C + SS_E + SS_R

primary_recon = (
    mu[None, None, :]
    + C_eff[:, None, :]
    + E_eff[None, :, :]
    + R
)

primary_recon_maxerr = max_abs(X - primary_recon)
primary_ss_relerr = abs(SS_primary_sum - SS_total) / SS_total


print("\n" + "=" * 100)
print("A. PRIMARY: CANDIDATE x EMOTION")
print("=" * 100)

print(f"Total SS                         : {fmt(SS_total)}")
print(
    f"Candidate SS                     : {fmt(SS_C)}"
    f"  ({pct(SS_C, SS_total):.3f}%)"
)
print(
    f"Emotion SS                       : {fmt(SS_E)}"
    f"  ({pct(SS_E, SS_total):.3f}%)"
)
print(
    f"CxE + unseparated residual SS    : {fmt(SS_R)}"
    f"  ({pct(SS_R, SS_total):.3f}%)"
)

print(f"\nSS sum                           : {fmt(SS_primary_sum)}")
print(f"Relative SS reconstruction error: {primary_ss_relerr:.3e}")
print(f"Max element reconstruction error: {primary_recon_maxerr:.3e}")


# =====================================================================
# B. BATCH x POSITION x EMOTION DECOMPOSITION
#
# Candidate c == (batch b, position p)
#
# X_bpe =
#   mu
# + B_b + P_p + E_e
# + BP_bp + BE_be + PE_pe
# + BPE_bpe
#
# With one observation per B x P x E cell,
# BPE contains the highest-order interaction plus unseparated error.
# =====================================================================

X3 = X.reshape(
    N_BATCH,
    N_POSITION,
    N_EMOTION,
    D,
)

mu3 = X3.mean(axis=(0, 1, 2))

B_eff = X3.mean(axis=(1, 2)) - mu3
P_eff = X3.mean(axis=(0, 2)) - mu3
E3_eff = X3.mean(axis=(0, 1)) - mu3

BP_mean = X3.mean(axis=2)
BP_eff = (
    BP_mean
    - mu3[None, None, :]
    - B_eff[:, None, :]
    - P_eff[None, :, :]
)

BE_mean = X3.mean(axis=1)
BE_eff = (
    BE_mean
    - mu3[None, None, :]
    - B_eff[:, None, :]
    - E3_eff[None, :, :]
)

PE_mean = X3.mean(axis=0)
PE_eff = (
    PE_mean
    - mu3[None, None, :]
    - P_eff[:, None, :]
    - E3_eff[None, :, :]
)

BPE = (
    X3
    - mu3[None, None, None, :]
    - B_eff[:, None, None, :]
    - P_eff[None, :, None, :]
    - E3_eff[None, None, :, :]
    - BP_eff[:, :, None, :]
    - BE_eff[:, None, :, :]
    - PE_eff[None, :, :, :]
)

SS_B = effect_ss(
    B_eff,
    multiplicity=N_POSITION * N_EMOTION,
)

SS_P = effect_ss(
    P_eff,
    multiplicity=N_BATCH * N_EMOTION,
)

SS_E3 = effect_ss(
    E3_eff,
    multiplicity=N_BATCH * N_POSITION,
)

SS_BP = effect_ss(
    BP_eff,
    multiplicity=N_EMOTION,
)

SS_BE = effect_ss(
    BE_eff,
    multiplicity=N_POSITION,
)

SS_PE = effect_ss(
    PE_eff,
    multiplicity=N_BATCH,
)

SS_BPE = effect_ss(BPE)

SS_secondary_sum = (
    SS_B
    + SS_P
    + SS_E3
    + SS_BP
    + SS_BE
    + SS_PE
    + SS_BPE
)

secondary_recon = (
    mu3[None, None, None, :]
    + B_eff[:, None, None, :]
    + P_eff[None, :, None, :]
    + E3_eff[None, None, :, :]
    + BP_eff[:, :, None, :]
    + BE_eff[:, None, :, :]
    + PE_eff[None, :, :, :]
    + BPE
)

secondary_recon_maxerr = max_abs(X3 - secondary_recon)
secondary_ss_relerr = abs(SS_secondary_sum - SS_total) / SS_total


print("\n" + "=" * 100)
print("B. SECONDARY: BATCH x POSITION x EMOTION")
print("=" * 100)

secondary_rows = [
    ("Batch seed (B)", SS_B),
    ("Position (P)", SS_P),
    ("Emotion (E)", SS_E3),
    ("Batch x Position (B x P)", SS_BP),
    ("Batch x Emotion (B x E)", SS_BE),
    ("Position x Emotion (P x E)", SS_PE),
    ("B x P x E + unseparated residual", SS_BPE),
]

for name, ss in secondary_rows:
    print(
        f"{name:<42}: {fmt(ss):>18}"
        f"  ({pct(ss, SS_total):7.3f}%)"
    )

print(f"\nSS sum                           : {fmt(SS_secondary_sum)}")
print(f"Relative SS reconstruction error: {secondary_ss_relerr:.3e}")
print(f"Max element reconstruction error: {secondary_recon_maxerr:.3e}")


# =====================================================================
# C. COLLAPSED ACCOUNTING CHECK
#
# Candidate = B + P + BP
# Residual  = BE + PE + BPE
# =====================================================================

SS_candidate_collapsed = SS_B + SS_P + SS_BP
SS_residual_collapsed = SS_BE + SS_PE + SS_BPE

candidate_match_abs = abs(SS_candidate_collapsed - SS_C)
residual_match_abs = abs(SS_residual_collapsed - SS_R)

candidate_match_rel = candidate_match_abs / max(abs(SS_C), 1e-30)
residual_match_rel = residual_match_abs / max(abs(SS_R), 1e-30)


print("\n" + "=" * 100)
print("C. COLLAPSED EFFECT ACCOUNTING")
print("=" * 100)

print(
    f"Candidate = B + P + BP           : "
    f"{fmt(SS_candidate_collapsed)}"
    f"  ({pct(SS_candidate_collapsed, SS_total):.3f}%)"
)
print(
    f"Primary Candidate SS             : "
    f"{fmt(SS_C)}"
    f"  ({pct(SS_C, SS_total):.3f}%)"
)
print(f"Candidate relative mismatch      : {candidate_match_rel:.3e}")

print()

print(
    f"Residual = BE + PE + BPE         : "
    f"{fmt(SS_residual_collapsed)}"
    f"  ({pct(SS_residual_collapsed, SS_total):.3f}%)"
)
print(
    f"Primary residual SS              : "
    f"{fmt(SS_R)}"
    f"  ({pct(SS_R, SS_total):.3f}%)"
)
print(f"Residual relative mismatch       : {residual_match_rel:.3e}")


# =====================================================================
# D. WITHIN-CANDIDATE MECHANISM SUMMARY
# =====================================================================

candidate_internal_total = SS_B + SS_P + SS_BP

if candidate_internal_total > 0:
    B_within_candidate_pct = 100.0 * SS_B / candidate_internal_total
    P_within_candidate_pct = 100.0 * SS_P / candidate_internal_total
    BP_within_candidate_pct = 100.0 * SS_BP / candidate_internal_total
else:
    B_within_candidate_pct = np.nan
    P_within_candidate_pct = np.nan
    BP_within_candidate_pct = np.nan


print("\n" + "=" * 100)
print("D. COMPOSITION OF THE CANDIDATE MAIN EFFECT")
print("=" * 100)

print(
    f"Batch seed share within Candidate       : "
    f"{B_within_candidate_pct:.3f}%"
)
print(
    f"Position share within Candidate         : "
    f"{P_within_candidate_pct:.3f}%"
)
print(
    f"Batch x Position share within Candidate : "
    f"{BP_within_candidate_pct:.3f}%"
)


# =====================================================================
# E. SAVE REPORTS
# =====================================================================

report = {
    "input": {
        "embedding_path": str(EMBED_PATH),
        "shape": list(X.shape),
        "dtype_for_analysis": str(X.dtype),
        "n_batch": N_BATCH,
        "n_position": N_POSITION,
        "n_candidate": N_CANDIDATE,
        "n_emotion": N_EMOTION,
        "embedding_dim": D,
        "emotions": EMOTIONS,
    },
    "primary_candidate_emotion": {
        "total_ss": SS_total,
        "candidate_ss": SS_C,
        "emotion_ss": SS_E,
        "candidate_by_emotion_plus_unseparated_ss": SS_R,
        "candidate_pct": pct(SS_C, SS_total),
        "emotion_pct": pct(SS_E, SS_total),
        "candidate_by_emotion_plus_unseparated_pct": pct(SS_R, SS_total),
        "reconstruction_max_abs_error": primary_recon_maxerr,
        "ss_relative_error": primary_ss_relerr,
    },
    "secondary_batch_position_emotion": {
        "batch_seed": {
            "ss": SS_B,
            "pct": pct(SS_B, SS_total),
        },
        "position": {
            "ss": SS_P,
            "pct": pct(SS_P, SS_total),
        },
        "emotion": {
            "ss": SS_E3,
            "pct": pct(SS_E3, SS_total),
        },
        "batch_x_position": {
            "ss": SS_BP,
            "pct": pct(SS_BP, SS_total),
        },
        "batch_x_emotion": {
            "ss": SS_BE,
            "pct": pct(SS_BE, SS_total),
        },
        "position_x_emotion": {
            "ss": SS_PE,
            "pct": pct(SS_PE, SS_total),
        },
        "batch_x_position_x_emotion_plus_unseparated": {
            "ss": SS_BPE,
            "pct": pct(SS_BPE, SS_total),
        },
        "reconstruction_max_abs_error": secondary_recon_maxerr,
        "ss_relative_error": secondary_ss_relerr,
    },
    "collapsed_checks": {
        "candidate_from_B_P_BP_ss": SS_candidate_collapsed,
        "candidate_primary_ss": SS_C,
        "candidate_relative_mismatch": candidate_match_rel,
        "residual_from_BE_PE_BPE_ss": SS_residual_collapsed,
        "residual_primary_ss": SS_R,
        "residual_relative_mismatch": residual_match_rel,
    },
    "candidate_effect_composition": {
        "batch_seed_pct_of_candidate_ss": B_within_candidate_pct,
        "position_pct_of_candidate_ss": P_within_candidate_pct,
        "batch_x_position_pct_of_candidate_ss": BP_within_candidate_pct,
    },
    "interpretation_note": (
        "Because there is one observation per candidate x emotion cell, "
        "the candidate-by-emotion term includes systematic interaction "
        "and remaining unseparated cell-level variation. In the secondary "
        "B x P x E decomposition, the highest-order B x P x E term likewise "
        "contains unseparated cell-level variation."
    ),
}

json_path = OUT_DIR / "variance_decomposition_300_report.json"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)


csv_path = OUT_DIR / "variance_decomposition_300_components.csv"

csv_rows = [
    ["primary", "Candidate", SS_C, pct(SS_C, SS_total)],
    ["primary", "Emotion", SS_E, pct(SS_E, SS_total)],
    [
        "primary",
        "Candidate x Emotion + unseparated",
        SS_R,
        pct(SS_R, SS_total),
    ],
    ["secondary", "Batch seed", SS_B, pct(SS_B, SS_total)],
    ["secondary", "Position", SS_P, pct(SS_P, SS_total)],
    ["secondary", "Emotion", SS_E3, pct(SS_E3, SS_total)],
    ["secondary", "Batch x Position", SS_BP, pct(SS_BP, SS_total)],
    ["secondary", "Batch x Emotion", SS_BE, pct(SS_BE, SS_total)],
    ["secondary", "Position x Emotion", SS_PE, pct(SS_PE, SS_total)],
    [
        "secondary",
        "Batch x Position x Emotion + unseparated",
        SS_BPE,
        pct(SS_BPE, SS_total),
    ],
]

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["analysis", "component", "ss", "percent_total_ss"])
    writer.writerows(csv_rows)


print("\n" + "=" * 100)
print("E. OUTPUT")
print("=" * 100)
print(f"JSON : {json_path}")
print(f"CSV  : {csv_path}")

print("\nDONE")
