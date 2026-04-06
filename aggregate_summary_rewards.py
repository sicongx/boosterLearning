import os
import glob
import numpy as np
import pandas as pd


# =========================
# 你需要改的部分
# =========================
RESULT_DIR = "results"   # 如果 seed_2024.npz 等文件就在当前目录，保持 "."
OUT_PER_SEED = os.path.join(RESULT_DIR, "summary_rewards_by_seed.csv")
OUT_FINAL = os.path.join(RESULT_DIR, "summary_rewards.csv")


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    mean = np.mean(x)
    sd = np.std(x, ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else np.nan
    ci95_low = mean - 1.96 * se if n > 0 else np.nan
    ci95_high = mean + 1.96 * se if n > 0 else np.nan

    return {
        "n_seeds": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    }


# 找到所有 seed 文件
files = sorted(glob.glob(os.path.join(RESULT_DIR, "seed_*.npz")))

if len(files) == 0:
    raise FileNotFoundError(f"No files like seed_*.npz found in {RESULT_DIR}")

rows = []

for fp in files:
    z = np.load(fp, allow_pickle=True)

    # 从文件内容里读 seed，最稳
    seed = int(z["seed"]) if "seed" in z.files else int(os.path.basename(fp).split("_")[1].split(".")[0])

    row = {
        "seed": seed,
        "train_repeats": int(z["train_repeats"]) if "train_repeats" in z.files else np.nan,
        "eval_epochs": int(z["eval_epochs"]) if "eval_epochs" in z.files else np.nan,
        "table": float(z["table"]) if "table" in z.files else np.nan,
        "data": float(z["data"]) if "data" in z.files else np.nan,
        "all": float(z["all_policy"]) if "all_policy" in z.files else np.nan,
        "none": float(z["none"]) if "none" in z.files else np.nan,
    }

    if "epoch_reward_shape" in z.files:
        shape_arr = z["epoch_reward_shape"]
        row["epoch_reward_shape"] = tuple(shape_arr.tolist())
    else:
        row["epoch_reward_shape"] = None

    rows.append(row)

seed_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)

# 保存逐 seed 结果
seed_df.to_csv(OUT_PER_SEED, index=False)

# 汇总四个 policy
summary_rows = []
for policy in ["table", "data", "all", "none"]:
    stats = mean_ci(seed_df[policy].values)
    summary_rows.append({
        "policy": policy,
        **stats
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_FINAL, index=False)

# 屏幕输出
print("\n===== Per-seed summary rewards =====")
print(seed_df[["seed", "table", "data", "all", "none"]].to_string(index=False))

print("\n===== Final summary across seeds =====")
print(summary_df.to_string(index=False))

# 顺便打印一个更方便和论文 figure 对照的版本
print("\n===== Mean ± SD =====")
for _, r in summary_df.iterrows():
    print(f"{r['policy']:>5} : {r['mean']:.6f} ± {r['sd']:.6f}")

print(f"\nSaved per-seed results to: {OUT_PER_SEED}")
print(f"Saved final summary to: {OUT_FINAL}")