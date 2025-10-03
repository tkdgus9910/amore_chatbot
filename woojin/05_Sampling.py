# make_sample_pkl.py  — 인자 없이 15% 샘플링 후 새 PKL 저장

import pandas as pd
import pickle
from pathlib import Path

# ✅ 여기만 바꿔 쓰면 됨
INP  = r"C:\Users\PC1\OneDrive\250801_아모레\df_sample_pages_old.pkl"
OUT  = str(Path(INP).with_name("df_sample_pages_15pct.pkl"))
FRAC = 0.15
SEED = 42

def load_pkl(path: str):
    """pd.read_pickle 먼저 시도, 실패 시 pickle.load로 재시도."""
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def main():
    inp = Path(INP)
    if not inp.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {inp}")

    df = load_pkl(str(inp))
    n0 = len(df)
    if n0 == 0:
        raise ValueError("입력 DataFrame이 비어 있습니다.")

    df_s = df.sample(frac=FRAC, random_state=SEED).sort_index()
    if df_s.empty:
        # 극단 케이스 보호
        df_s = df.sample(n=1, random_state=SEED).sort_index()

    out = Path(OUT)
    df_s.to_pickle(str(out))
    print(f"[OK] 샘플링 저장 완료: {out}")
    print(f" - 원본 행수: {n0:,}")
    print(f" - 샘플 행수: {len(df_s):,}  (frac={FRAC}, seed={SEED})")

if __name__ == "__main__":
    main()
