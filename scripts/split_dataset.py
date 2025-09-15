import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

SRC = Path("data/cve_clean.csv")
assert SRC.exists(), f"Нет файла {SRC} — см. README, как его получить"

df = pd.read_csv(SRC)
manual = df[df.label_source == "manual"]
smet   = df[df.label_source == "smet"]

# manual → 80/20 (test = 20% строго «золотые»)
train_man, test = train_test_split(manual, test_size=0.20, random_state=42, shuffle=True, stratify=None)

# train_full = всё smet + 80% manual
train_full = pd.concat([smet, train_man]).reset_index(drop=True)

# train_full → 90/10
train, val = train_test_split(train_full, test_size=0.10, random_state=42, shuffle=True, stratify=None)

outdir = Path("data")
outdir.mkdir(parents=True, exist_ok=True)
train.to_csv(outdir / "train.csv", index=False)
val.to_csv(outdir / "val.csv", index=False)
test.to_csv(outdir / "test.csv", index=False)

print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")
