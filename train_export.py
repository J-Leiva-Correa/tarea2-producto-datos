from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

ARTIFACTS_DIR = Path("model")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 1) Dataset
wine = load_wine(as_frame=True)
X_raw = wine.frame.drop(columns=["target"])
y = wine.target

# 2) Renombrar columnas para JSON válido
col_map = {
    "alcohol": "alcohol",
    "malic_acid": "malic_acid",
    "ash": "ash",
    "alcalinity_of_ash": "alcalinity_of_ash",
    "magnesium": "magnesium",
    "total_phenols": "total_phenols",
    "flavanoids": "flavanoids",
    "nonflavanoid_phenols": "nonflavanoid_phenols",
    "proanthocyanins": "proanthocyanins",
    "color_intensity": "color_intensity",
    "hue": "hue",
    "od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines",
    "proline": "proline",
}
X = X_raw.rename(columns=col_map)[list(col_map.values())].copy()
feature_order = list(X.columns)

# 3) Split (métrica informativa)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 4) Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=8, random_state=42)),
    ("clf", GaussianNB()),
])

# 5) Entrena y evalúa
pipe.fit(Xtr, ytr)
acc = accuracy_score(yte, pipe.predict(Xte))
print(f"Accuracy holdout: {acc:.4f}")

# 6) Guardar artefactos
joblib.dump(pipe, ARTIFACTS_DIR / "wine_nb_pipeline.joblib")
(ARTIFACTS_DIR / "feature_order.json").write_text(json.dumps(feature_order, indent=2))
metadata = {
    "model_name": "wine_nb_pipeline",
    "algorithm": "GaussianNB",
    "preprocessing": ["StandardScaler", "PCA(n_components=8)"],
    "feature_order": feature_order,
    "classes": [int(c) for c in np.unique(y)],
    "target_names": list(wine.target_names),
    "holdout_accuracy": float(acc),
}
(ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
print("Artefactos guardados en ./model")
