# 📚 Rangkuman Capstone ML — Web Page Phishing Detection
> Ringkasan tahapan + kode inti untuk persiapan ujian

---

## 🗺️ Alur Besar

```
Import Library → Load Data → EDA → Preprocessing → Split+Scale → Modelling → CV → Tuning → Feature Engineering → Kesimpulan
```

---

## 1. Import Library

> Hafal kelompoknya, bukan satu per satu

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import PolynomialFeatures

RANDOM_STATE = 42
```

---

## 2. EDA

> Tiga fungsi wajib + identifikasi tipe kolom

```python
df.head()       # lihat 5 baris pertama
df.info()       # tipe data & jumlah non-null
df.describe()   # statistik deskriptif

# Distribusi target
df['status'].value_counts()

# Identifikasi kolom numerik vs kategorikal binary
for col in num_cols:
    if set(df[col].unique()).issubset({-1, 0, 1}):
        binary_cols.append(col)
    else:
        true_num_cols.append(col)
```

---

## 3. Preprocessing

> Urutan: Missing → Drop → Outlier → Encoding → (Split) → Scaling

### Missing Value
```python
df.isnull().sum().sum()   # cek total missing

# Jika ada:
df.dropna()               # drop baris (missing tinggi)
df.fillna(df.median())    # isi dengan median (numerik)
df.fillna(df.mode()[0])   # isi dengan modus (kategorikal)
```

### Drop Kolom Tidak Relevan
```python
df = df.drop(columns=['url'])
```

### Outlier Detection + Winsorizing (IQR)
```python
# Deteksi
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Treatment — Winsorizing
df_processed[col] = df_processed[col].clip(lower=lower, upper=upper)
```

### Encoding
```python
# Label Encoding → untuk target / kolom binary
le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])
```

### Split + Scaling
```python
# Split dulu — BARU scaling!
X = df_processed.drop(columns=['status'])
y = df_processed['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scaling — fit pada train, transform ke train & test
scaler = StandardScaler()
X_train_scaled[true_num_cols] = scaler.fit_transform(X_train[true_num_cols])
X_test_scaled[true_num_cols]  = scaler.transform(X_test[true_num_cols])
```

---

## 4. Modelling & Evaluasi

> LR pakai data scaled, RF tidak perlu

```python
# Logistic Regression (Linear) — pakai scaled
lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)

# Random Forest (Non-linear) — tidak perlu scaled
rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Evaluasi
print(classification_report(y_test, y_pred))
print(f"F1 : {f1_score(y_test, y_pred):.4f}")

# Cek overfitting — bandingkan train vs test
gap = accuracy_score(y_train, model.predict(X_train)) - accuracy_score(y_test, y_pred)
# gap > 0.05 → overfitting
# train score < 0.75 → underfitting
```

---

## 5. Cross Validation

> Klasifikasi → StratifiedKFold | Regresi → KFold

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

print(f"Mean F1 : {scores.mean():.4f}")
print(f"Std     : {scores.std():.4f}")

# Kesimpulan:
# Std kecil  → model stabil
# Mean CV ≈ F1 Test Set (selisih < 0.05) → model konsisten
```

---

## 6. Hyperparameter Tuning

> GridSearchCV = coba SEMUA kombinasi | RandomizedSearchCV = coba sebagian acak

```python
param_grid = {
    'n_estimators'     : [50, 100, 200],
    'max_depth'        : [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid = param_grid,
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring    = 'f1',
    n_jobs     = -1
)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

best_model = grid_search.best_estimator_
```

---

## 7. Feature Engineering

### Binning
```python
# Ubah angka kontinu → kategori
df['page_rank_bin'] = pd.cut(
    df['page_rank'],
    bins=[0, 3, 6, 10],
    labels=['low', 'medium', 'high']
)
```

### Polynomial Features
```python
# Buat fitur baru dari kombinasi fitur yang ada
poly = PolynomialFeatures(degree=2, include_bias=False)

poly_train = poly.fit_transform(X_train[top_features])  # fit pada train
poly_test  = poly.transform(X_test[top_features])       # transform test
```

---

## 🧠 Yang Wajib Diingat

| Pertanyaan | Jawaban Singkat |
|---|---|
| Kenapa split sebelum scaling? | Cegah data leakage — model tidak boleh "melihat" test set |
| KFold vs Stratified KFold? | Klasifikasi → Stratified, Regresi → KFold biasa |
| GridSearch vs Randomized? | Grid = semua kombinasi (sedikit param), Random = acak (banyak param) |
| Kenapa RF tidak perlu scaling? | Berbasis pohon, tidak hitung jarak — tidak peduli skala |
| Metrik prioritas phishing? | Recall — lebih bahaya kalau phishing lolos tidak terdeteksi |
| Std kecil artinya apa? | Model stabil dan konsisten di berbagai pembagian data |
| F1 turun setelah tuning — buruk? | Tidak selalu — bisa berarti overfitting berkurang (lebih sehat) |
| Binning untuk apa? | Ubah angka kontinu → kategori (sederhanakan fitur) |
| Polynomial untuk apa? | Buat fitur baru dari kombinasi fitur yang ada |

---

## 📊 Hasil Akhir Dataset Ini

| Tahap | F1 Score |
|---|---|
| Logistic Regression | 0.9274 |
| Random Forest | 0.9617 |
| Setelah Tuning | 0.9600 |
| Setelah Feature Engineering | 0.9617 |

**Model terbaik:** Random Forest (`max_depth=20, min_samples_split=2, n_estimators=100`)
**Recall final:** 0.9668 → 96.68% phishing berhasil terdeteksi
