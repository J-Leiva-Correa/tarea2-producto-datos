# TAREA 2 – Despliegue de Modelo ML como API (FastAPI + Render)

## 1. Descripción
Este proyecto convierte un modelo de Machine Learning en un **producto de datos** accesible vía **API REST** usando **FastAPI** y desplegado en **Render**.  
El modelo clasifica vinos (dataset *Wine* de scikit-learn) con un pipeline: `StandardScaler` → `PCA(n_components=8)` → `GaussianNB`.

---

## 2. Instalación local

```bash
# 1) Crear/activar entorno virtual (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Instalar dependencias
pip install -r requirements.txt


```

# 3. Entrenar/Exportar modelo (opcional)

El repositorio ya incluye los artefactos listos en `./model`.  
Si deseas regenerarlos:

```bash
python train_export.py

```
# 4. Ejecutar servidor local

```bash
uvicorn main:app --reload


```
# 5. API desplegada (Render)

- Base URL → https://tarea2-producto-datos.onrender.com  
- Health → https://tarea2-producto-datos.onrender.com/health  
- Docs (Swagger) → https://tarea2-producto-datos.onrender.com/docs  

⚠️ **Nota (plan Free de Render):**  
Si no hay tráfico por ~15 minutos, la instancia “duerme” y la primera request puede tardar 30–60 segundos. Luego responde normal.

# 6. Contrato de la API

**POST** `/predict`  

### Entrada (JSON, 13 floats)
- `alcohol` (>0)  
- `malic_acid` (>0)  
- `ash` (>0)  
- `alcalinity_of_ash` (>0)  
- `magnesium` (>0)  
- `total_phenols` (>0)  
- `flavanoids` (≥0)  
- `nonflavanoid_phenols` (≥0)  
- `proanthocyanins` (≥0)  
- `color_intensity` (≥0)  
- `hue` (>0)  
- `od280_od315_of_diluted_wines` (>0)  
- `proline` (>0)  

### Ejemplo de request válido
```json
{
  "alcohol": 13.2,
  "malic_acid": 1.8,
  "ash": 2.4,
  "alcalinity_of_ash": 15.0,
  "magnesium": 100.0,
  "total_phenols": 2.4,
  "flavanoids": 2.0,
  "nonflavanoid_phenols": 0.3,
  "proanthocyanins": 1.7,
  "color_intensity": 5.5,
  "hue": 1.0,
  "od280_od315_of_diluted_wines": 3.0,
  "proline": 1000.0
}

```
# 7. Ejemplos de error (422)

### Campo faltante (`proline`)
```json
{
  "detail": "Entrada inválida",
  "errors": [
    { "loc": ["body","proline"], "msg": "Field required", "type": "missing" }
  ]
}

{
  "detail": "Entrada inválida",
  "errors": [
    { "loc": ["body","alcohol"], "msg": "Input should be greater than 0", "type": "greater_than" }
  ]
}


```
# 8. Cliente externo (`client.py`)

Ejecuta pruebas contra Render:

```bash
python client.py
BASE_URL = "http://127.0.0.1:8000"


```
# 9. Estructura del repositorio

```text
.
├── main.py                 # API FastAPI (/health, /predict)
├── client.py               # Cliente externo con ≥3 pruebas
├── train_export.py         # Entrena y exporta artefactos (opcional)
├── requirements.txt        # Dependencias (API, ML y cliente)
├── model/
│   ├── wine_nb_pipeline.joblib
│   ├── feature_order.json
│   └── metadata.json
└── README.md

```
# 10. Autor
**Joaquín Leiva Correa – Magíster en Data Science (UDD)**
