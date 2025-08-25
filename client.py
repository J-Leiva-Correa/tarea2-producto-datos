# client.py
import requests
import json
from pprint import pprint

# Cambia esta URL a la de Render en el Paso 5
BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{BASE_URL}/predict"
HEALTH_URL = f"{BASE_URL}/health"

def pretty(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def run_tests():
    # 0) Health check (opcional, útil para debug)
    print("== GET /health ==")
    r = requests.get(HEALTH_URL, timeout=10)
    pretty(r.json())

    tests = [
        {
            "name": "Caso A (valores moderados)",
            "payload": {
                "alcohol": 13.2, "malic_acid": 1.8, "ash": 2.4, "alcalinity_of_ash": 15.0,
                "magnesium": 100.0, "total_phenols": 2.4, "flavanoids": 2.0, "nonflavanoid_phenols": 0.3,
                "proanthocyanins": 1.7, "color_intensity": 5.5, "hue": 1.0,
                "od280_od315_of_diluted_wines": 3.0, "proline": 1000.0
            },
        },
        {
            "name": "Caso B (fenoles altos / proline alto)",
            "payload": {
                "alcohol": 14.1, "malic_acid": 1.2, "ash": 2.3, "alcalinity_of_ash": 13.0,
                "magnesium": 110.0, "total_phenols": 3.2, "flavanoids": 2.8, "nonflavanoid_phenols": 0.2,
                "proanthocyanins": 1.9, "color_intensity": 7.0, "hue": 1.05,
                "od280_od315_of_diluted_wines": 3.2, "proline": 1400.0
            },
        },
        {
            "name": "Caso C (malic_acid alto / color_intensity bajo)",
            "payload": {
                "alcohol": 12.2, "malic_acid": 3.0, "ash": 2.1, "alcalinity_of_ash": 20.0,
                "magnesium": 90.0, "total_phenols": 1.9, "flavanoids": 1.2, "nonflavanoid_phenols": 0.5,
                "proanthocyanins": 1.3, "color_intensity": 2.5, "hue": 0.95,
                "od280_od315_of_diluted_wines": 2.1, "proline": 600.0
            },
        },
    ]

    for t in tests:
        print(f"\n== POST /predict :: {t['name']} ==")
        print("Payload:")
        pretty(t["payload"])
        resp = requests.post(PREDICT_URL, json=t["payload"], timeout=10)
        print("Response:")
        pretty(resp.json())

if __name__ == "__main__":
    run_tests()
