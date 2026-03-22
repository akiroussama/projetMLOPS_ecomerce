"""
Script de generation de trafic pour remplir Grafana avant la soutenance.
Lancer 15 minutes avant de passer.

Usage:
    python scripts/generate_grafana_traffic.py

Duree: ~2 minutes. Genere ~270 requetes sur l'API.
"""

import requests
import time

API_URL = "http://rakuten-mlops.duckdns.org:8200"
TOKEN = "rakuten-soutenance-2024"

HEADERS_AUTH = {"Authorization": f"Bearer {TOKEN}"}

PRODUCTS = [
    {"designation": "FIFA 24 Edition Standard PS5", "description": "Le jeu de football ultime sur PlayStation 5"},
    {"designation": "Harry Potter et la Chambre des Secrets", "description": "Roman de J.K. Rowling edition de poche tome 2"},
    {"designation": "Casque Gaming RGB Razer BlackShark", "description": "Casque audio surround 7.1 pour PC et console"},
    {"designation": "Lego Technic 4x4 Mercedes Benz Zetros", "description": "Set de construction vehicule tout terrain 2110 pieces"},
    {"designation": "Perceuse Visseuse Bosch Professional 18V", "description": "Perceuse sans fil avec 2 batteries lithium ion"},
    {"designation": "The Legend of Zelda Tears of the Kingdom", "description": "Jeu d aventure Nintendo Switch open world"},
    {"designation": "Carnet de Voyage Moleskine Noir Grand Format", "description": "Carnet ligne couverture rigide 240 pages"},
    {"designation": "Lampe de Bureau LED Rechargeable USB", "description": "Lampe architecte tactile 3 niveaux luminosite"},
    {"designation": "Pack 6 Figurines One Piece Luffy Zoro Nami", "description": "Collection manga figurines vinyle 15cm"},
    {"designation": "Veste Softshell Homme Columbia Randonnee", "description": "Veste impermeable legere trekking montagne"},
]


def health_check():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def make_prediction(product):
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json=product,
            headers=HEADERS_AUTH,
            timeout=10
        )
        return r.status_code
    except Exception:
        return None


def make_prediction_no_auth():
    """Genere des 401 -- prouve que la securite fonctionne."""
    try:
        r = requests.post(f"{API_URL}/predict", json=PRODUCTS[0], timeout=5)
        return r.status_code
    except Exception:
        return None


def main():
    print("=" * 50)
    print("  GENERATION TRAFIC GRAFANA - Soutenance MLOps")
    print("=" * 50)

    # Verification API
    print("\n[1/4] Verification API...", end=" ", flush=True)
    if not health_check():
        print("ERREUR - API inaccessible. Verifier le VPS.")
        return
    print("OK")

    # Phase 1 : Health checks (100 requetes)
    print("[2/4] Health checks (100 req)...", end=" ", flush=True)
    ok = 0
    for _ in range(100):
        if health_check():
            ok += 1
        time.sleep(0.05)
    print(f"{ok}/100 OK")

    # Phase 2 : Predictions avec auth (150 requetes)
    print("[3/4] Predictions authentifiees (150 req)...", end=" ", flush=True)
    ok = 0
    for i in range(150):
        product = PRODUCTS[i % len(PRODUCTS)]
        status = make_prediction(product)
        if status == 200:
            ok += 1
        time.sleep(0.1)
    print(f"{ok}/150 OK")

    # Phase 3 : Requetes sans auth (20 req -> genere des 4xx dans Grafana)
    print("[4/4] Requetes sans token (20 req -> 401 intentionnels)...", end=" ", flush=True)
    errors = 0
    for _ in range(20):
        status = make_prediction_no_auth()
        if status in (401, 403):
            errors += 1
        time.sleep(0.05)
    print(f"{errors}/20 x 401")

    # Bilan
    print("\n" + "=" * 50)
    print("  GRAFANA EST PRET !")
    print("  -> Requests/sec : courbes visibles")
    print("  -> Status 2xx   : ~250 requetes reussies")
    print("  -> Status 4xx   : ~20 (montre la secu)")
    print("  -> P95 latency  : mesures stables")
    print("=" * 50)
    print("\nOuvre Grafana maintenant -> les 4 panneaux ont des donnees.")


def slow_drip():
    """Mode demo : 1 prediction toutes les 5s + health check toutes les 2s.
    Laisser tourner en arriere-plan pendant toute la soutenance.
    Arreter avec Ctrl+C.
    """
    print("=" * 50)
    print("  MODE DEMO - SLOW DRIP (Ctrl+C pour arreter)")
    print("  1 prediction/5s + health toutes les 2s")
    print("=" * 50)

    i = 0
    while True:
        # Health check
        ok = health_check()
        status_str = "OK" if ok else "FAIL"
        print(f"  [health] {status_str}", flush=True)
        time.sleep(2)

        # Prediction
        product = PRODUCTS[i % len(PRODUCTS)]
        status = make_prediction(product)
        print(f"  [predict] {product['designation'][:30]}... -> HTTP {status}", flush=True)
        i += 1
        time.sleep(3)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        slow_drip()
    else:
        main()
