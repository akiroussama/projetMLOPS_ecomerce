import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH pour que "import src...." fonctionne
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
