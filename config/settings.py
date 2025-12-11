"""Configuration settings for the orchestrator."""

from pathlib import Path
import sys

MODEL_NAME = "gpt-oss:20b"

MODEL_COMPAT = {
    "gpt-oss:20b": {"native_tools": True, "json_fallback": False},
    "qwen2.5-coder:14b": {"native_tools": False, "json_fallback": True},
}

DEFAULT_MODEL_COMPAT = {"native_tools": True, "json_fallback": False}

MAX_BYTES_PER_FILE = 500 * 1024  # 500 KB par défaut
MAX_TOTAL_BYTES = 5 * 1024 * 1024  # 5 Mo max total envoyés au modèle
SCRIPT_NAME = Path(sys.argv[0] or "orchestrateur.py").name  # Nom du script à exclure
VERBOSE = False  # Mode verbeux (défini par argument CLI)
REASONING_LEVEL = "medium"  # Niveau de réflexion : low, medium, high
EXEC_TIMEOUT = 30  # Timeout pour l'exécution de code (en secondes)
MAX_RETRIES = 3  # Nombre maximum de tentatives d'exécution
MAX_AUTONOMY_ITERATIONS = 20  # Nombre max d'itérations autonomes
AUTONOMY_TIMEOUT = 5  # Timeout entre itérations autonomes (secondes)
AUTONOMY = True  # Mode autonomie (enchaînement automatique des tool calls)


def get_model_config(model_name: str):
    """Retourne la configuration d'un modèle donné."""
    return MODEL_COMPAT.get(model_name, DEFAULT_MODEL_COMPAT)
