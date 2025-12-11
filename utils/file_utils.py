from pathlib import Path
from typing import Dict

from rich.console import Console
from tqdm import tqdm

console = Console(force_terminal=True)

try:
    import config.settings as cfg

    MAX_BYTES_PER_FILE = cfg.MAX_BYTES_PER_FILE
    MAX_TOTAL_BYTES = cfg.MAX_TOTAL_BYTES
    SCRIPT_NAME = cfg.SCRIPT_NAME
except Exception as exc:
    console.print(
        f"[yellow]⚠️ Impossible d'importer config.settings : {exc}. Utilisation des valeurs par défaut.[/yellow]"
    )
    MAX_BYTES_PER_FILE = 500 * 1024  # 500 KB par défaut
    MAX_TOTAL_BYTES = 5 * 1024 * 1024  # 5 Mo max total envoyés au modèle
    SCRIPT_NAME = "orchestrateur.py"  # Nom du script à exclure

EXCLUDED_PATTERNS = {
    ".venv",
    "__pycache__",
    ".git",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "dist",
    "build",
    ".env",
    ".vscode",
    ".idea",
}

SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".go",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".rst",
}

# nouveaux seuils
BIG_FILE_THRESHOLD = 30 * 1024  # 30 KB : au-delà, on tronque
BIG_FILE_PREVIEW = 8 * 1024  # on n'envoie que 8 KB dans le prompt


def read_file(path: Path, max_bytes: int = MAX_BYTES_PER_FILE) -> str:
    """Lit un fichier en UTF-8 et le tronque à `max_bytes`."""
    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read(max_bytes)
        return content
    except Exception as e:
        console.print(f"[red]Erreur de lecture {path} : {e}[/red]")
        return ""


def collect_files(
    root: Path,
    max_total_bytes: int = MAX_TOTAL_BYTES,
    max_bytes_per_file: int = MAX_BYTES_PER_FILE,
) -> Dict[str, str]:
    files_data: Dict[str, str] = {}
    total_bytes = 0
    console.print(f"[cyan]🔍 Scan du répertoire {root}...[/cyan]")

    all_files = list(root.rglob("*"))
    console.print(f"[dim]   Fichiers trouvés : {len([f for f in all_files if f.is_file()])}[/dim]")

    for path in tqdm(all_files, desc="Collecte des fichiers", unit="fichier"):
        if not path.is_file():
            continue
        if path.name == SCRIPT_NAME:
            console.print(f"[dim]   ⏭️  Exclusion : {path.name}[/dim]")
            continue
        if any(pattern in str(path) for pattern in EXCLUDED_PATTERNS):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        rel_path = str(path.relative_to(root))
        size = path.stat().st_size
        # quantité maximale qu'on accepterait normalement
        size_to_read = min(size, max_bytes_per_file)
        # si fichier volumineux, on n'en lit qu'un extrait
        truncated = False
        if size >= BIG_FILE_THRESHOLD:
            truncated = True
            size_to_read = min(BIG_FILE_PREVIEW, size_to_read)
            console.print(
                f"[yellow]   ⚠️ {rel_path} volumineux ({size / 1024:.1f} KB), "
                f"envoi d'un extrait de {size_to_read / 1024:.1f} KB seulement.[/yellow]"
            )

        if total_bytes + size_to_read > max_total_bytes:
            console.print("[yellow]⚠️  Limite totale de bytes atteinte, arrêt de la collecte.[/yellow]")
            break

        content = read_file(path, max_bytes=size_to_read)
        if not content:
            continue

        if truncated:
            content += (
                "\n\n[... File truncated in initial context. "
                "Use the read_file tool with this path to inspect the full content ...]"
            )

        files_data[rel_path] = content
        total_bytes += size_to_read
        console.print(f"[dim]   ✓ {rel_path} ({size_to_read / 1024:.1f} KB)[/dim]")

    console.print(f"[green]✅ Collecté {len(files_data)} fichiers ({total_bytes / 1024:.1f} KB).[/green]")
    return files_data
