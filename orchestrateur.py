#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code-Assistant avec Ollama (gpt-oss:20b)
========================================
Ce script lit le contenu d'un répertoire, l'envoie au modèle
et vous propose une interface interactive pour poser des questions
ou demander de l'aide sur le code présent dans ce répertoire.
Fonctionnalités
---------------
* Recherche récursive de tous les fichiers du répertoire
* Limitation automatique de la taille (par défaut 500 KB par fichier)
* Envoi d'un prompt structuré contenant les fichiers
* Interaction via chat (prompts utilisateurs → réponses modèles)
* Support du *tool-calling* : le modèle peut demander à lire un fichier précis
  (fonction `read_file` exposée comme outil).
* Exécution sécurisée de code Python avec feedback au modèle
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import ollama
from tqdm import tqdm
from rich.console import Console
from rich.markdown import Markdown
console = Console(force_terminal=True)
# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "qwen2.5-coder:14b"
# Configuration de compatibilité par modèle
MODEL_COMPAT = {
    "gpt-oss:20b": {"native_tools": True, "json_fallback": False},
    "qwen2.5-coder:14b": {"native_tools": False, "json_fallback": True},
}
DEFAULT_MODEL_COMPAT = {"native_tools": True, "json_fallback": False}

MODEL_NATIVE_TOOLS = MODEL_COMPAT.get(MODEL_NAME, DEFAULT_MODEL_COMPAT)["native_tools"]
MODEL_JSON_FALLBACK = MODEL_COMPAT.get(MODEL_NAME, DEFAULT_MODEL_COMPAT)["json_fallback"]
MAX_BYTES_PER_FILE = 500 * 1024  # 500 KB par défaut
MAX_TOTAL_BYTES = 5 * 1024 * 1024  # 5 Mo max total envoyés au modèle
SCRIPT_NAME = "analyse_fichiers_llm.py"  # Nom du script à exclure
VERBOSE = False  # Mode verbeux (défini par argument CLI)
REASONING_LEVEL = "medium"  # Niveau de réflexion : low, medium, high
EXEC_TIMEOUT = 30  # Timeout pour l'exécution de code (en secondes)
MAX_RETRIES = 3  # Nombre maximum de tentatives d'exécution
MAX_AUTONOMY_ITERATIONS = 20  # Nombre max d'itérations autonomes
AUTONOMY_TIMEOUT = 5  # Timeout entre itérations autonomes (secondes)
AUTONOMY = False  # Mode autonomie (enchaînement automatique des tool calls)
# -----------------------------
# Helpers
# -----------------------------
def log_verbose(message: str):
    """Affiche un message uniquement si VERBOSE est activé."""
    if VERBOSE:
        console.print(f"[dim magenta]🔍 DEBUG: {message}[/dim magenta]")
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
    excluded_patterns = {
        ".venv", "__pycache__", ".git", "node_modules", 
        ".pytest_cache", ".mypy_cache", ".tox", "dist", 
        "build", ".env", ".vscode", ".idea"
    }
    # nouveaux seuils
    BIG_FILE_THRESHOLD = 30 * 1024       # 30 KB : au-delà, on tronque
    BIG_FILE_PREVIEW = 8 * 1024         # on n'envoie que 8 KB dans le prompt
    for path in tqdm(all_files, desc="Collecte des fichiers", unit="fichier"):
        if not path.is_file():
            continue
        if path.name == SCRIPT_NAME:
            console.print(f"[dim]   ⏭️  Exclusion : {path.name}[/dim]")
            continue
        if any(pattern in str(path) for pattern in excluded_patterns):
            continue
        if path.suffix.lower() not in {
            ".py", ".js", ".ts", ".go", ".java", ".cpp", ".c", ".h", ".hpp",
            ".md", ".json", ".yaml", ".yml", ".txt", ".rst"
        }:
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
def build_prompt(files_data: Dict[str, str], native_tools: bool = True) -> str:
    """
    Construit un prompt structuré à envoyer au modèle.
    On sépare chaque fichier par un délimiteur clair.
    """
    parts = [
        "You are a methodical code assistant that MUST follow structured debugging and development processes.",
        "",
        "=== AVAILABLE TOOLS (NEVER invent others) ===",
        "1. list_files(directory_path='.')  → List files in directory",
        "2. read_file(file_path='myfile.py')  → Read complete file content",
        "3. write_file(file_path, content, line_start, line_end)  → Modify files with line precision",
        "4. execute_code(file_path='myfile.py')  → Run Python files",
        "5. create_venv(venv_path='.venv')  → Create virtual environment",
        "",
        "=== MANDATORY PARAMETER RULES ===",
        "❌ read_file: ONLY file_path (NEVER line_start/line_end)",
        "✅ write_file: ALL 4 parameters required (file_path, content, line_start, line_end)",
        "✅ list_files: directory_path only (optional, default='.')",
        "✅ execute_code: file_path only",
        "✅ create_venv: venv_path only (optional, default='.venv')",
        "",
        "=== WRITE_FILE LINE EXAMPLES ===",
        "• write_file('f.py', 'code', 1, -1)     ← Replace ENTIRE file",
        "• write_file('f.py', 'code', 50, 49)    ← INSERT at line 50 (no deletion)",
        "• write_file('f.py', 'code', 10, 20)    ← REPLACE lines 10-20",
        "",
        "=== MANDATORY WORKFLOW - NO SHORTCUTS ===",
        "For ANY task, you MUST follow this exact sequence:",
        "1. 🔍 EXPLORE: Call list_files() to understand project structure",
        "2. 📖 READ: Call read_file() on relevant files to understand current code",
        "3. 📊 ANALYZE: Think through what changes are needed",
        "4. ✏️ IMPLEMENT: Use write_file() with precise line numbers",
        "5. ✅ VERIFY: Use execute_code() or read_file() to confirm changes",
        "",
        "=== CRITICAL BEHAVIORAL RULES ===",
        "• NEVER say 'already did' - ALWAYS execute the requested tool calls",
        "• NEVER take shortcuts or assume previous work",
        "• ALWAYS read files before modifying them to count lines",
        "• NEVER invent tools or use tools that don't exist",
        "• Each tool call must have a clear purpose and be executed",
        "• Follow the 5-step workflow for every code task",
    ]

    if not native_tools:
        parts.extend([
            "",
            "=== TOOL CALL FORMAT (JSON OUTPUT ONLY) ===",
            "When you need a tool, respond *only* with JSON objects using the schema: {\"name\": \"tool_name\", \"arguments\": { ... }}",
            "For multiple tool calls, return an array of such objects.",
            "Do NOT wrap the JSON in prose. No markdown, no extra text.",
        ])

    parts.append("")
    parts.append("Current files in repository:")
    
    for filename, content in files_data.items():
        parts.append(f"--- {filename} ---")
        parts.append(content)
    
    final_prompt = "\n\n".join(parts)
    console.print(f"[dim]📏 Prompt total : {len(final_prompt)} caractères[/dim]")
    return final_prompt
def detect_dangerous_patterns(code: str) -> List[str]:
    """
    Détecte des patterns potentiellement dangereux dans le code.
    Retourne une liste de warnings.
    """
    warnings = []
    
    # Patterns à surveiller
    patterns = {
        r'os\.chdir\s*\(': "Changement de répertoire (os.chdir)",
        r'shutil\.rmtree': "Suppression récursive de répertoire (shutil.rmtree)",
        r'subprocess\.(call|run|Popen)': "Exécution de commandes système (subprocess)",
        r'\.\.\/': "Traversée de répertoire (../)",
        r'os\.remove|os\.unlink': "Suppression de fichier (os.remove/unlink)",
        r'open\s*\([^)]*["\']w["\']': "Écriture dans un fichier (open mode 'w')",
        r'eval\s*\(|exec\s*\(': "Exécution de code dynamique (eval/exec)",
        r'__import__': "Import dynamique (__import__)",
        r'socket\.|urllib\.|requests\.': "Accès réseau",
    }
    
    for pattern, description in patterns.items():
        if re.search(pattern, code):
            warnings.append(f"⚠️  {description}")

    return warnings


def parse_json_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extraie des tool calls encodés en JSON depuis une réponse texte.

    Supporte plusieurs formats :
    - JSON direct : {"name": "...", "arguments": {...}}
    - JSON dans des blocs de code markdown
    - Tableaux de tool calls multiples
    Retourne une liste de tool calls au format attendu par Ollama.
    """
    if not text:
        return []

    candidates: List[str] = []
    stripped = text.strip()

    # JSON direct en début de message
    if stripped.startswith("{") or stripped.startswith("["):
        candidates.append(stripped)

    # Blocs de code markdown (```json ...``` ou ``` ... ```)
    code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    for block in code_block_pattern.findall(text):
        block = block.strip()
        if block:
            candidates.append(block)

    # JSON inline minimal contenant un "name" pour couvrir les cas restants
    inline_pattern = re.compile(r"(\{\s*\"name\"\s*:\s*\"[^\"]+\"[\s\S]*?\})")
    for match in inline_pattern.findall(text):
        match = match.strip()
        if match:
            candidates.append(match)

    tool_calls: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def add_tool_call(name: str, arguments: Any) -> None:
        call_id = f"json-tool-{len(tool_calls) + 1}"
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments if arguments is not None else {},
                },
            }
        )

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "name" in obj and "arguments" in obj:
                add_tool_call(obj.get("name", ""), obj.get("arguments", {}))
            else:
                for value in obj.values():
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except Exception:
            log_verbose(f"Échec de parsing JSON (fallback tool calls) pour: {candidate[:100]}")
            continue
        walk(parsed)

    return tool_calls
# -----------------------------
# Tool (function calling) definition
# -----------------------------
def read_file_tool(file_path: str) -> str:
    """
    Fonction appelée par le modèle pour lire un fichier précis
    (utilisée via le tool-calling d'Ollama).
    """
    root = Path.cwd()
    full_path = root / file_path
    if not full_path.exists() or not full_path.is_file():
        return f"[ERROR] File '{file_path}' not found."
    return read_file(full_path)
def list_files_tool(directory_path: str = ".") -> str:
    """
    Outil pour lister les fichiers et dossiers dans un répertoire.
    Retourne une liste formatée des fichiers et dossiers.
    """
    try:
        # Chemin relatif au répertoire racine du projet
        root = Path.cwd()
        target_dir = root / directory_path
        log_verbose(f"list_files_tool: {target_dir}")
        
        if not target_dir.exists():
            return f"[ERROR] Directory does not exist: {directory_path}"
        
        if not target_dir.is_dir():
            return f"[ERROR] Not a directory: {directory_path}"
        
        # Exclure certains dossiers/fichiers
        excluded_patterns = {
            ".venv", "__pycache__", ".git", "node_modules", 
            ".pytest_cache", ".mypy_cache", ".tox", "dist", 
            "build", ".env", ".vscode", ".idea", ".DS_Store"
        }
        
        items = []
        
        # Lister le contenu
        for item in sorted(target_dir.iterdir()):
            # Ignorer les fichiers/dossiers exclus
            if any(pattern in str(item.name) for pattern in excluded_patterns):
                continue
                
            relative_path = item.relative_to(root)
            
            if item.is_dir():
                # Compter les fichiers dans le dossier
                try:
                    file_count = len([f for f in item.iterdir() if f.is_file()])
                    items.append(f"📁 {relative_path}/ ({file_count} fichiers)")
                except PermissionError:
                    items.append(f"📁 {relative_path}/ (accès refusé)")
            else:
                # Afficher la taille du fichier
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/(1024*1024):.1f}MB"
                
                # Icône selon l'extension
                ext = item.suffix.lower()
                if ext == ".py":
                    icon = "🐍"
                elif ext in [".js", ".ts", ".jsx", ".tsx"]:
                    icon = "📜"
                elif ext in [".md", ".txt", ".rst"]:
                    icon = "📄"
                elif ext in [".json", ".yaml", ".yml"]:
                    icon = "⚙️"
                elif ext in [".html", ".css"]:
                    icon = "🌐"
                else:
                    icon = "📄"
                
                items.append(f"{icon} {relative_path} ({size_str})")
        
        if not items:
            return f"[INFO] Directory '{directory_path}' is empty (or contains only excluded files/folders)"
        
        # Formatter la réponse
        header = f"📂 Contents of '{directory_path}':"
        return header + "\n" + "\n".join(f"  {item}" for item in items)
        
    except Exception as e:
        return f"[ERROR] Failed to list directory '{directory_path}': {e}"
def write_file_tool(
    file_path: str,
    content: str,
    line_start: int,  # Maintenant obligatoire
    line_end: int,    # Maintenant obligatoire
) -> str:
    """
    Fonction appelée par le modèle pour écrire/modifier un fichier.
    OBLIGATOIRE : Spécifier line_start et line_end pour toute modification.
    - Pour INSERTION à la ligne N : line_start=N, line_end=N-1
    - Pour REMPLACEMENT lignes M-N : line_start=M, line_end=N  
    - Pour REMPLACEMENT COMPLET : line_start=1, line_end=-1
    Demande confirmation à l'utilisateur avant d'écrire.
    """
    root = Path.cwd()
    full_path = root / file_path

    # Sécurité chemin
    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(root)):
            return f"[ERROR] Path '{file_path}' is outside the project directory."
    except Exception as e:
        return f"[ERROR] Invalid path: {e}"

    # Refus de contenu vide (éviter les effacements accidentels)
    if content is None or content.strip() == "":
        return (
            "[ERROR] Empty 'content' passed to write_file_tool; refusing to overwrite the file.\n"
            "You must provide non-empty content."
        )

    # Charger l'existant s'il existe
    existing_text = ""
    if full_path.exists():
        try:
            existing_text = full_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"[ERROR] Failed to read existing file before patching: {e}"

    # TOUJOURS en mode PATCH PAR LIGNES (plus de mode remplacement automatique)
    lines = existing_text.splitlines()

    # Gestion du line_end spécial -1 (fin de fichier)
    if line_end == -1:
        line_end = len(lines)

    # Normalisation des indices (1-based inclusif -> 0-based / slice)
    if line_start < 1:
        line_start = 1

    start_idx = max(line_start - 1, 0)
    end_idx = min(line_end, len(lines))

    new_block_lines = content.splitlines()

    # Cas fichier inexistant 
    if not lines:
        new_lines = new_block_lines
        mode_desc = "création fichier"
    else:
        # Insertion ou remplacement
        if line_start > line_end:
            # Mode INSERTION (line_start=N, line_end=N-1)
            new_lines = lines[:start_idx] + new_block_lines + lines[start_idx:]
            mode_desc = f"insertion ligne {line_start}"
        else:
            # Mode REMPLACEMENT (line_start=M, line_end=N)
            new_lines = lines[:start_idx] + new_block_lines + lines[end_idx:]
            if line_start == 1 and line_end >= len(lines):
                mode_desc = "remplacement complet"
            else:
                mode_desc = f"remplacement lignes {line_start}-{line_end}"

    final_content = "\n".join(new_lines)

    # Aperçu
    console.print(
        f"\n[yellow]⚠️  Le modèle veut modifier/créer le fichier : [bold]{file_path}[/bold][/yellow]"
    )
    console.print(f"[dim]   Mode : {mode_desc}[/dim]")

    preview = final_content[:500]
    if len(final_content) > 500:
        preview += f"\n... ({len(final_content) - 500} caractères supplémentaires)"

    console.print("[dim]Aperçu du contenu final :[/dim]")
    console.print("[dim]" + "─" * 60 + "[/dim]")
    console.print(preview)
    console.print("[dim]" + "─" * 60 + "[/dim]")

    # Confirmation utilisateur
    confirmation = input("Autoriser cette modification ? (o/n) : ")

    if confirmation.lower() not in ["o", "oui", "y", "yes"]:
        return "[CANCELLED] User cancelled the file modification."

    # Écriture disque
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with full_path.open("w", encoding="utf-8") as f:
            f.write(final_content)
        console.print(f"[green]✅ Fichier '{file_path}' modifié avec succès ![/green]")
        return (
            f"[SUCCESS] File '{file_path}' written successfully "
            f"({len(final_content)} bytes, {mode_desc})."
        )
    except Exception as e:
        return f"[ERROR] Failed to write file: {e}"
def execute_code_tool(file_path: str) -> str:
    """
    Exécute un fichier Python de manière sécurisée.
    Demande confirmation à l'utilisateur avant d'exécuter.
    """
    root = Path.cwd()
    full_path = root / file_path
    
    # Vérifier que le fichier existe
    if not full_path.exists() or not full_path.is_file():
        return f"[ERROR] File '{file_path}' not found."
    
    # Vérifier que c'est un fichier Python
    if not file_path.endswith('.py'):
        return f"[ERROR] Only Python (.py) files can be executed. Got: {file_path}"
    
    # Vérifier que le chemin est sûr
    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(root)):
            return f"[ERROR] Path '{file_path}' is outside the project directory."
    except Exception as e:
        return f"[ERROR] Invalid path: {e}"
    
    # Lire le contenu pour analyse
    try:
        code_content = full_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"[ERROR] Failed to read file: {e}"
    
    # Détecter les patterns dangereux
    warnings = detect_dangerous_patterns(code_content)
    
    # Afficher les informations d'exécution
    console.print(f"\n[cyan]🚀 Le modèle veut exécuter : [bold]{file_path}[/bold][/cyan]")
    console.print(f"[dim]Timeout : {EXEC_TIMEOUT}s[/dim]")
    
    if warnings:
        console.print("\n[yellow]⚠️  Avertissements de sécurité :[/yellow]")
        for warning in warnings:
            console.print(f"[dim]   {warning}[/dim]")
    
    # Afficher un aperçu du code
    preview = code_content[:300]
    if len(code_content) > 300:
        preview += f"\n... ({len(code_content) - 300} caractères supplémentaires)"
    
    console.print("\n[dim]Aperçu du code :[/dim]")
    console.print("[dim]" + "─" * 60 + "[/dim]")
    console.print(preview)
    console.print("[dim]" + "─" * 60 + "[/dim]")
    
    # Demander confirmation
    confirmation = input("Autoriser l'exécution ? (o/n) : ")
    
    if confirmation.lower() not in ['o', 'oui', 'y', 'yes']:
        return "[CANCELLED] User cancelled the execution."
    
    # Exécuter le code
    console.print(f"[dim]⏳ Exécution en cours (timeout: {EXEC_TIMEOUT}s)...[/dim]")
    log_verbose(f"Exécution de {file_path} avec timeout={EXEC_TIMEOUT}s")
    
    # Choisir le bon interpréteur Python (venv si présent)
    venv_dir = root / ".venv"
    venv_python_unix = venv_dir / "bin" / "python"
    venv_python_win = venv_dir / "Scripts" / "python.exe"
    python_exe = "python3"
    using_venv = False
    if venv_python_unix.exists():
        python_exe = str(venv_python_unix)
        using_venv = True
    elif venv_python_win.exists():
        python_exe = str(venv_python_win)
        using_venv = True
    if using_venv:
        console.print(f"[dim]🐍 Utilisation de l'environnement virtuel : {venv_dir}[/dim]")
    else:
        console.print("[dim]🐍 Aucun venv détecté, utilisation de python3 global[/dim]")
    try:
        result = subprocess.run(
            [python_exe, str(full_path)],
            cwd=str(root),  # Répertoire de travail = projet
            timeout=EXEC_TIMEOUT,
            capture_output=True,
            text=True,
            env={
                "PYTHONPATH": str(root),
                "PATH": "/usr/bin:/bin",  # PATH minimal
                **({"VIRTUAL_ENV": str(venv_dir)} if using_venv else {}),
            }
        )
        # Limiter la taille des outputs
        stdout = result.stdout[:10000] if result.stdout else ""
        stderr = result.stderr[:10000] if result.stderr else ""
        # Afficher les résultats
        if result.returncode == 0:
            console.print("[green]✅ Exécution réussie ![/green]")
        else:
            console.print(f"[yellow]⚠️  Exécution terminée avec code {result.returncode}[/yellow]")
        
        if stdout:
            console.print("\n[dim]📤 Sortie standard (stdout) :[/dim]")
            console.print("[dim]" + "─" * 60 + "[/dim]")
            console.print(stdout[:500])  # Limiter l'affichage
            if len(stdout) > 500:
                console.print(f"[dim]... ({len(stdout) - 500} caractères supplémentaires)[/dim]")
            console.print("[dim]" + "─" * 60 + "[/dim]")
        
        if stderr:
            console.print("\n[dim]📤 Erreurs (stderr) :[/dim]")
            console.print("[dim]" + "─" * 60 + "[/dim]")
            console.print(stderr[:500])
            if len(stderr) > 500:
                console.print(f"[dim]... ({len(stderr) - 500} caractères supplémentaires)[/dim]")
            console.print("[dim]" + "─" * 60 + "[/dim]")
        
        # Préparer la réponse pour le modèle
        response = f"[EXECUTION RESULT]\n"
        response += f"Return code: {result.returncode}\n"
        response += f"\nSTDOUT:\n{stdout}\n"
        response += f"\nSTDERR:\n{stderr}\n"
        
        if result.returncode == 0:
            response += "\n[SUCCESS] Execution completed successfully."
        else:
            response += f"\n[ERROR] Execution failed with return code {result.returncode}."
        
        log_verbose(
            f"Exécution terminée : returncode={result.returncode}, stdout={len(stdout)} chars, stderr={len(stderr)} chars"
        )
        return response
        
    except subprocess.TimeoutExpired:
        error_msg = (
            f"[ERROR] Execution timed out after {EXEC_TIMEOUT} seconds. "
            f"The script may contain an infinite loop or is taking too long."
        )
        console.print(f"[red]❌ {error_msg}[/red]")
        log_verbose("Timeout lors de l'exécution")
        return error_msg
    
    except Exception as e:
        error_msg = f"[ERROR] Execution failed: {e}"
        console.print(f"[red]❌ {error_msg}[/red]")
        log_verbose(f"Exception lors de l'exécution : {e}")
        return error_msg
def create_venv_tool(venv_path: str = ".venv") -> str:
    """
    Crée un environnement virtuel Python dans le projet.
    L'outil `execute_code` utilisera automatiquement ce venv s'il existe
    (par défaut : .venv à la racine du projet).
    """
    root = Path.cwd()
    venv_dir = root / venv_path
    # Vérifier que le chemin est sûr (pas de traversée de répertoire)
    try:
        venv_dir = venv_dir.resolve()
        if not str(venv_dir).startswith(str(root)):
            return f"[ERROR] Venv path '{venv_path}' is outside the project directory."
    except Exception as e:
        return f"[ERROR] Invalid venv path: {e}"
    console.print(f"\n[cyan]🐍 Le modèle veut créer un environnement virtuel : [bold]{venv_path}[/bold][/cyan]")
    console.print("[dim]Commande exécutée : python3 -m venv <venv_path>[/dim]")
    # Si le venv existe déjà, on ne le recrée pas
    if venv_dir.exists():
        console.print(f"[yellow]⚠️  L'environnement virtuel existe déjà : {venv_dir}[/yellow]")
        return f"[VENV] Virtual environment already exists at: {venv_dir}"
    # Demander confirmation
    confirmation = input("\n[bold cyan]Autoriser la création de cet environnement virtuel ? (o/n) : [/bold cyan]")
    if confirmation.lower() not in ["o", "oui", "y", "yes"]:
        return "[CANCELLED] User cancelled virtual environment creation."
    console.print(f"[dim]⏳ Création de l'environnement virtuel dans : {venv_dir}[/dim]")
    try:
        result = subprocess.run(
            ["python3", "-m", "venv", str(venv_dir)],
            cwd=str(root),
            timeout=EXEC_TIMEOUT,
            capture_output=True,
            text=True,
        )
        stdout = result.stdout[:5000] if result.stdout else ""
        stderr = result.stderr[:5000] if result.stderr else ""
        if result.returncode == 0:
            console.print("[green]✅ Environnement virtuel créé avec succès ![/green]")
        else:
            console.print(f"[yellow]⚠️  Création terminée avec code {result.returncode}[/yellow]")
        if stdout:
            console.print("\n[dim]📤 Sortie standard (stdout) :[/dim]")
            console.print("[dim]" + "─" * 60 + "[/dim]")
            console.print(stdout)
            console.print("[dim]" + "─" * 60 + "[/dim]")
        if stderr:
            console.print("\n[dim]📤 Erreurs (stderr) :[/dim]")
            console.print("[dim]" + "─" * 60 + "[/dim]")
            console.print(stderr)
            console.print("[dim]" + "─" * 60 + "[/dim]")
        response = "[VENV RESULT]\n"
        response += f"Return code: {result.returncode}\n"
        response += f"\nSTDOUT:\n{stdout}\n"
        response += f"\nSTDERR:\n{stderr}\n"
        if result.returncode == 0:
            response += "\n[SUCCESS] Virtual environment created successfully."
        else:
            response += "\n[ERROR] Failed to create virtual environment."
        log_verbose(
            f"Création du venv terminée : returncode={result.returncode}, stdout={len(stdout)} chars, stderr={len(stderr)} chars"
        )
        return response
    except subprocess.TimeoutExpired:
        error_msg = f"[ERROR] Venv creation timed out after {EXEC_TIMEOUT} seconds."
        console.print(f"[red]❌ {error_msg}[/red]")
        log_verbose("Timeout lors de la création du venv")
        return error_msg
    except Exception as e:
        error_msg = f"[ERROR] Venv creation failed: {e}"
        console.print(f"[red]❌ {error_msg}[/red]")
        log_verbose(f"Exception lors de la création du venv : {e}")
        return error_msg
# We expose these functions to Ollama as tools
TOOLS = [
    {
        "name": "list_files",
        "description": "List files in directory. Parameter: directory_path (string, optional, default='.')",
        "parameters": {
            "type": "object",
            "properties": {
                "directory_path": {"type": "string", "description": "Directory path"}
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read file content. Parameter: file_path (string, required)",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path"}
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": """Modify files with precise line control. THREE modes:
1. FULL REPLACEMENT: line_start=1, line_end=-1 (replaces entire file)  
2. REPLACE LINES: line_start=10, line_end=20 (replaces lines 10-20)
3. INSERT AT LINE: line_start=15, line_end=14 (inserts at line 15)

CRITICAL: ALWAYS read file first to count lines and understand structure.""",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file"},
                "content": {"type": "string", "description": "Code/text to write"}, 
                "line_start": {"type": "integer", "description": "Start line (1-based). For insertion at line N: use N"},
                "line_end": {"type": "integer", "description": "End line (1-based). For insertion: use line_start-1. For end of file: use -1"}
            },
            "required": ["file_path", "content", "line_start", "line_end"],
        },
    },
    {
        "name": "execute_code",
        "description": "Execute Python file. Parameter: file_path (string, required)",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Python file path"}
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "create_venv",
        "description": "Create virtual environment. Parameter: venv_path (string, optional, default='.venv')",
        "parameters": {
            "type": "object",
            "properties": {
                "venv_path": {"type": "string", "description": "Virtual environment path"}
            },
            "required": [],
        },
    },
]
def send_pending_tool_response(
    messages: List[Dict],
    function_name: str,
    tool_identifier: str,
    pending_tracker: Set[Tuple[str, str]],
) -> None:
    """Envoie immédiatement une réponse de tool indiquant l'attente utilisateur."""
    key = (function_name, tool_identifier)
    if key in pending_tracker:
        log_verbose(
            f"Pending tool response déjà envoyé pour {function_name} -> {tool_identifier}"
        )
        return
    pending_message = {
        "role": "tool",
        "name": function_name,
        "content": "[PENDING] Awaiting user confirmation..."
    }
    messages.append(pending_message)
    pending_tracker.add(key)
    console.print(
        "[dim yellow]⏳ Réponse temporaire envoyée au modèle (en attente de confirmation utilisateur).[/dim yellow]"
    )
    log_verbose(f"Pending tool response envoyé pour {function_name} -> {tool_identifier}")
   
# -----------------------------
# Chat loop
# -----------------------------
def chat_loop(files_data: Dict[str, str]):
    """
    Boucle interactive : utilisateur pose des questions,
    le modèle répond. Si le modèle demande d'appeler une fonction,
    nous la gérons ici.
    """
    # Ollama préfère un message initial "user" plutôt que "system"
    initial_context = build_prompt(files_data, native_tools=MODEL_NATIVE_TOOLS)
    
    console.print("[cyan]💬 Vous êtes maintenant connecté au modèle. Tapez votre question ou 'exit' pour quitter.[/cyan]")
    console.print(f"[dim]📦 Contexte chargé : {len(files_data)} fichiers en mémoire[/dim]")
    console.print(f"[dim]🛠️  Le modèle peut lire, modifier et exécuter des fichiers (avec votre confirmation)[/dim]")
    console.print(f"[dim]⏱️  Timeout d'exécution : {EXEC_TIMEOUT}s | Max tentatives : {MAX_RETRIES}[/dim]")
    console.print(f"[dim]💡 Tip: Le modèle utilisera les outils automatiquement, pas besoin de les demander explicitement[/dim]")

    def call_model_and_stream(current_messages: List[Dict]) -> Tuple[Dict, bool, List[Dict]]:
        """Appelle le modèle et gère le streaming, retourne le message assistant et les tool calls."""
        console.print("[dim]🤖 Le modèle réfléchit...[/dim]")
        log_verbose(f"Nombre de messages dans l'historique : {len(current_messages)}")

        try:
            log_verbose(f"Appel à ollama.chat() avec modèle {MODEL_NAME}, reasoning={REASONING_LEVEL}")
            ollama_params = {
                "model": MODEL_NAME,
                "messages": current_messages,
                "stream": True,
                "options": {
                    "num_ctx": 16384,  # Contexte maximum pour gpt-oss
                    "temperature": 0.2,  # Encore plus strict
                    "repeat_penalty": 1.3,  # Anti-répétition
                },
            }
            if MODEL_NATIVE_TOOLS:
                ollama_params["tools"] = TOOLS

            response = ollama.chat(**ollama_params)
            log_verbose("Réponse du modèle reçue, début du streaming")
        except Exception as e:
            console.print(f"[red]❌ Erreur lors de l'appel au modèle : {e}[/red]")
            log_verbose(f"Exception complète : {e}")
            return {"role": "assistant", "content": ""}, False, []

        assistant_content = ""
        thinking_content = ""
        has_tool_calls = False
        tool_calls_data: List[Dict] = []

        chunk_count = 0
        is_thinking = False

        console.print("[bold blue]Assistant> [/bold blue]")
        for chunk in response:
            chunk_count += 1
            log_verbose(f"Chunk #{chunk_count} reçu : {chunk}")

            if "message" not in chunk:
                continue

            msg = chunk["message"]

            if "thinking" in msg and msg["thinking"]:
                thinking_part = msg["thinking"]
                thinking_content += thinking_part
                if not is_thinking:
                    console.print("[dim cyan]💭 Réflexion en cours...[/dim cyan]", end="")
                    is_thinking = True
                console.print(thinking_part, end="")

            if "content" in msg and msg["content"]:
                if is_thinking:
                    console.print("\n[bold blue]💬 Réponse:[/bold blue]")
                    is_thinking = False
                content_part = msg["content"]
                assistant_content += content_part
                console.print(content_part, end="")

            if "tool_calls" in msg and msg["tool_calls"]:
                has_tool_calls = True
                tool_calls_data = msg["tool_calls"]
                log_verbose(f"Tool calls détectés : {tool_calls_data}")
                console.print(f"\n[yellow]🔧 Le modèle appelle un outil...[/yellow]")
                break

        console.print()
        log_verbose(
            f"Streaming terminé. Thinking: {len(thinking_content)} chars, Contenu: {len(assistant_content)} chars"
        )

        if not has_tool_calls and MODEL_JSON_FALLBACK:
            try:
                fallback_calls = parse_json_tool_calls(assistant_content)
            except Exception as parse_error:
                log_verbose(f"Erreur lors du parsing JSON fallback : {parse_error}")
                fallback_calls = []

            if fallback_calls:
                console.print("[yellow]🔧 Tool calls JSON détectés, conversion en format natif.[/yellow]")
                log_verbose(f"Tool calls convertis depuis JSON : {fallback_calls}")
                has_tool_calls = True
                tool_calls_data = fallback_calls
                assistant_content = ""

        assistant_message = {"role": "assistant", "content": assistant_content}
        if has_tool_calls:
            assistant_message["tool_calls"] = tool_calls_data

        return assistant_message, has_tool_calls, tool_calls_data
    messages = []
    context_sent = False
    execution_count = {}  # Track executions per question (multi-actions prêt)
    pending_tool_responses: Set[Tuple[str, str]] = set()
    tool_call_depth = 0  # Limite la profondeur des appels automatiques (compatible autonomie)
    while True:
        # Réaffiche l'invite utilisateur avant chaque interaction
        console.print("\n[bold green]Vous> [/bold green]", end="")
        
        # CORRECTION: Permettre un input multi-lignes
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "" and len(lines) > 0:
                    # Ligne vide après du contenu = fin de saisie
                    break
                lines.append(line)
                if lines and not line.endswith((".", ":", ",")):
                    # Si pas de ponctuation de continuation, demander si c'est fini
                    console.print("[dim]   (appuyez sur Entrée pour terminer, ou continuez à taper)[/dim]", end="")
                    continue
            except (KeyboardInterrupt, EOFError):
                console.print("\n[cyan]👋 Au revoir![/cyan]")
                return
        
        user_input = "\n".join(lines).strip()
        if user_input.lower() in {"exit", "quit"}:
            console.print("[cyan]👋 Au revoir![/cyan]")
            break
        
        if not user_input:
            continue
        # Reset execution count for new question
        execution_count = {}
        tool_call_depth = 0  # Reset pour nouvelle question
        # Premier message : on inclut le contexte
        if not context_sent:
            full_message = f"{initial_context}\n\n---\n\nUser question: {user_input}"
            messages.append({"role": "user", "content": full_message})
            context_sent = True
            log_verbose(f"Premier message envoyé avec contexte ({len(initial_context)} caractères)")
        else:
            messages.append({"role": "user", "content": user_input})
            log_verbose(f"Message utilisateur ajouté : {user_input[:100]}...")
        assistant_message, has_tool_calls, tool_calls_data = call_model_and_stream(messages)
        messages.append(assistant_message)

        while has_tool_calls and tool_call_depth < MAX_AUTONOMY_ITERATIONS:
            log_verbose(f"Traitement de {len(tool_calls_data)} tool call(s)")

            tool_call_depth += 1

            for call in tool_calls_data:
                function_name = call.get("function", {}).get("name", "")
                arguments = call.get("function", {}).get("arguments", {})
                
                # 🔍 LOGGING DÉTAILLÉ DES TOOL CALLS
                console.print(f"\n[cyan]📋 DEBUG Tool Call:[/cyan]")
                console.print(f"[dim]   Fonction : {function_name}[/dim]")
                console.print(f"[dim]   Arguments bruts : {arguments}[/dim]")
                console.print(f"[dim]   Type arguments : {type(arguments)}[/dim]")
                
                log_verbose(f"Tool call brut complet : {call}")
                
                # ✅ VALIDATION DES TOOL CALLS - Bloquer les hallucinations
                valid_tools = {"list_files", "read_file", "write_file", "execute_code", "create_venv"}
                if function_name not in valid_tools:
                    console.print(f"[red]   ❌ OUTIL HALLUCINÉ : {function_name} n'existe pas ![/red]")
                    
                    available_tools_list = """
AVAILABLE TOOLS (and ONLY these tools exist):

1. list_files(directory_path='.')
   - Lists files and directories in the specified path
   - Parameter: directory_path (optional, default: current directory)

2. read_file(file_path='filename.py') 
   - Reads content of a specific file
   - Parameter: file_path (required, string)

3. write_file(file_path='filename.py', content='code...', line_start=1, line_end=-1)
   - Creates or modifies files with precise line control
   - Parameters: file_path (required), content (required), line_start (required), line_end (required)
   - Examples:
     * Insert at line 50: line_start=50, line_end=49
     * Replace lines 10-20: line_start=10, line_end=20  
     * Replace entire file: line_start=1, line_end=-1

4. execute_code(file_path='script.py')
   - Executes Python files
   - Parameter: file_path (required, string)

5. create_venv(venv_path='.venv')
   - Creates Python virtual environment
   - Parameter: venv_path (optional, default: .venv)

CRITICAL: Do NOT invent tools like 'apply_patch', 'git_diff', 'patch_file', etc. 
ONLY use the tools listed above. No exceptions."""

                    result = f"[ERROR] Tool '{function_name}' does not exist!\n{available_tools_list}"
                    
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "name": function_name  # Garder le nom halluciné pour que le modèle comprenne
                    })
                    continue  # Passer au tool call suivant
                
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                        console.print(f"[dim]   Arguments après parsing : {arguments}[/dim]")
                    except json.JSONDecodeError as e:
                        console.print(f"[red]   ❌ Erreur parsing JSON : {e}[/red]")
                        result = f"[ERROR] Invalid JSON arguments: {e}"
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "name": function_name
                        })
                        continue
                
                if function_name == "read_file":
                    file_path = arguments.get("file_path") or arguments.get("path", "")
                    
                    # ⚠️ VALIDATION STRICTE: read_file ne prend QUE file_path
                    invalid_params = [key for key in arguments.keys() if key not in {"file_path", "path"}]
                    if invalid_params:
                        console.print(f"[red]   ❌ PARAMÈTRES INCORRECTS pour read_file: {invalid_params}[/red]")
                        result = (
                            f"[ERROR] read_file called with invalid parameters: {invalid_params}\n"
                            "read_file ONLY accepts 'file_path' parameter.\n"
                            "Do NOT use line_start, line_end, or any other parameters with read_file!\n"
                            "Correct usage: read_file(file_path='filename.py')"
                        )
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "name": function_name
                        })
                        continue
                    
                    if not file_path:
                        console.print(f"[red]   ❌ ARGUMENT MANQUANT: file_path est vide ou absent ![/red]")
                        console.print(f"[red]   Arguments reçus : {arguments}[/red]")
                        result = "[ERROR] Missing required argument 'file_path'. The model did not provide the file path to read."
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "name": function_name
                        })
                        continue
                    
                    console.print(f"[green]   ✓ Fichier détecté : {file_path}[/green]")
                    console.print(f"[dim]   📂 Lecture de : {file_path}[/dim]")
                    file_content = read_file_tool(file_path)
                    log_verbose(f"Fichier lu : {len(file_content)} caractères")
                    
                    # Ajouter le résultat de l'outil
                    messages.append({
                        "role": "tool",
                        "content": file_content,
                        "name": function_name
                    })
                    
                elif function_name == "list_files":
                    directory_path = arguments.get("directory_path", ".")
                    
                    console.print(f"[green]   ✓ Répertoire détecté : {directory_path}[/green]")
                    console.print(f"[dim]   📂 Listage de : {directory_path}[/dim]")
                    file_list = list_files_tool(directory_path)
                    log_verbose(f"Répertoire listé : {len(file_list)} caractères")
                    
                    # Ajouter le résultat de l'outil
                    messages.append({
                        "role": "tool",
                        "content": file_list,
                        "name": function_name
                    })
                    
                elif function_name == "write_file":
                    file_path = arguments.get("file_path") or arguments.get("path", "")
                    content = arguments.get("content", "")
                    # ➕ Récupération des arguments OBLIGATOIRES de lignes
                    raw_line_start = arguments.get("line_start")
                    raw_line_end = arguments.get("line_end")
                    
                    # Vérifications
                    if not file_path:
                        console.print(
                            f"[red]   ❌ ARGUMENT MANQUANT: file_path/path est vide ou absent ![/red]"
                        )
                        console.print(f"[red]   Arguments reçus : {arguments}[/red]")
                        result = (
                            "[ERROR] Missing required argument 'file_path' or 'path'. "
                            "The model must specify which file to write to."
                        )
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "name": function_name
                        })
                        continue

                    if content is None or content.strip() == "":
                        console.print(
                            f"[red]   ❌ Refus d'écrire un contenu vide dans '{file_path}'[/red]"
                        )
                        console.print(f"[red]   Arguments reçus : {arguments}[/red]")
                        result = (
                            "[ERROR] MISSING CONTENT PARAMETER! The 'write_file' tool was called with empty content.\n"
                            "You MUST provide the 'content' parameter with the actual code/text to write."
                        )
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "name": function_name
                        })
                        continue
                        
                    # Vérification des paramètres OBLIGATOIRES line_start/line_end
                    if raw_line_start is None or raw_line_end is None:
                        console.print(
                            f"[red]   ❌ ARGUMENTS MANQUANTS: line_start et line_end sont obligatoires ![/red]"
                        )
                        console.print(f"[red]   Arguments reçus : {arguments}[/red]")
                        result = (
                            "[ERROR] MISSING line_start AND line_end PARAMETERS!\n"
                            "The write_file tool now requires BOTH line_start and line_end parameters.\n"
                            "Examples:\n"
                            "- Insert at line 50: line_start=50, line_end=49\n"
                            "- Replace lines 10-20: line_start=10, line_end=20\n"
                            "- Replace entire file: line_start=1, line_end=-1\n"
                            f"Current arguments: {arguments}"
                        )
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "name": function_name
                        })
                        continue

                    # Conversion robuste des numéros de ligne
                    try:
                        line_start = int(raw_line_start)
                        line_end = int(raw_line_end)
                    except (ValueError, TypeError):
                        console.print(
                            f"[red]   ❌ line_start/line_end invalides: {raw_line_start}, {raw_line_end}[/red]"
                        )
                        result = f"[ERROR] Invalid line_start or line_end: must be integers. Got: {raw_line_start}, {raw_line_end}"
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "name": function_name
                        })
                        continue

                    console.print(f"[green]   ✓ Fichier détecté : {file_path}[/green]")
                    
                    if line_start > line_end:
                        mode_msg = f"insertion ligne {line_start}"
                    elif line_start == 1 and line_end == -1:
                        mode_msg = "remplacement complet du fichier"
                    else:
                        mode_msg = f"remplacement lignes {line_start}-{line_end}"
                        
                    console.print(
                        f"[dim]   ✏️  Écriture ({mode_msg}) dans : {file_path}[/dim]"
                    )

                    # Appel réel avec gestion de lignes obligatoire
                    result = write_file_tool(file_path, content, line_start, line_end)
                    log_verbose(f"Résultat de l'écriture : {result}")

                    # Ajouter le résultat de l'outil
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "name": function_name
                    })
                    
                elif function_name == "execute_code":
                    file_path = arguments.get("file_path") or arguments.get("path", "")
                    
                    if not file_path:
                        console.print(f"[red]   ❌ ARGUMENT MANQUANT: file_path/path est vide ou absent ![/red]")
                        console.print(f"[red]   Arguments reçus : {arguments}[/red]")
                        result = "[ERROR] Missing required argument 'file_path' or 'path'. The model must specify which file to execute."
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "name": function_name
                        })
                        continue
                    
                    console.print(f"[green]   ✓ Fichier détecté : {file_path}[/green]")
                    
                    # Vérifier la limite de tentatives
                    execution_count[file_path] = execution_count.get(file_path, 0) + 1
                    
                    if execution_count[file_path] > MAX_RETRIES:
                        result = f"[ERROR] Maximum execution attempts ({MAX_RETRIES}) reached for {file_path}. Please review the code manually."
                        console.print(f"[red]   ❌ Limite de tentatives atteinte ({MAX_RETRIES})[/red]")
                        log_verbose(f"Limite d'exécution atteinte pour {file_path}")
                    else:
                        send_pending_tool_response(
                            messages,
                            function_name,
                            file_path,
                            pending_tool_responses,
                        )
                        console.print(f"[dim]   🚀 Exécution de : {file_path} (tentative {execution_count[file_path]}/{MAX_RETRIES})[/dim]")
                        result = execute_code_tool(file_path)
                        log_verbose(f"Résultat de l'exécution : {result[:200]}...")
                    
                    # Ajouter le résultat de l'outil
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "name": function_name
                    })
                    pending_tool_responses.discard((function_name, file_path))
                    
                elif function_name == "create_venv":
                    venv_path = arguments.get("venv_path") or ".venv"
                    send_pending_tool_response(
                        messages,
                        function_name,
                        venv_path,
                        pending_tool_responses,
                    )
                    console.print(f"[green]   ✓ Chemin d'environnement virtuel : {venv_path}[/green]")
                    console.print(f"[dim]   🐍 Création de l'environnement virtuel : {venv_path}[/dim]")
                    result = create_venv_tool(venv_path)
                    log_verbose(f"Résultat de la création du venv : {result[:200]}...")
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "name": function_name
                    })
                    pending_tool_responses.discard((function_name, venv_path))
                    
                else:
                    console.print(f"[red]   ❌ Outil inconnu : {function_name}[/red]")
            

            if AUTONOMY and tool_call_depth >= MAX_AUTONOMY_ITERATIONS:
                console.print(
                    "[yellow]⚠️  Limite d'autonomie atteinte, arrêt des appels automatiques supplémentaires.[/yellow]"
                )
                break

            if not AUTONOMY and tool_call_depth >= 5:
                console.print(
                    "[yellow]⚠️  Le modèle a fait plusieurs actions. "
                    "Tapez 'continuer' pour qu'il continue, ou posez une nouvelle question.[/yellow]"
                )
                console.print("[bold green]Vous> [/bold green]", end="")
                continue_input = input()
                if continue_input.lower() not in {"continuer", "continue", "c"}:
                    user_input = continue_input
                    messages.append({"role": "user", "content": user_input})
                    execution_count = {}
                    tool_call_depth = 0
                    assistant_message, has_tool_calls, tool_calls_data = call_model_and_stream(messages)
                    messages.append(assistant_message)
                    if not has_tool_calls:
                        break
                else:
                    log_verbose("L'utilisateur a confirmé la poursuite des actions.")

            assistant_message, has_tool_calls, tool_calls_data = call_model_and_stream(messages)
            messages.append(assistant_message)

            if not (AUTONOMY and has_tool_calls and tool_call_depth < MAX_AUTONOMY_ITERATIONS):
                if AUTONOMY and has_tool_calls and tool_call_depth >= MAX_AUTONOMY_ITERATIONS:
                    console.print(
                        "[yellow]⚠️  Arrêt automatique: limite d'itérations atteinte avant nouveaux tool calls.[/yellow]"
                    )
                break

        log_verbose("Réponse de l'assistant ajoutée à l'historique")
# -----------------------------
# Main
# -----------------------------
def main():
    global VERBOSE, REASONING_LEVEL, EXEC_TIMEOUT, MAX_RETRIES, MODEL_NAME, MODEL_NATIVE_TOOLS, MODEL_JSON_FALLBACK, AUTONOMY
    
    parser = argparse.ArgumentParser(description="Ollama Code-Assistant")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Répertoire contenant le code à analyser (par défaut le répertoire courant).",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Nom du modèle Ollama à utiliser (par défaut: %(default)s).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Active le mode verbeux (affiche tous les détails de debug)",
    )
    parser.add_argument(
        "-r", "--reasoning",
        choices=["low", "medium", "high"],
        default="medium",
        help="Niveau de réflexion du modèle (low=rapide, medium=équilibré, high=approfondi)",
    )
    parser.add_argument(
        "-t", "--exec-timeout",
        type=int,
        default=30,
        help="Timeout pour l'exécution de code en secondes (par défaut: 30)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Nombre maximum de tentatives d'exécution par fichier (par défaut: 3)",
    )
    parser.add_argument(
        "--autonomy",
        action="store_true",
        help="Active l'autonomie: le modèle enchaîne automatiquement les tool calls sans attente utilisateur",
    )
    args = parser.parse_args()
    VERBOSE = args.verbose
    REASONING_LEVEL = args.reasoning
    EXEC_TIMEOUT = args.exec_timeout
    MAX_RETRIES = args.max_retries
    MODEL_NAME = args.model
    AUTONOMY = args.autonomy

    compat = MODEL_COMPAT.get(MODEL_NAME, DEFAULT_MODEL_COMPAT)
    MODEL_NATIVE_TOOLS = compat.get("native_tools", DEFAULT_MODEL_COMPAT["native_tools"])
    MODEL_JSON_FALLBACK = compat.get("json_fallback", DEFAULT_MODEL_COMPAT["json_fallback"])

    if VERBOSE:
        console.print("[magenta]🔍 Mode VERBOSE activé[/magenta]")
    
    console.print(f"\n[bold cyan]🚀 Ollama Code-Assistant[/bold cyan]")
    console.print(f"[dim]Modèle : {MODEL_NAME}[/dim]")
    console.print(f"[dim]Niveau de réflexion : {REASONING_LEVEL}[/dim]")
    console.print(f"[dim]Timeout d'exécution : {EXEC_TIMEOUT}s[/dim]")
    console.print(f"[dim]Max tentatives : {MAX_RETRIES}[/dim]\n")
    console.print(
        f"[dim]Compat tools natifs : {MODEL_NATIVE_TOOLS} | JSON fallback : {MODEL_JSON_FALLBACK}[/dim]"
    )
    root = Path(args.directory).resolve()
    if not root.is_dir():
        console.print(f"[red]❌ Erreur : {root} n'est pas un répertoire valide.[/red]")
        sys.exit(1)
    files_data = collect_files(root)
    if not files_data:
        console.print("[red]❌ Aucun fichier pertinent trouvé. Arrêt.[/red]")
        sys.exit(1)
    # On passe en mode chat
    chat_loop(files_data)
if __name__ == "__main__":
    main()
