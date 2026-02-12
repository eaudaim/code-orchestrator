#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code-Assistant avec Ollama (gpt-oss-20b-64k:latest)
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
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import ollama
from rich.console import Console
from rich.markdown import Markdown
from core.model import call_model_and_stream, set_model_context
from core.prompt import build_prompt, set_prompt_console
from utils.file_utils import collect_files, read_file
from runtime.executor import *  # noqa: F403
from runtime.executor import set_executor_environment
from runtime.loop import autonomy_loop, set_loop_environment

console = Console(force_terminal=True)
set_prompt_console(console)
# -----------------------------
# Configuration
# -----------------------------
try:
    import config.settings as cfg
    MODEL_NAME = cfg.MODEL_NAME
    MODEL_COMPAT = cfg.MODEL_COMPAT
    DEFAULT_MODEL_COMPAT = cfg.DEFAULT_MODEL_COMPAT
    MAX_BYTES_PER_FILE = cfg.MAX_BYTES_PER_FILE
    MAX_TOTAL_BYTES = cfg.MAX_TOTAL_BYTES
    SCRIPT_NAME = cfg.SCRIPT_NAME
    VERBOSE = cfg.VERBOSE
    REASONING_LEVEL = cfg.REASONING_LEVEL
    EXEC_TIMEOUT = cfg.EXEC_TIMEOUT
    MAX_RETRIES = cfg.MAX_RETRIES
    MAX_AUTONOMY_ITERATIONS = cfg.MAX_AUTONOMY_ITERATIONS
    AUTONOMY_TIMEOUT = cfg.AUTONOMY_TIMEOUT
    AUTONOMY = cfg.AUTONOMY
    get_model_config = cfg.get_model_config
except Exception as exc:
    console.print(
        f"[yellow]⚠️ Impossible d'importer config.settings : {exc}. Utilisation des valeurs par défaut.[/yellow]"
    )
    MODEL_NAME = "gpt-oss-20b-64k:latest"
    MODEL_COMPAT = {
        "gpt-oss-20b-64k:latest": {"native_tools": True, "json_fallback": False},
        "qwen2.5-coder:14b": {"native_tools": False, "json_fallback": True},
    }
    DEFAULT_MODEL_COMPAT = {"native_tools": True, "json_fallback": False}
    MAX_BYTES_PER_FILE = 500 * 1024  # 500 KB par défaut
    MAX_TOTAL_BYTES = 5 * 1024 * 1024  # 5 Mo max total envoyés au modèle
    SCRIPT_NAME = Path(__file__).name  # Nom du script à exclure
    VERBOSE = False  # Mode verbeux (défini par argument CLI)
    REASONING_LEVEL = "medium"  # Niveau de réflexion : low, medium, high
    EXEC_TIMEOUT = 30  # Timeout pour l'exécution de code (en secondes)
    MAX_RETRIES = 3  # Nombre maximum de tentatives d'exécution
    MAX_AUTONOMY_ITERATIONS = 20  # Nombre max d'itérations autonomes
    AUTONOMY_TIMEOUT = 5  # Timeout entre itérations autonomes (secondes)
    AUTONOMY = True  # Mode autonomie (enchaînement automatique des tool calls)

    def get_model_config(model_name: str):
        return MODEL_COMPAT.get(model_name, DEFAULT_MODEL_COMPAT)

model_config = get_model_config(MODEL_NAME)
MODEL_NATIVE_TOOLS = model_config["native_tools"]
MODEL_JSON_FALLBACK = model_config["json_fallback"]
# -----------------------------
# Helpers
# -----------------------------
def log_verbose(message: str):
    """Affiche un message uniquement si VERBOSE est activé."""
    if VERBOSE:
        console.print(f"[dim magenta]🔍 DEBUG: {message}[/dim magenta]")


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
    {
        "name": "git_init",
        "description": "Initialize a git repository if it is not already present.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "git_commit",
        "description": "Stage all files and create a commit with the provided message.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message (non-empty)"}
            },
            "required": ["message"],
        },
    },
    {
        "name": "git_rollback",
        "description": "Reset repository to a previous commit using git reset --hard HEAD~<steps> (steps must be > 0).",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of commits to roll back (positive integer)",
                    "minimum": 1,
                    "default": 1,
                }
            },
            "required": [],
        },
    },
    {
        "name": "git_history",
        "description": "Show the 10 most recent commits in one-line format.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
set_model_context(
    TOOLS,
    MODEL_NATIVE_TOOLS,
    MODEL_JSON_FALLBACK,
    debug_logger=log_verbose,
    shared_console=console,
    json_parser=parse_json_tool_calls,
)
# -----------------------------
# Chat loop
# -----------------------------
def chat_loop(files_data: Dict[str, str]):
    """
    Boucle interactive : utilisateur pose des questions,
    le modèle répond. Si le modèle demande d'appeler une fonction,
    nous la gérons ici.
    """
    global AUTONOMY

    initial_context = build_prompt(files_data, native_tools=MODEL_NATIVE_TOOLS)

    console.print("[cyan]💬 Vous êtes maintenant connecté au modèle. Tapez votre question ou 'exit' pour quitter.[/cyan]")
    console.print(f"[dim]📦 Contexte chargé : {len(files_data)} fichiers en mémoire[/dim]")
    console.print(f"[dim]🛠️  Le modèle peut lire, modifier et exécuter des fichiers (avec votre confirmation)[/dim]")
    console.print(f"[dim]⏱️  Timeout d'exécution : {EXEC_TIMEOUT}s | Max tentatives : {MAX_RETRIES}[/dim]")
    console.print(f"[dim]💡 Tip: Le modèle utilisera les outils automatiquement, pas besoin de les demander explicitement[/dim]")

    messages = []
    context_sent = False

    while True:
        console.print("\n[bold green]Vous> [/bold green]", end="")

        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "" and len(lines) > 0:
                    break
                lines.append(line)
                if lines and not line.endswith((".", ":", ",")):
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

        if not context_sent:
            full_message = f"{initial_context}\n\n---\n\nUser question: {user_input}"
            messages.append({"role": "user", "content": full_message})
            context_sent = True
            log_verbose(f"Premier message envoyé avec contexte ({len(initial_context)} caractères)")
        else:
            messages.append({"role": "user", "content": user_input})
            log_verbose(f"Message utilisateur ajouté : {user_input[:100]}...")

        autonomy_loop(messages, files_data)
# -----------------------------
# Main
# -----------------------------
def main():
    global VERBOSE, REASONING_LEVEL, EXEC_TIMEOUT, MAX_RETRIES, MODEL_NAME, MODEL_NATIVE_TOOLS, MODEL_JSON_FALLBACK, AUTONOMY
    
    parser = argparse.ArgumentParser(
        description="Ollama Code-Assistant (utilisez --autonomy pour enchaîner les tool calls sans confirmation)"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Répertoire contenant le code à analyser (par défaut le répertoire courant).",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Répertoire de travail à analyser (par défaut: répertoire courant de l'orchestrateur)",
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

    work_dir = Path(args.work_dir).resolve() if args.work_dir else Path(args.directory).resolve()

    if not work_dir.is_dir():
        console.print(f"[red]❌ Work directory not found: {work_dir}[/red]")
        sys.exit(1)

    set_executor_environment(
        console,
        log_verbose,
        VERBOSE,
        EXEC_TIMEOUT,
        MAX_RETRIES,
        AUTONOMY,
        SCRIPT_NAME,
        work_dir,
    )
    set_loop_environment(
        console,
        log_verbose,
        MODEL_NAME,
        REASONING_LEVEL,
        MAX_AUTONOMY_ITERATIONS,
        AUTONOMY_TIMEOUT,
        AUTONOMY,
        EXEC_TIMEOUT,
        MAX_RETRIES,
    )

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
    files_data = collect_files(work_dir)
    if not files_data:
        console.print(
            "[yellow]ℹ️  Aucun fichier pertinent collecté ; démarrage avec un contexte initial vide.[/yellow]"
        )
    # On passe en mode chat
    chat_loop(files_data)
if __name__ == "__main__":
    main()
