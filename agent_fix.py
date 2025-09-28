#!/usr/bin/env python3
# agent_fix.py — Agent local "réparateur" pour projet Python (GPT-OSS)
# - Streaming + logs + timeouts
# - Outils : list_files, read_file, apply_patch (find→replace), apply_edit_b64, run_harness
# - Enveloppes d'erreurs {"ok": true/false} pour TOUS les tools
# - Feedback d'erreur au modèle + anti-répétition d'appels invalides
# - Conçu pour bosser avec auto_harness.py agrégé

import ast
import base64
import difflib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import ollama

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────────────
MODEL = "gpt-oss:20b"
PROJECT_DIR = Path(".").resolve()
SRC_DIR = PROJECT_DIR / "src"

VERBOSE = True
SHOW_THINK = False       # coupe l'affichage verbeux du "thinking"
TEMPERATURE = 0.2
MAX_TURNS = 12

DEBUG_RAW_CHUNKS = False
STREAM_TIMEOUT_SEC = 120
IDLE_TIMEOUT_SEC = 30

SYSTEM = """Tu es l'agent de réparation Python. Objectif : faire réussir 'python auto_harness.py' (exit=0) sans tricher.
Outils autorisés : list_files, read_file, simple_patch, apply_edit_b64, run_harness. Modifie uniquement src/*.py et fais des corrections minimales.
Interdits : changer auto_harness.py, tests/, requirements.txt, pyproject.toml, ou masquer les erreurs (ex: return True).

Règles de travail STRICTES :
1. Lance run_harness pour identifier les erreurs, puis lis au plus UNE fois chaque fichier pertinent (maximum 2 read_file par fichier sur tout le run).
2. Après run_harness et une lecture ciblée, passe immédiatement à l'action : appelle simple_patch sans attendre. Utilise find/replace précis, par ex. simple_patch("src/numops/utils.py", "return s  # laisse passer", "raise ValueError(f'not a valid integer: {s!r}')").
3. Préfère simple_patch(path, find, replace). N'utilise apply_edit_b64 qu'en dernier recours.
4. Après chaque modification, relance run_harness pour vérifier la correction.
5. Respecte les schémas des outils et évite les boucles de lecture ou d'analyse sans action.

Agis vite, réponds brièvement, pense en termes de corrections concrètes. Ta priorité est d'appliquer des patches efficaces dès les tours 2-3.
"""

# ───────────────────────────────────────────────────────────────────────────────
# GUARDS & FEEDBACK (points 1 et 2)
# ───────────────────────────────────────────────────────────────────────────────
ERROR_BUDGET = 10          # nb max d'erreurs d'outils tolérées sur tout le run
RETRY_PER_TOOL = 2         # nb max de répétitions d'un même appel invalide
_last_bad_calls = {}       # (tool_name, json_args) -> count

def _ok(payload):
    """Envelope succès outil → format stable pour le modèle."""
    return {"ok": True, "result": payload}

def _err(tool, code, message, expected=None, args=None):
    """Envelope erreur outil (structurée et actionnable par le modèle)."""
    return {"ok": False, "error": {
        "tool": tool, "code": code, "message": message,
        "expected": expected, "args": args
    }}

# ───────────────────────────────────────────────────────────────────────────────
# ÉTAT DE L'AGENT & UTILS
# ───────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentState:
    files_read: set = field(default_factory=set)
    errors_identified: List[str] = field(default_factory=list)
    patches_applied: int = 0

    def record_errors(self, errors: List[str]):
        for err in errors:
            if err not in self.errors_identified:
                self.errors_identified.append(err)


class ReadLimiter:
    """Implémente une stratégie progressive de restriction des lectures."""

    MAX_TRACKED_FILES = 20
    MAX_TOTAL_READS = 3
    MAX_RECOVERY_TOKENS = 2

    def __init__(self):
        self.turn = 0
        self.stage = 0
        self.total_reads_per_file: Dict[str, int] = {}
        self.file_order: Deque[str] = deque()
        self.recovery_allowances: Dict[str, int] = {}
        self.recovery_granted: Dict[str, int] = {}
        self.last_failed_patch: Optional[str] = None

    def _stage_for_turn(self, turn: int) -> int:
        if turn <= 3:
            return 1  # libre
        if turn <= 6:
            return 2  # limité
        return 3  # blocage

    def start_turn(self, turn: int):
        """Met à jour le tour courant et log la phase active."""
        self.turn = turn
        new_stage = self._stage_for_turn(turn)
        if new_stage != self.stage:
            self.stage = new_stage
            if self.stage == 1:
                print("[ReadLimiter] Phase 1: lectures libres (limite absolue 3 par fichier).")
            elif self.stage == 2:
                print("[ReadLimiter] Phase 2: limitation stricte à 2 lectures par fichier.")
            else:
                print("[ReadLimiter] Phase 3: blocage des lectures (sauf récupération après échec de patch).")

    def _ensure_capacity(self, path: str):
        if path in self.total_reads_per_file:
            return
        if len(self.total_reads_per_file) >= self.MAX_TRACKED_FILES:
            oldest = self.file_order.popleft()
            self.total_reads_per_file.pop(oldest, None)
            self.recovery_allowances.pop(oldest, None)
            self.recovery_granted.pop(oldest, None)
            print(f"[ReadLimiter] Capacité atteinte, purge du suivi pour {oldest}.")
        self.file_order.append(path)
        self.total_reads_per_file[path] = 0

    def _increment_read_count(self, path: str):
        self._ensure_capacity(path)
        self.total_reads_per_file[path] = self.total_reads_per_file.get(path, 0) + 1

    def _consume_recovery_token(self, path: str) -> bool:
        tokens = self.recovery_allowances.get(path, 0)
        if tokens <= 0:
            return False
        tokens -= 1
        if tokens <= 0:
            self.recovery_allowances.pop(path, None)
        else:
            self.recovery_allowances[path] = tokens
        print(f"[ReadLimiter] Lecture de récupération autorisée pour {path} (jetons restants: {tokens}).")
        return True

    def request_read(self, path: str):
        """Vérifie si la lecture est autorisée pour ce tour."""
        if not path:
            return True, None

        current_reads = self.total_reads_per_file.get(path, 0)
        if current_reads >= self.MAX_TOTAL_READS:
            message = (
                f"Lecture refusée pour {path}: limite absolue de {self.MAX_TOTAL_READS} lectures atteinte."
            )
            return False, message

        stage = self.stage or self._stage_for_turn(self.turn)

        if stage == 1:
            self._increment_read_count(path)
            return True, None

        if stage == 2:
            if current_reads >= 2:
                if self._consume_recovery_token(path):
                    self._increment_read_count(path)
                    return True, None
                message = (
                    f"Lecture refusée pour {path}: limite de 2 lectures atteinte en phase 2."
                )
                return False, message
            self._increment_read_count(path)
            remaining = max(0, 2 - self.total_reads_per_file.get(path, 0))
            warning = (
                f"[ReadLimiter] Tour {self.turn}: lecture autorisée pour {path}. "
                f"Il reste {remaining} lecture(s) avant blocage phase 2."
            )
            print(warning)
            return True, None

        # Phase 3 (tour >= 7)
        if self._consume_recovery_token(path):
            self._increment_read_count(path)
            return True, None
        message = (
            f"Lecture bloquée pour {path}: phase 3 active, effectue un patch avant de relire."
        )
        if self.last_failed_patch and self.last_failed_patch != path:
            message += f" Dernier patch en échec sur {self.last_failed_patch}."
        return False, message

    def record_patch_result(self, path: str, result: dict):
        if not path or not isinstance(result, dict):
            return

        applied = bool(result.get("applied"))
        had_error = "error" in result

        if applied and not had_error:
            self.recovery_allowances.pop(path, None)
            self.last_failed_patch = None
            return

        tokens_granted = self.recovery_granted.get(path, 0)
        if tokens_granted >= self.MAX_RECOVERY_TOKENS:
            self.last_failed_patch = path
            detail = result.get("error") or result.get("warning")
            if detail:
                print(
                    f"[ReadLimiter] Patch en échec sur {path}, mais quota de récupération épuisé ({detail})."
                )
            else:
                print(
                    f"[ReadLimiter] Patch en échec sur {path}, mais quota de récupération épuisé."
                )
            return

        self.recovery_allowances[path] = self.recovery_allowances.get(path, 0) + 1
        self.recovery_granted[path] = tokens_granted + 1
        self.last_failed_patch = path
        detail = result.get("error") or result.get("warning")
        suffix = f" ({detail})" if detail else ""
        print(
            f"[ReadLimiter] Patch en échec sur {path}, octroi d'une lecture de récupération.{suffix}"
        )

        # Nettoyage des entrées nulles
        if self.recovery_allowances.get(path, 0) <= 0:
            self.recovery_allowances.pop(path, None)

# ───────────────────────────────────────────────────────────────────────────────
# UTILS
# ───────────────────────────────────────────────────────────────────────────────
def run(cmd, timeout=90, env=None):
    p = subprocess.run(
        cmd, cwd=str(PROJECT_DIR), text=True,
        capture_output=True, timeout=timeout, env=env
    )
    return {
        "returncode": p.returncode,
        "stdout": (p.stdout or "")[-12000:],
        "stderr": (p.stderr or "")[-12000:],
    }

def list_files():
    excluded_exact = {
        ".git",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "venv",
        "env",
        "site-packages",
        "__pycache__",
        "build",
        "dist",
    }
    excluded_prefixes = (".venv",)
    files = []
    for root, dirs, filenames in os.walk(PROJECT_DIR):
        pruned_dirs = []
        for d in dirs:
            if any(d.startswith(prefix) for prefix in excluded_prefixes):
                continue
            if d in excluded_exact:
                continue
            pruned_dirs.append(d)
        dirs[:] = pruned_dirs
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            full_path = Path(root) / filename
            files.append(str(full_path.relative_to(PROJECT_DIR)))
    return {"files": files}

def read_file(path: str):
    fp = (PROJECT_DIR / path).resolve()
    if not fp.is_file() or PROJECT_DIR not in fp.parents:
        return {"error": "invalid path"}
    if fp.stat().st_size > 1_000_000:
        return {"error": "file too large"}
    return {"path": path, "content": fp.read_text(encoding="utf-8", errors="ignore")}

def _guard_src_py(path: str):
    fp = (PROJECT_DIR / path).resolve()
    if PROJECT_DIR not in fp.parents or not str(fp).startswith(str(SRC_DIR)):
        return fp, {"error": f"forbidden path: {path}"}
    if not fp.suffix == ".py":
        return fp, {"error": f"not a .py file: {path}"}
    return fp, None

def _write_src_file(path: str, content: str):
    fp, err = _guard_src_py(path)
    if err:
        return err
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8")
    return {"applied": [path]}

def apply_edit_b64(path: str, content_b64: str):
    """Remplace entièrement un fichier sous src/ avec contenu en Base64 (JSON-safe)."""
    try:
        raw_bytes = base64.b64decode(content_b64.encode("ascii"), validate=True)
        raw = raw_bytes.decode("utf-8", errors="strict")
    except Exception as e:
        return {"error": f"base64 decode failed: {e}"}

    fp, err = _guard_src_py(path)
    if err:
        return err

    original = ""
    if fp.exists():
        original = fp.read_text(encoding="utf-8", errors="ignore")

    ok, detail = _check_python_syntax(raw, path)
    if not ok:
        return {"error": "syntax_error", "detail": detail, "path": path}

    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(raw, encoding="utf-8")
    return {"applied": [path], "diff_preview": _make_unified_diff(original, raw, path)}

def _make_unified_diff(old: str, new: str, path: str):
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}"
    )
    preview = "".join(list(diff))[:4000]
    return preview


def _check_python_syntax(text: str, path_label: str) -> Tuple[bool, str]:
    try:
        ast.parse(text, filename=path_label)
        return True, "syntax ok"
    except SyntaxError as e:
        loc = f"{getattr(e, 'filename', path_label)}:{getattr(e, 'lineno', '?')}:{getattr(e, 'offset', '?')}"
        return False, f"SyntaxError at {loc}: {e.msg}"

def apply_patch(path: str, edits: list):
    """
    Applique des remplacements "find -> replace" sur le contenu actuel.
    edits: [{"find": "...", "replace": "..."}]
    Retourne un aperçu de diff.
    """
    fp, err = _guard_src_py(path)
    if err:
        return err
    if not fp.exists():
        return {"error": f"file not found: {path}"}

    original = fp.read_text(encoding="utf-8", errors="ignore")

    if not isinstance(edits, list):
        return {"error": "invalid_edits_type", "message": "edits must be a list"}
    if not edits:
        return {"error": "empty_edits", "message": "edits cannot be empty"}

    normalized = []
    for idx, e in enumerate(edits):
        if not isinstance(e, dict):
            return {"error": "invalid_edit_item", "index": idx, "message": "each edit must be an object"}
        find = e.get("find")
        if not isinstance(find, str):
            return {"error": "invalid_find", "index": idx, "message": "'find' must be a string"}
        replace = e.get("replace", "")
        if not isinstance(replace, str):
            return {"error": "invalid_replace", "index": idx, "message": "'replace' must be a string"}
        normalized.append((find, replace))

    updated = original
    applied_count = 0
    for find, replace in normalized:
        occurrences = updated.count(find)
        if occurrences:
            applied_count += occurrences
        updated = updated.replace(find, replace)

    if applied_count == 0:
        return {"warning": "no changes applied (find strings not found?)"}

    ok, detail = _check_python_syntax(updated, path)
    if not ok:
        return {"error": "syntax_error", "detail": detail, "path": path}

    fp.write_text(updated, encoding="utf-8")
    return {
        "applied": [path],
        "diff_preview": _make_unified_diff(original, updated, path)
    }

ERROR_HINTS = {
    "utils.to_int": "Dans src/numops/utils.py -> to_int(s): après strip, si s n'est pas un entier décimal, lever ValueError (ne pas retourner s).",
    "utils.merge_dicts": "Dans src/numops/utils.py -> merge_dicts(a,b): fusionner en additionnant les valeurs des clés communes (pas simple override {**a, **b}).",
    "utils.append_item": "Dans src/numops/utils.py -> append_item: paramètre par défaut doit être None et créer une nouvelle liste si None (pas bucket = []).",
    "stats.mean": "Dans src/numops/stats.py -> mean: utiliser une division flottante sum(values) / len(values), pas //.",
    "stats.median": "Dans src/numops/stats.py -> median: si n est pair, renvoyer la moyenne (vals[mid-1] + vals[mid]) / 2.",
    "core.running_total": "Dans src/numops/core.py -> running_total: accumuler puis append le cumul (ne pas préfixer par 0).",
    "cli.mypy": "Dans src/numops/cli.py: typer ou séparer arr_i (int) pour runtotal et arr_f (float) pour mean pour satisfaire mypy.",
}


def detect_errors(text: str) -> List[str]:
    """Analyse la sortie de l'harness et identifie les erreurs spécifiques."""
    if not text:
        return []

    errors = []
    if re.search(r"src\.numops\.utils\.to_int", text) and "ValueError" in text and "Got:\n    'x'" in text:
        errors.append("utils.to_int")
    if re.search(r"src\.numops\.utils\.merge_dicts", text) and '== {"a": 1, "b": 5, "c": 4}' in text:
        errors.append("utils.merge_dicts")
    if re.search(r"src\.numops\.utils\.append_item", text) and "Expected:\n    ['b']" in text:
        errors.append("utils.append_item")
    if re.search(r"src\.numops\.stats\.mean", text) and "Expected:\n    0.2" in text and "Got:\n    0.0" in text:
        errors.append("stats.mean")
    if re.search(r"src\.numops\.stats\.median", text) and "median([1, 2, 3, 4])" in text and "Expected:\n    2.5" in text:
        errors.append("stats.median")
    if re.search(r"src\.numops\.core\.running_total", text) and "Expected:\n    [1, 3, 6]" in text and "Got:\n    [0, 1, 3, 6]" in text:
        errors.append("core.running_total")
    if 'mypy' in text and 'src/numops/cli.py' in text and 'mean(arr)' in text:
        errors.append("cli.mypy")
    return errors


def analyze_harness_output(text: str) -> str:
    """Transforme la sortie de l'harness en suggestions actionnables pour le modèle."""
    error_ids = detect_errors(text)
    hints = [ERROR_HINTS[eid] for eid in error_ids if eid in ERROR_HINTS]
    if not hints:
        return ""
    return "Voici des corrections ciblées à effectuer avec simple_patch (find→replace) ou apply_edit_b64 :\n- " + "\n- ".join(hints)


def run_harness():
    # S'assure que la CLI --help voit le package 'numops' sous src/
    env = dict(os.environ, PYTHONPATH=str(PROJECT_DIR / "src"))
    return run([sys.executable, "auto_harness.py"], env=env)

# ───────────────────────────────────────────────────────────────────────────────
# TOOLS (schémas pour le modèle)
# ───────────────────────────────────────────────────────────────────────────────
TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "Lister les .py du repo",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Lire un fichier du repo",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "simple_patch",
        "description": "Apply one find/replace",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "find": {"type": "string"}, 
                "replace": {"type": "string"}
            },
            "required": ["path", "find", "replace"]
        }
    }},
    {"type": "function", "function": {
        "name": "apply_edit_b64",
        "description": "Remplacer un fichier sous src/ avec contenu complet encodé en base64",
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string"},
                           "content_b64": {"type": "string"}
                       },
                       "required": ["path", "content_b64"]}
    }},
    {"type": "function", "function": {
        "name": "run_harness",
        "description": "Exécuter auto_harness.py",
        "parameters": {"type": "object", "properties": {}}
    }},
]

def _allowed_names_for_phase(phase: str) -> List[str]:
    if phase == "TEST":
        return ["run_harness"]
    if phase == "PATCH":
        # En PATCH on force l'action: uniquement patch; test autorisé après patch (phase TEST)
        return ["simple_patch", "apply_edit_b64"]
    # READ par défaut
    return ["read_file"]  # volontairement sans list_files pour éviter l'évitement

def _tools_by_name(names: List[str]):
    wanted = set(names)
    out = []
    for t in TOOLS:
        fn = t.get("function", {}).get("name")
        if fn in wanted:
            out.append(t)
    return out

# ───────────────────────────────────────────────────────────────────────────────
# FALLBACK : bloc EDIT texte si pas de tool call
# ───────────────────────────────────────────────────────────────────────────────
EDIT_BLOCK_RE = re.compile(
    r"<<<EDIT\s+PATH:\s*(?P<path>[^\r\n]+)\s+CONTENT:\s*(?P<content>.*?)\s+END_EDIT\s+>>>EDIT",
    re.DOTALL
)

def try_parse_and_apply_edit_blocks(assistant_text: str):
    if not assistant_text:
        return []
    changes = []
    for m in EDIT_BLOCK_RE.finditer(assistant_text):
        path = m.group("path").strip()
        content = m.group("content")
        res = _write_src_file(path, content)
        changes.append({"path": path, "result": res})
        print(f"\n◀── FALLBACK EDIT applied for {path}: {json.dumps(res, ensure_ascii=False)}")
    return changes

# ───────────────────────────────────────────────────────────────────────────────
# TOOL EXECUTOR (VALIDATION + ENVELOPPES + FEEDBACK)
# ───────────────────────────────────────────────────────────────────────────────
def call_tool(name, args, read_limiter: Optional[ReadLimiter] = None):
    """
    Exécute l'outil demandé avec validation stricte.
    Retourne toujours une enveloppe: {"ok": True/False, ...}
    """
    if VERBOSE:
        preview_args = json.dumps(args or {}, ensure_ascii=False)
        if len(preview_args) > 200:
            preview_args = preview_args[:200] + "…"
        print(f"\n───▶ TOOL CALL: {name} args={preview_args}...")

    args = args or {}
    try:
        # --- Outils métiers avec validation de schéma ---
        if name == "read_file":
            path = args.get("path")
            if not isinstance(path, str) or not path:
                return _err("read_file", "SCHEMA_ERROR",
                            "invalid arguments for read_file",
                            expected={"path": "<str>"}, args=args)
            if read_limiter:
                allowed, message = read_limiter.request_read(path)
                if not allowed:
                    if message:
                        print(f"[ReadLimiter] {message}")
                    expected = None
                    if read_limiter:
                        expected = {
                            "stage": read_limiter.stage,
                            "remaining_tokens": read_limiter.recovery_allowances.get(path, 0),
                        }
                    return _err(
                        "read_file",
                        "READ_LIMIT",
                        message or "lecture bloquée par le ReadLimiter",
                        expected=expected,
                        args={"path": path},
                    )
            res = read_file(path=path)
            return _ok(res)

        elif name == "simple_patch":
            path = args.get("path")
            find = args.get("find")
            replace = args.get("replace")
            if not isinstance(path, str) or not isinstance(find, str) or not isinstance(replace, str):
                return _err("simple_patch", "SCHEMA_ERROR",
                            "invalid arguments for simple_patch",
                            expected={"path":"<str>", "find":"<str>", "replace":"<str>"},
                            args=args)
            # Convertir en format edits pour apply_patch
            edits = [{"find": find, "replace": replace}]
            res = apply_patch(path=path, edits=edits)
            if read_limiter:
                read_limiter.record_patch_result(path, res)
            return _ok(res)

        elif name == "apply_edit_b64":
            path = args.get("path")
            content_b64 = args.get("content_b64")
            if not isinstance(path, str) or not isinstance(content_b64, str):
                return _err("apply_edit_b64", "SCHEMA_ERROR",
                            "invalid arguments for apply_edit_b64",
                            expected={"path":"<str>", "content_b64":"<base64>"},
                            args=args)
            res = apply_edit_b64(path=path, content_b64=content_b64)
            if "error" in res:
                return _err("apply_edit_b64", "DECODE_ERROR", res["error"], args={"path": path})
            return _ok(res)

        # --- Outils simples ---
        elif name == "list_files":
            return _ok(list_files())

        elif name == "run_harness":
            res = run_harness()
            return _ok(res)

        else:
            return _err(name, "UNKNOWN_TOOL", f"unknown tool '{name}'", args=args)

    except Exception as e:
        return _err(name, "TOOL_RUNTIME_ERROR", f"{e!r}", args=args)

def main():
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": """PHASE=TEST (initial) pour exécuter run_harness une fois, puis PHASE=READ → PATCH → TEST.
Appelle run_harness (JSON une ligne). Ensuite: READ ≤2 fichiers, puis COMMIT + PATCH_SKETCH + PATCH(JSON) + TEST(JSON)."""}
    ]

    state = AgentState()
    limiter = ReadLimiter()
    # Compteur de patchs au dernier test, pour empêcher run_harness prématuré en PATCH
    last_test_patch_count = 0
    phase = "TEST"  # TEST (init) -> READ -> PATCH -> TEST

    try:
        for turn in range(MAX_TURNS):
            limiter.start_turn(turn + 1)
            if VERBOSE:
                print(f"\n===== TURN {turn+1}/{MAX_TURNS} =====")

            content_buf = []
            start_global = time.time()
            last_chunk_time = start_global
            got_any_chunk = False
            # Sélection d'outils visibles selon la phase (réduction de choix = action)
            allowed_names = _allowed_names_for_phase(phase)
            allowed_tools = _tools_by_name(allowed_names)

            stream = ollama.chat(
                model=MODEL,
                messages=msgs,
                tools=allowed_tools,
                stream=True,
                options={"temperature": TEMPERATURE},
            )

            # Boucle de streaming des chunks
            for chunk in stream:
                now = time.time()

                if DEBUG_RAW_CHUNKS:
                    try:
                        print("\n[RAW CHUNK]", json.dumps(chunk, ensure_ascii=False, default=str)[:2000])
                    except Exception:
                        print("\n[RAW CHUNK] <non-serializable>")

                msg = chunk.get("message") or {}

                # 1) DONE
                if chunk.get("done"):
                    tool_calls = msg.get("tool_calls")
                    assistant_text = "".join(content_buf).strip()
                    if assistant_text:
                        print()
                    if VERBOSE:
                        dur = time.time() - start_global
                        print(f"─── time: {dur:.2f}s  (assistant streamed)")

                    if tool_calls:
                        # Exécuter chaque tool call
                        for tc in tool_calls:
                            # IMPORTANT: recalculer les outils autorisés à CHAQUE tool_call,
                            # car 'phase' a pu changer au tool_call précédent.
                            allowed_names = _allowed_names_for_phase(phase)
                            name = tc["function"]["name"]
                            args = tc["function"].get("arguments", {}) or {}

                            # 0) Filtrage dur par phase (bloque les outils non autorisés)
                            if name not in allowed_names:
                                print(f"◀── PHASE_BLOCK: '{name}' interdit en phase {phase}")
                                # Retourner une erreur outillée au modèle
                                block = _err(name, "PHASE_BLOCK",
                                             f"'{name}' non autorisé en phase {phase}",
                                             expected={"allowed_tools": allowed_names}, args=args)
                                msgs.append({"role": "tool", "name": name, "content": json.dumps(block, ensure_ascii=False)})
                                # Nudge ciblé vers l'action attendue
                                if phase == "PATCH":
                                    msgs.append({"role": "user", "content":
                                        "PHASE=PATCH. STOP_REPEATING lectures. "
                                        "Produis COMMIT + PATCH_SKETCH + PATCH(JSON) + TEST(JSON)."})
                                elif phase == "TEST":
                                    msgs.append({"role": "user", "content":
                                        "PHASE=TEST. Appelle run_harness (UN JSON une ligne)."})
                                else:
                                    msgs.append({"role": "user", "content":
                                        "PHASE=READ. Appelle read_file (≤2) puis passe en PATCH."})
                                # On n'exécute pas le tool non autorisé
                                continue

                            # 1) Exécuter l'outil (avec enveloppe ok/err)
                            if name == "run_harness" and phase == "PATCH" and state.patches_applied == last_test_patch_count:
                                print("◀── PHASE_BLOCK: 'run_harness' interdit en PATCH sans patch préalable")
                                block = _err(name, "PHASE_BLOCK",
                                             "run_harness non autorisé: applique un patch d'abord",
                                             expected={"allowed_tools": _allowed_names_for_phase("PATCH")},
                                             args=args)
                                msgs.append({"role": "tool", "name": name, "content": json.dumps(block, ensure_ascii=False)})
                                msgs.append({"role": "user", "content":
                                    "PHASE=PATCH. Produis COMMIT + PATCH_SKETCH + PATCH(JSON). "
                                    "Ensuite seulement, appelle run_harness."})
                                continue

                            result = call_tool(name, args, read_limiter=limiter)

                            # 2) Pousser le résultat outillé AU MODÈLE
                            msgs.append({"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)})

                            # 3) Logs humains compatibles enveloppe
                            if name == "run_harness":
                                payload = result.get("result", {})
                                stdout = payload.get("stdout") or ""
                                stderr = payload.get("stderr") or ""
                                print("◀── HARNESS stdout:\n" + stdout[:4000])
                                if stderr.strip():
                                    print("◀── HARNESS stderr:\n" + stderr[:4000])
                                print(f"◀── HARNESS returncode: {payload.get('returncode')}")
                                detected = detect_errors(stdout)
                                if detected:
                                    state.record_errors(detected)
                                suggestion = analyze_harness_output(stdout)
                                if suggestion:
                                    msgs.append({"role": "user", "content": suggestion})
                                # Après un test: prochaine phase = READ (nouvelle cible)
                                phase = "READ"
                                # Mémoriser l'état des patchs constaté à ce test
                                last_test_patch_count = state.patches_applied
                            else:
                                preview = json.dumps(result, ensure_ascii=False)
                                if len(preview) > 1200:
                                    preview = preview[:1200] + "…"
                                print(f"◀── TOOL RESULT: {name} -> {preview}")

                            if name == "read_file":
                                path = args.get("path")
                                if isinstance(path, str):
                                    state.files_read.add(path)
                                # Après une lecture réussie, on bascule en PATCH et on exige l'action
                                if result.get("ok"):
                                    phase = "PATCH"
                                    msgs.append({"role": "user", "content":
                                        "PHASE=PATCH. Produis maintenant: "
                                        "COMMIT, PATCH_SKETCH, puis un PATCH(JSON) et un TEST(JSON) sur une seule ligne chacun."})
                                else:
                                    # Si READ_LIMIT ou autre erreur: pousser vers PATCH
                                    err = result.get("error", {})
                                    if err and err.get("code") == "READ_LIMIT":
                                        phase = "PATCH"
                                        msgs.append({"role": "user", "content":
                                            "STOP_REPEATING. PHASE=PATCH. "
                                            "COMMIT + PATCH_SKETCH + PATCH(JSON) + TEST(JSON). Pas d'autres lectures."})

                            if name in {"simple_patch", "apply_edit_b64"} and result.get("ok"):
                                payload = result.get("result") or {}
                                if payload.get("applied"):
                                    state.patches_applied += 1
                                    # Après patch appliqué, demander immédiatement le test
                                    phase = "TEST"
                                    msgs.append({"role": "user", "content":
                                        "Patch appliqué. Appelle maintenant run_harness (UN JSON une ligne)."})
                                    # (last_test_patch_count < patches_applied) => run_harness autorisé au tour suivant

                            # 4) FEEDBACK anti-répétition
                            key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                            if not result.get("ok"):
                                _last_bad_calls[key] = _last_bad_calls.get(key, 0) + 1

                                # a) informer le modèle de l'erreur + schéma attendu
                                err = result["error"]
                                msgs.append({"role":"user","content":(
                                    f"Tool error: {err['tool']} ({err['code']}). "
                                    f"{err['message']}. Expected schema: {err.get('expected')}."
                                    " Corrige tes arguments et réessaie."
                                )})

                                # b) si répétition, le signaler explicitement
                                if _last_bad_calls[key] >= RETRY_PER_TOOL:
                                    msgs.append({"role":"user","content":(
                                        f"STOP_REPEATING: l'appel '{name}' avec ces arguments a déjà échoué "
                                        f"{_last_bad_calls[key]} fois. Reformule différemment ou choisis un autre outil."
                                    )})
                            else:
                                if key in _last_bad_calls:
                                    del _last_bad_calls[key]

                        # Laisser le modèle réagir au prochain tour
                        break

                    # Pas de tool → tenter fallback bloc EDIT
                    changes = try_parse_and_apply_edit_blocks(assistant_text)
                    if changes:
                        state.patches_applied += 1
                        msgs.append({"role": "assistant", "content": assistant_text})
                        res = run_harness()
                        print("◀── HARNESS (post-fallback) stdout:\n" + (res.get("stdout") or "")[:4000])
                        detected = detect_errors(res.get("stdout"))
                        if detected:
                            state.record_errors(detected)
                        msgs.append({"role": "tool", "name": "run_harness", "content": json.dumps(_ok(res))})
                        last_test_patch_count = state.patches_applied
                        break

                    # Sinon, pousser la réponse + nudge vers outils
                    if assistant_text:
                        msgs.append({"role": "assistant", "content": assistant_text})
                    # Nudge ciblé selon la phase
                    if phase == "READ":
                        msgs.append({"role": "user", "content":
                            "PHASE=READ. Appelle read_file (≤2) si nécessaire, puis passe en PATCH: "
                            "COMMIT + PATCH_SKETCH + PATCH(JSON) + TEST(JSON)."})
                    elif phase == "PATCH":
                        msgs.append({"role": "user", "content":
                            "PHASE=PATCH. STOP_REPEATING lectures. "
                            "Produis COMMIT + PATCH_SKETCH + PATCH(JSON) + TEST(JSON)."})
                    else:  # TEST
                        msgs.append({"role": "user", "content":
                            "PHASE=TEST. Appelle run_harness (UN JSON une ligne)."})
                    break

                # 2) Flux de texte normal (rare avec GPT-OSS)
                delta = msg.get("content", "")
                if delta:
                    got_any_chunk = True
                    last_chunk_time = now
                    content_buf.append(delta)
                    print(delta, end="", flush=True)

                # 3) Flux 'thinking' (silencieux par défaut)
                think = msg.get("thinking", None)
                if think:
                    got_any_chunk = True
                    last_chunk_time = now
                    if SHOW_THINK:
                        print(think, end="", flush=True)

                # 4) Timeouts
                if now - last_chunk_time > IDLE_TIMEOUT_SEC:
                    print(f"\n[AGENT] Timeout inactivité ({IDLE_TIMEOUT_SEC}s). Tour suivant.")
                    break
                if now - start_global > STREAM_TIMEOUT_SEC:
                    print(f"\n[AGENT] Timeout global ({STREAM_TIMEOUT_SEC}s). Tour suivant.")
                    break

            else:
                print("\n[WARN] stream ended without 'done'. Tour suivant.")

            if not got_any_chunk:
                print("[AGENT] Aucun chunk reçu de la part du modèle.")

        if VERBOSE:
            print("\n[END] Boucle terminée.")

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Arrêt demandé par l'utilisateur.")
    except Exception as e:
        print(f"\n[ERROR] Exception non gérée: {e}")

# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
