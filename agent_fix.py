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
from typing import List, Tuple

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
MAX_TURNS = 8

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
    turns_without_patch: int = 0
    read_only_turns: int = 0
    constraint_active: bool = False

    def record_errors(self, errors: List[str]):
        for err in errors:
            if err not in self.errors_identified:
                self.errors_identified.append(err)

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
def call_tool(name, args):
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

# ───────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE (streaming + timeouts + feedback)
# ───────────────────────────────────────────────────────────────────────────────
def _should_inject_constraint(state: AgentState) -> bool:
    if state.constraint_active:
        return False
    cond_errors_wait = bool(state.errors_identified and state.files_read and state.turns_without_patch >= 2)
    cond_read_loop = state.read_only_turns >= 4
    return cond_errors_wait or cond_read_loop


def _inject_constraint(state: AgentState, msgs):
    state.constraint_active = True
    errors = state.errors_identified or []
    if errors:
        errors_text = "\n".join(f"- {err}" for err in errors)
    else:
        errors_text = "- Corrige une des erreurs détectées précédemment en modifiant le code."
    constraint_message = (
        "STOP: tu relis les fichiers sans corriger.\n"
        "Interdiction de relancer read_file tant qu'un simple_patch réussi n'a pas été appliqué.\n"
        "Erreurs prioritaires à corriger :\n"
        f"{errors_text}\n"
        "Passe immédiatement à l'action avec simple_patch sur l'une de ces erreurs, sans relecture supplémentaire."
    )
    print("\n[GUARD] Injection de contrainte : forcer simple_patch.")
    msgs.append({"role": "user", "content": constraint_message})


def main():
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Commence par exécuter run_harness pour voir ce qui casse, puis corrige."}
    ]

    state = AgentState()

    try:
        for turn in range(MAX_TURNS):
            if VERBOSE:
                print(f"\n===== TURN {turn+1}/{MAX_TURNS} =====")

            content_buf = []
            start_global = time.time()
            last_chunk_time = start_global
            got_any_chunk = False
            turn_had_read = False
            turn_used_non_read_tool = False
            turn_applied_patch = False

            stream = ollama.chat(
                model=MODEL,
                messages=msgs,
                tools=TOOLS,
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
                        turn_used_non_read_tool = False
                        turn_had_read = False
                        # Exécuter chaque tool call
                        for tc in tool_calls:
                            name = tc["function"]["name"]
                            args = tc["function"].get("arguments", {}) or {}

                            # 1) Exécuter l'outil (avec enveloppe ok/err)
                            result = call_tool(name, args)

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
                            else:
                                preview = json.dumps(result, ensure_ascii=False)
                                if len(preview) > 1200:
                                    preview = preview[:1200] + "…"
                                print(f"◀── TOOL RESULT: {name} -> {preview}")

                            if name == "read_file":
                                turn_had_read = True
                                path = args.get("path")
                                if isinstance(path, str):
                                    state.files_read.add(path)
                            else:
                                turn_used_non_read_tool = True

                            if name == "simple_patch" and result.get("ok"):
                                payload = result.get("result") or {}
                                if payload.get("applied"):
                                    state.patches_applied += 1
                                    turn_applied_patch = True

                            if name == "apply_edit_b64" and result.get("ok"):
                                payload = result.get("result") or {}
                                if payload.get("applied"):
                                    turn_applied_patch = True
                                    turn_used_non_read_tool = True

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
                        turn_applied_patch = True
                        state.patches_applied += 1
                        turn_used_non_read_tool = True
                        msgs.append({"role": "assistant", "content": assistant_text})
                        res = run_harness()
                        print("◀── HARNESS (post-fallback) stdout:\n" + (res.get("stdout") or "")[:4000])
                        detected = detect_errors(res.get("stdout"))
                        if detected:
                            state.record_errors(detected)
                        msgs.append({"role": "tool", "name": "run_harness", "content": json.dumps(_ok(res))})
                        break

                    # Sinon, pousser la réponse + nudge vers outils
                    if assistant_text:
                        msgs.append({"role": "assistant", "content": assistant_text})
                    msgs.append({"role": "user",
                                 "content": "Aucun outil appelé. Appelle `simple_patch`, `apply_edit_b64`, `read_file` ou `run_harness` maintenant."})
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

            if turn_applied_patch:
                state.turns_without_patch = 0
                state.read_only_turns = 0
                state.constraint_active = False
            else:
                state.turns_without_patch += 1
                if turn_had_read and not turn_used_non_read_tool:
                    state.read_only_turns += 1
                else:
                    state.read_only_turns = 0

            if _should_inject_constraint(state):
                _inject_constraint(state, msgs)

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
