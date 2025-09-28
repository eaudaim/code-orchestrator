#!/usr/bin/env python3
# agent_fix.py — Agent local “réparateur” pour projet Python (GPT-OSS)
# - Streaming + logs + timeouts
# - Outils : list_files, read_file, apply_patch (find→replace), apply_edit_b64, run_harness
# - Enveloppes d’erreurs {"ok": true/false} pour TOUS les tools
# - Feedback d’erreur au modèle + anti-répétition d’appels invalides
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
from pathlib import Path
from typing import Tuple

import ollama

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────────────
MODEL = "gpt-oss:20b"
PROJECT_DIR = Path(".").resolve()
SRC_DIR = PROJECT_DIR / "src"

VERBOSE = True
SHOW_THINK = False       # coupe l’affichage verbeux du "thinking"
TEMPERATURE = 0.2
MAX_TURNS = 8

DEBUG_RAW_CHUNKS = False
STREAM_TIMEOUT_SEC = 120
IDLE_TIMEOUT_SEC = 30

SYSTEM = """Tu es un ingénieur logiciel.
Objectif : faire réussir 'python auto_harness.py' (exit=0) sans tricher.
Tu peux : lister/lire des fichiers, MODIFIER MINIMALEMENT des fichiers sous src/*.py, lancer l'harness.
Interdit : modifier auto_harness.py, tests/, requirements.txt, pyproject.toml. Ne masque pas les erreurs (ex: return True).

IMPORTANT — Ordre de préférence pour éditer :
1) apply_patch(path, edits=[{"find":"...", "replace":"..."}])  # remplacements ciblés JSON
2) apply_edit_b64(path, content_b64)                          # contenu ENTIER encodé Base64
3) (dernier recours) bloc texte <<<EDIT … END_EDIT >>>EDIT    # si les tools ne marchent pas

Respecte STRICTEMENT les schémas d’outils. Si un outil renvoie une erreur de schéma, corrige tes arguments et réessaie.

Procédure par tour :
- run_harness (si pertinent)
- read_file sur le(s) fichier(s) cassé(s)
- propose correction MINIMALE et appelle apply_patch, puis run_harness
Sois concis. Appelle un outil dès que possible. Évite le thinking verbeux.
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
    return {"files": [str(p.relative_to(PROJECT_DIR)) for p in PROJECT_DIR.rglob("*.py")]}

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

    if not isinstance(edits, list) or not edits:
        return {"error": "edits must be a non-empty list"}

    normalized = []
    for e in edits:
        if not isinstance(e, dict):
            return {"error": "edits must be a non-empty list"}
        find = e.get("find")
        replace = e.get("replace", "")
        if not isinstance(find, str) or not isinstance(replace, str):
            return {"error": "edits must be a non-empty list"}
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

def run_harness():
    # S’assure que la CLI --help voit src/ (PYTHONPATH)
    env = dict(os.environ, PYTHONPATH=str(PROJECT_DIR))
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
        "name": "apply_patch",
        "description": "Appliquer des remplacements ciblés (find→replace) sur un fichier sous src/",
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string"},
                           "edits": {
                               "type": "array",
                               "items": {
                                   "type": "object",
                                   "properties": {
                                       "find": {"type": "string"},
                                       "replace": {"type": "string"}
                                   },
                                   "required": ["find"]
                               }
                           }
                       },
                       "required": ["path", "edits"]}
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

        elif name == "apply_patch":
            path = args.get("path")
            edits = args.get("edits")
            if not isinstance(path, str) or not isinstance(edits, list):
                return _err("apply_patch", "SCHEMA_ERROR",
                            "invalid arguments for apply_patch",
                            expected={"path":"<str>",
                                      "edits":"[{find:<str>, replace:<str>}]"},
                            args=args)
            clean = []
            for e in edits:
                if not isinstance(e, dict):
                    continue
                f = e.get("find")
                r = e.get("replace", "")
                if isinstance(f, str) and isinstance(r, str):
                    clean.append({"find": f, "replace": r})
            if not clean:
                return _err("apply_patch", "SCHEMA_ERROR", "no valid edits", args=args)
            res = apply_patch(path=path, edits=clean)
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
def main():
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Commence par exécuter run_harness pour voir ce qui casse, puis corrige."}
    ]

    try:
        for turn in range(MAX_TURNS):
            if VERBOSE:
                print(f"\n===== TURN {turn+1}/{MAX_TURNS} =====")

            content_buf = []
            start_global = time.time()
            last_chunk_time = start_global
            got_any_chunk = False

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
                        # Exécuter chaque tool call
                        for tc in tool_calls:
                            name = tc["function"]["name"]
                            args = tc["function"].get("arguments", {}) or {}

                            # 1) Exécuter l’outil (avec enveloppe ok/err)
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
                            else:
                                preview = json.dumps(result, ensure_ascii=False)
                                if len(preview) > 1200:
                                    preview = preview[:1200] + "…"
                                print(f"◀── TOOL RESULT: {name} -> {preview}")

                            # 4) FEEDBACK anti-répétition
                            key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                            if not result.get("ok"):
                                _last_bad_calls[key] = _last_bad_calls.get(key, 0) + 1

                                # a) informer le modèle de l’erreur + schéma attendu
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
                        msgs.append({"role": "assistant", "content": assistant_text})
                        res = run_harness()
                        print("◀── HARNESS (post-fallback) stdout:\n" + (res.get("stdout") or "")[:4000])
                        msgs.append({"role": "tool", "name": "run_harness", "content": json.dumps(_ok(res))})
                        break

                    # Sinon, pousser la réponse + nudge vers outils
                    if assistant_text:
                        msgs.append({"role": "assistant", "content": assistant_text})
                    msgs.append({"role": "user",
                                 "content": "Aucun outil appelé. Appelle `apply_patch`, `apply_edit_b64`, `read_file` ou `run_harness` maintenant."})
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

