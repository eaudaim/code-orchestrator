#!/usr/bin/env python3
# auto_harness.py — oracle automatique AGRÉGÉ
# Exécute TOUT (doctests, smoke, CLI, Ruff, Mypy, Pytest si présent),
# affiche un résumé et sort non-zéro s'il y a au moins un FAIL.
# Les outils non installés sont marqués SKIPPED (sans stopper les autres étapes).

import sys
import subprocess
import re
import importlib
import inspect
import pathlib
from typing import get_origin, get_args, Optional, Union, List, Dict

PROJECT = pathlib.Path(".").resolve()
SRC = PROJECT / "src"

# ───────────────────────── Utils ─────────────────────────
def run(cmd, timeout=180):
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

def ensure_tool(modname: str, friendly: str | None = None) -> bool:
    """
    Vérifie si `python -m <modname> --version` marche.
    Ne tente PAS d'installation réseau (pour éviter blocages hors-ligne).
    """
    probe = run([sys.executable, "-m", modname, "--version"])
    if probe["returncode"] == 0:
        return True
    print(f"[INFO] {friendly or modname} non disponible → SKIPPED")
    return False

def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ───────────────────────── Pytest (optionnel) ─────────────────────────
def pytest_available() -> bool:
    try:
        import pytest  # noqa: F401
        return True
    except Exception:
        return False

def run_pytest_if_any():
    tests_dir = PROJECT / "tests"
    if not tests_dir.exists():
        return {"status": "SKIPPED", "detail": "no tests/ directory"}
    has_tests = any(tests_dir.rglob("test_*.py"))
    if not has_tests:
        return {"status": "SKIPPED", "detail": "no test_*.py files"}
    if not pytest_available():
        return {"status": "SKIPPED", "detail": "pytest not installed"}
    rc = run([sys.executable, "-m", "pytest", "-q"])
    out = rc["stdout"] or rc["stderr"]
    print(out)
    return {"status": "PASS" if rc["returncode"] == 0 else "FAIL", "detail": out}

# ───────────────────────── Doctests ─────────────────────────
def run_doctests():
    import doctest
    failures = 0; tried = 0
    sys.path.insert(0, str(PROJECT))
    for p in SRC.rglob("*.py"):
        mod_path = p.relative_to(PROJECT).with_suffix("")
        mod_name = ".".join(mod_path.parts)
        try:
            m = importlib.import_module(mod_name)
            tried += 1
            f, _ = doctest.testmod(m, verbose=False, optionflags=doctest.ELLIPSIS)
            failures += f
        except Exception as e:
            failures += 1
            print(f"[DOCTEST ERROR] {mod_name} -> {e}")
    msg = f"[DOCTEST] tried={tried} failures={failures}"
    print(msg)
    return {"status": "PASS" if failures == 0 else "FAIL", "detail": msg}

# ───────────────────────── CLI --help (si détectable) ─────────────────────────
def detect_top_package():
    if not SRC.exists():
        return None
    pkgs = [d.name for d in SRC.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
    if len(pkgs) == 1:
        return pkgs[0]
    mains = list(SRC.rglob("__main__.py"))
    if mains:
        return mains[0].parent.name
    return None

def run_cli_help():
    pkg = detect_top_package()
    if not pkg:
        return {"status": "SKIPPED", "detail": "no package/__main__ detected"}
    rc = run([sys.executable, "-m", pkg, "--help"])
    out = rc["stdout"] or rc["stderr"]
    if rc["returncode"] == 0 and re.search(r"(usage|help)", out, flags=re.IGNORECASE):
        print(f"[CLI] {pkg} --help OK")
        return {"status": "PASS", "detail": out}
    print(f"[CLI] {pkg} --help FAIL\n{out}")
    return {"status": "FAIL", "detail": out}

# ───────────────────────── Smoke des fonctions ─────────────────────────
def value_for_hint(h):
    origin = get_origin(h)
    if h in (int,): return 0
    if h in (float,): return 0.0
    if h in (str,): return ""
    if h in (bool,): return False
    if origin in (list, List): return []
    if origin in (dict, Dict): return {}
    if origin in (Union, Optional):
        for a in get_args(h):
            if a is not type(None):
                return value_for_hint(a)
        return None
    return None

def smoke_functions():
    sys.path.insert(0, str(PROJECT))
    failures = 0; tested = 0
    for p in SRC.rglob("*.py"):
        mod_path = p.relative_to(PROJECT).with_suffix("")
        mod_name = ".".join(mod_path.parts)
        try:
            m = importlib.import_module(mod_name)
        except Exception as e:
            print(f"[IMPORT FAIL] {mod_name}: {e}")
            failures += 1; continue
        for name, fn in inspect.getmembers(m, inspect.isfunction):
            if fn.__module__ != m.__name__: continue
            sig = inspect.signature(fn)
            if any(param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD) for param in sig.parameters.values()):
                continue
            kwargs = {}
            ok = True
            for pname, param in sig.parameters.items():
                if param.default is not inspect._empty:
                    kwargs[pname] = param.default; continue
                v = value_for_hint(param.annotation)
                if v is None: ok = False; break
                kwargs[pname] = v
            if not ok: continue
            try:
                fn(**kwargs); tested += 1
            except Exception as e:
                failures += 1
                print(f"[SMOKE FAIL] {mod_name}.{name}({kwargs}) -> {e}")
    msg = f"[SMOKE] tested={tested} failures={failures}"
    print(msg)
    return {"status": "PASS" if failures == 0 else "FAIL", "detail": msg}

# ───────────────────────── Ruff & Mypy ─────────────────────────
def run_ruff_safe():
    """
    Règles "safe only" (erreurs sûres, pas de pinaillage de style) :
    E9 (erreurs fatales), F63/F7/F82 (undefined names / misuse)
    """
    if not ensure_tool("ruff", "Ruff"):
        return {"status": "SKIPPED", "detail": "ruff not installed"}
    rc = run([sys.executable, "-m", "ruff", "check", "src", "--select=E9,F63,F7,F82"])
    out = rc["stdout"] or rc["stderr"]
    print(out)
    return {"status": "PASS" if rc["returncode"] == 0 else "FAIL", "detail": out}

def run_mypy_tolerant():
    """
    Mypy tolérant (ignore libs non typées) mais remonte les erreurs de tes modules.
    """
    if not ensure_tool("mypy", "Mypy"):
        return {"status": "SKIPPED", "detail": "mypy not installed"}
    rc = run([
        sys.executable, "-m", "mypy", "src",
        "--ignore-missing-imports",
        "--show-error-codes",
        "--pretty",
    ])
    out = rc["stdout"] or rc["stderr"]
    print(out)
    return {"status": "PASS" if rc["returncode"] == 0 else "FAIL", "detail": out}

# ───────────────────────── Main (agrégé) ─────────────────────────
def main():
    results = []

    print_section("1) Pytest (si présent)")
    results.append(("pytest", run_pytest_if_any()))

    print_section("2) Doctests")
    results.append(("doctest", run_doctests()))

    print_section("3) CLI --help")
    results.append(("cli_help", run_cli_help()))

    print_section("4) Smoke des fonctions")
    results.append(("smoke", smoke_functions()))

    print_section("5) Ruff (safe errors)")
    results.append(("ruff", run_ruff_safe()))

    print_section("6) Mypy (tolérant)")
    results.append(("mypy", run_mypy_tolerant()))

    # Résumé
    print_section("RÉSUMÉ")
    any_fail = False
    for name, res in results:
        status = res["status"]
        if status == "FAIL":
            any_fail = True
        print(f"{name:10s} : {status}")

    # Code de sortie : 0 si tout PASS ou SKIPPED ; 1 si au moins un FAIL
    sys.exit(1 if any_fail else 0)

if __name__ == "__main__":
    main()

