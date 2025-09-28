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
    per_file_failures = {}
    for p in SRC.rglob("*.py"):
        mod_path = p.relative_to(PROJECT).with_suffix("")
        mod_name = ".".join(mod_path.parts)
        try:
            m = importlib.import_module(mod_name)
            tried += 1
            f, _ = doctest.testmod(m, verbose=False, optionflags=doctest.ELLIPSIS)
            failures += f
            if f:
                key = p.name
                per_file_failures[key] = per_file_failures.get(key, 0) + f
        except Exception as e:
            failures += 1
            print(f"[DOCTEST ERROR] {mod_name} -> {e}")
            key = p.name
            per_file_failures[key] = per_file_failures.get(key, 0) + 1
    msg = f"[DOCTEST] tried={tried} failures={failures}"
    if per_file_failures:
        breakdown = ", ".join(f"{name}: {count}" for name, count in sorted(per_file_failures.items()))
        msg += f" | breakdown: {breakdown}"
    print(msg)
    return {
        "status": "PASS" if failures == 0 else "FAIL",
        "detail": msg,
        "failures": per_file_failures,
    }

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
    per_file_errors = {}
    if rc["returncode"] != 0:
        for raw_line in out.splitlines():
            line = raw_line.strip()
            if "error:" not in line:
                continue
            if line.count(":") < 2:
                continue
            m = re.match(r"(?P<path>.+?):(\d+):\s*error:", line)
            if not m:
                continue
            path_text = m.group("path").strip()
            if not path_text:
                continue
            candidate = pathlib.Path(path_text)
            if not candidate.exists():
                possible = (PROJECT / path_text).resolve()
                if possible.exists():
                    candidate = possible
            key = candidate.name if candidate.exists() else pathlib.Path(path_text).name
            per_file_errors[key] = per_file_errors.get(key, 0) + 1
    return {
        "status": "PASS" if rc["returncode"] == 0 else "FAIL",
        "detail": out,
        "errors": per_file_errors,
    }


def _parse_breakdown(detail: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not detail:
        return counts
    m = re.search(r"breakdown:\s*(.+)", detail)
    if not m:
        return counts
    tail = m.group(1)
    for chunk in tail.split(","):
        name, _, count = chunk.partition(":")
        name = name.strip()
        count = count.strip()
        if not name or not count:
            continue
        try:
            counts[name] = counts.get(name, 0) + int(count)
        except ValueError:
            continue
    return counts


def count_errors(results):
    doctest_counts: Dict[str, int] = {}
    mypy_counts: Dict[str, int] = {}
    doctest_total = 0
    mypy_total = 0

    for name, res in results:
        if name == "doctest":
            breakdown = {}
            if isinstance(res, dict):
                breakdown = res.get("failures") or {}
                if not breakdown:
                    breakdown = _parse_breakdown(res.get("detail", ""))
            for fname, qty in breakdown.items():
                doctest_counts[fname] = doctest_counts.get(fname, 0) + qty
                doctest_total += qty
        elif name == "mypy":
            breakdown = {}
            if isinstance(res, dict):
                breakdown = res.get("errors") or {}
                if not breakdown:
                    breakdown = _parse_breakdown(res.get("detail", ""))
            for fname, qty in breakdown.items():
                mypy_counts[fname] = mypy_counts.get(fname, 0) + qty
                mypy_total += qty

    return {
        "doctest": {"total": doctest_total, "per_file": doctest_counts},
        "mypy": {"total": mypy_total, "per_file": mypy_counts},
        "total": doctest_total + mypy_total,
    }

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

    counts = count_errors(results)
    print("\n" + "=" * 70)
    print("DÉCOMPTE DES ERREURS")
    print("=" * 70)

    doctest_total = counts["doctest"]["total"]
    doctest_label = "erreur" if doctest_total == 1 else "erreurs"
    doctest_breakdown = counts["doctest"]["per_file"]
    if doctest_breakdown:
        parts = ", ".join(f"{name}: {count}" for name, count in sorted(doctest_breakdown.items()))
        print(f"Doctests: {doctest_total} {doctest_label} ({parts})")
    else:
        print(f"Doctests: {doctest_total} {doctest_label}")

    mypy_total = counts["mypy"]["total"]
    mypy_label = "erreur" if mypy_total == 1 else "erreurs"
    mypy_breakdown = counts["mypy"]["per_file"]
    if mypy_breakdown:
        parts = ", ".join(f"{name}: {count}" for name, count in sorted(mypy_breakdown.items()))
        print(f"Mypy: {mypy_total} {mypy_label} ({parts})")
    else:
        print(f"Mypy: {mypy_total} {mypy_label}")

    total_errors = counts["total"]
    total_label = "erreur détectée" if total_errors == 1 else "erreurs détectées"
    print(f"TOTAL: {total_errors} {total_label}")

    # Code de sortie : 0 si tout PASS ou SKIPPED ; 1 si au moins un FAIL
    sys.exit(1 if any_fail else 0)

if __name__ == "__main__":
    main()

