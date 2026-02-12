import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from utils.file_utils import read_file

try:
    import config.settings as cfg

    VERBOSE = cfg.VERBOSE
    EXEC_TIMEOUT = cfg.EXEC_TIMEOUT
    MAX_RETRIES = cfg.MAX_RETRIES
    AUTONOMY = cfg.AUTONOMY
    SCRIPT_NAME = cfg.SCRIPT_NAME if hasattr(cfg, "SCRIPT_NAME") else Path(__file__).name
except Exception:
    VERBOSE = False
    EXEC_TIMEOUT = 30
    MAX_RETRIES = 3
    AUTONOMY = True
    SCRIPT_NAME = Path(__file__).name

console = None
log_verbose = lambda message: None  # noqa: E731
WORK_DIR = Path.cwd()


def _resolve_within_workdir(user_path: str) -> Path:
    candidate = (WORK_DIR / user_path).resolve()
    root = WORK_DIR.resolve()

    is_within = False
    if hasattr(candidate, "is_relative_to"):
        is_within = candidate.is_relative_to(root)
    else:
        try:
            candidate.relative_to(root)
            is_within = True
        except ValueError:
            is_within = False

    if not is_within:
        raise ValueError(
            f"[ERROR] Path '{user_path}' is outside the project directory (WORK_DIR)."
        )

    return candidate


def set_executor_environment(
    console_obj,
    logger,
    verbose: bool,
    exec_timeout: int,
    max_retries: int,
    autonomy_flag: bool,
    script_name: str,
    work_dir: Path,
) -> None:
    global console, log_verbose, VERBOSE, EXEC_TIMEOUT, MAX_RETRIES, AUTONOMY, SCRIPT_NAME, WORK_DIR

    console = console_obj
    log_verbose = logger
    VERBOSE = verbose
    EXEC_TIMEOUT = exec_timeout
    MAX_RETRIES = max_retries
    AUTONOMY = autonomy_flag
    SCRIPT_NAME = script_name
    WORK_DIR = work_dir


def detect_dangerous_patterns(code: str) -> List[str]:
    warnings = []
    patterns = {
        r"os\.chdir\s*\(": "Changement de répertoire (os.chdir)",
        r"shutil\.rmtree": "Suppression récursive de répertoire (shutil.rmtree)",
        r"subprocess\.(call|run|Popen)": "Exécution de commandes système (subprocess)",
        r"\.\.\/": "Traversée de répertoire (../)",
        r"os\.remove|os\.unlink": "Suppression de fichier (os.remove/unlink)",
        r"open\s*\([^)]*[\"\']w[\"\']": "Écriture dans un fichier (open mode 'w')",
        r"eval\s*\(|exec\s*\(": "Exécution de code dynamique (eval/exec)",
        r"__import__": "Import dynamique (__import__)",
        r"socket\.|urllib\.|requests\.": "Accès réseau",
    }

    for pattern, description in patterns.items():
        if re.search(pattern, code):
            warnings.append(f"⚠️  {description}")

    return warnings


def read_file_tool(file_path: str) -> str:
    try:
        full_path = _resolve_within_workdir(file_path)
    except ValueError as error:
        return str(error)

    if full_path == Path(__file__).resolve() or full_path.name == SCRIPT_NAME:
        return "[ERROR] Reading the orchestrator script is not allowed."

    if not full_path.exists() or not full_path.is_file():
        return f"[ERROR] File '{file_path}' not found."
    return read_file(full_path)


def list_files_tool(directory_path: str = ".") -> str:
    try:
        target_dir = _resolve_within_workdir(directory_path)
        log_verbose(f"list_files_tool: {target_dir}")

        if not target_dir.exists():
            return f"[ERROR] Directory does not exist: {directory_path}"

        if not target_dir.is_dir():
            return f"[ERROR] Not a directory: {directory_path}"

        excluded_patterns = {
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
            ".DS_Store",
        }

        items = []

        for item in sorted(target_dir.iterdir()):
            if any(pattern in str(item.name) for pattern in excluded_patterns):
                continue

            if item.is_file() and (
                item.name == SCRIPT_NAME or item.resolve() == Path(__file__).resolve()
            ):
                log_verbose(f"   ⏭️  Exclusion : {item.name}")
                continue

            relative_path = item.relative_to(WORK_DIR)

            if item.is_dir():
                try:
                    file_count = len([f for f in item.iterdir() if f.is_file()])
                    items.append(f"📁 {relative_path}/ ({file_count} fichiers)")
                except PermissionError:
                    items.append(f"📁 {relative_path}/ (accès refusé)")
            else:
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/(1024*1024):.1f}MB"

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
            return (
                f"[INFO] Directory '{directory_path}' is empty (or contains only excluded files/folders)"
            )

        header = f"📂 Contents of '{directory_path}':"
        return header + "\n" + "\n".join(f"  {item}" for item in items)

    except Exception as e:
        return f"[ERROR] Failed to list directory '{directory_path}': {e}"


def write_file_tool(
    file_path: str,
    content: str,
    line_start: int,
    line_end: int,
) -> Tuple[bool, str]:
    root = WORK_DIR
    full_path = WORK_DIR / file_path

    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(root)):
            return False, f"[ERROR] Path '{file_path}' is outside the project directory."
    except Exception as e:
        return False, f"[ERROR] Invalid path: {e}"

    if content is None or content.strip() == "":
        return (
            False,
            "[ERROR] Empty 'content' passed to write_file_tool; refusing to overwrite the file.\n"
            "You must provide non-empty content.",
        )

    existing_text = ""
    if full_path.exists():
        try:
            existing_text = full_path.read_text(encoding="utf-8")
        except Exception as e:
            return False, f"[ERROR] Failed to read existing file before patching: {e}"

    lines = existing_text.splitlines()

    if line_end == -1:
        line_end = len(lines)

    if line_start < 1:
        line_start = 1

    start_idx = max(line_start - 1, 0)
    end_idx = min(line_end, len(lines))

    new_block_lines = content.splitlines()

    if not lines:
        new_lines = new_block_lines
        mode_desc = "création fichier"
    else:
        if line_start > line_end:
            new_lines = lines[:start_idx] + new_block_lines + lines[start_idx:]
            mode_desc = f"insertion ligne {line_start}"
        else:
            new_lines = lines[:start_idx] + new_block_lines + lines[end_idx:]
            if line_start == 1 and line_end >= len(lines):
                mode_desc = "remplacement complet"
            else:
                mode_desc = f"remplacement lignes {line_start}-{line_end}"

    final_content = "\n".join(new_lines)

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

    if AUTONOMY:
        console.print("[dim]🤖 Mode autonomie: modification appliquée sans confirmation utilisateur.[/dim]")
        confirmation = "o"
    else:
        confirmation = input("Autoriser cette modification ? (o/n) : ")

    if confirmation.lower() not in ["o", "oui", "y", "yes"]:
        return False, "[CANCELLED] User cancelled the file modification."

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with full_path.open("w", encoding="utf-8") as f:
            f.write(final_content)
        console.print(f"[green]✅ Fichier '{file_path}' modifié avec succès ![/green]")
        return (
            True,
            f"[SUCCESS] File '{file_path}' written successfully "
            f"({len(final_content)} bytes, {mode_desc}).",
        )
    except Exception as e:
        return False, f"[ERROR] Failed to write file: {e}"


def execute_code_tool(file_path: str) -> str:
    root = WORK_DIR
    full_path = root / file_path

    if not full_path.exists() or not full_path.is_file():
        return f"[ERROR] File '{file_path}' not found."

    if not file_path.endswith('.py'):
        return f"[ERROR] Only Python (.py) files can be executed. Got: {file_path}"

    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(root)):
            return f"[ERROR] Path '{file_path}' is outside the project directory."
    except Exception as e:
        return f"[ERROR] Invalid path: {e}"

    try:
        code_content = full_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"[ERROR] Failed to read file: {e}"

    warnings = detect_dangerous_patterns(code_content)

    console.print(f"\n[cyan]🚀 Le modèle veut exécuter : [bold]{file_path}[/bold][/cyan]")
    console.print(f"[dim]Timeout : {EXEC_TIMEOUT}s[/dim]")

    if warnings:
        console.print("\n[yellow]⚠️  Avertissements de sécurité :[/yellow]")
        for warning in warnings:
            console.print(f"[dim]   {warning}[/dim]")

    preview = code_content[:300]
    if len(code_content) > 300:
        preview += f"\n... ({len(code_content) - 300} caractères supplémentaires)"

    console.print("\n[dim]Aperçu du code :[/dim]")
    console.print("[dim]" + "─" * 60 + "[/dim]")
    console.print(preview)
    console.print("[dim]" + "─" * 60 + "[/dim]")

    if AUTONOMY:
        console.print("[dim]🤖 Mode autonomie: exécution autorisée sans confirmation utilisateur.[/dim]")
        confirmation = "o"
    else:
        confirmation = input("Autoriser l'exécution ? (o/n) : ")

    if confirmation.lower() not in ['o', 'oui', 'y', 'yes']:
        return "[CANCELLED] User cancelled the execution."

    console.print(f"[dim]⏳ Exécution en cours (timeout: {EXEC_TIMEOUT}s)...[/dim]")
    log_verbose(f"Exécution de {file_path} avec timeout={EXEC_TIMEOUT}s")

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
            cwd=str(root),
            timeout=EXEC_TIMEOUT,
            capture_output=True,
            text=True,
            env={
                "PYTHONPATH": str(root),
                "PATH": "/usr/bin:/bin",
                **({"VIRTUAL_ENV": str(venv_dir)} if using_venv else {}),
            }
        )
        stdout = result.stdout[:10000] if result.stdout else ""
        stderr = result.stderr[:10000] if result.stderr else ""
        if result.returncode == 0:
            console.print("[green]✅ Exécution réussie ![/green]")
        else:
            console.print(f"[yellow]⚠️  Exécution terminée avec code {result.returncode}[/yellow]")

        if stdout:
            console.print("\n[dim]📤 Sortie standard (stdout) :[/dim]")
            console.print("[dim]" + "─" * 60 + "[/dim]")
            console.print(stdout[:500])
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
    root = WORK_DIR
    venv_dir = root / venv_path
    try:
        venv_dir = venv_dir.resolve()
        if not str(venv_dir).startswith(str(root)):
            return f"[ERROR] Venv path '{venv_path}' is outside the project directory."
    except Exception as e:
        return f"[ERROR] Invalid venv path: {e}"
    console.print(f"\n[cyan]🐍 Le modèle veut créer un environnement virtuel : [bold]{venv_path}[/bold][/cyan]")
    console.print("[dim]Commande exécutée : python3 -m venv <venv_path>[/dim]")
    if venv_dir.exists():
        console.print(f"[yellow]⚠️  L'environnement virtuel existe déjà : {venv_dir}[/yellow]")
        return f"[VENV] Virtual environment already exists at: {venv_dir}"
    if AUTONOMY:
        confirmation = "o"
    else:
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
            console.print(f"[yellow]⚠️  Création terminée avec code {result.returncode}")
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
        error_msg = f"[ERROR] Failed to create virtual environment: {e}"
        console.print(f"[red]❌ {error_msg}[/red]")
        log_verbose(f"Exception lors de la création du venv : {e}")
        return error_msg


def _run_git_command(args: List[str], root: Path):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            timeout=EXEC_TIMEOUT,
            capture_output=True,
            text=True,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                **({"HOME": os.environ["HOME"]} if "HOME" in os.environ else {}),
            },
        )
    except subprocess.TimeoutExpired:
        return (
            None,
            "",
            "",
            f"[ERROR] Git command 'git {' '.join(args)}' timed out after {EXEC_TIMEOUT}s.",
        )
    except FileNotFoundError:
        return None, "", "", "[ERROR] Git is not installed on the system."
    except Exception as e:
        return None, "", "", f"[ERROR] Failed to run git command: {e}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    return result, stdout, stderr, ""


def _git_not_initialized_message() -> str:
    return "[ERROR] Git repository is not initialized. Run git_init first."


def _git_get_config_value(key: str, root: Path, scope: str = "local"):
    args = ["config"]

    if scope == "local":
        args.append("--local")
    elif scope == "global":
        args.append("--global")

    args.extend(["--get", key])

    result, stdout, stderr, error_msg = _run_git_command(args, root)

    if error_msg:
        return None, error_msg

    if not result:
        return None, f"[ERROR] Unable to run git config for {key}."

    if result.returncode == 0:
        return stdout.strip(), None

    if result.returncode == 1:
        return None, None

    combined_output = "\n".join(filter(None, [stdout, stderr]))
    return (
        None,
        f"[ERROR] git config failed for {key} with code {result.returncode}."
        + (f"\n{combined_output}" if combined_output else ""),
    )


def _git_set_config_value(key: str, value: str, root: Path):
    result, stdout, stderr, error_msg = _run_git_command(["config", key, value], root)

    if error_msg:
        return False, error_msg

    if not result or result.returncode != 0:
        combined_output = "\n".join(filter(None, [stdout, stderr]))
        return (
            False,
            f"[ERROR] Unable to set git config {key} (code {result.returncode if result else 'unknown'})."
            + (f"\n{combined_output}" if combined_output else ""),
        )

    return True, ""


def git_ensure_identity(root: Path) -> Tuple[bool, str]:
    git_dir = root / ".git"

    if not git_dir.exists():
        return False, _git_not_initialized_message()

    local_name, local_name_err = _git_get_config_value("user.name", root, scope="local")
    if local_name_err:
        return False, local_name_err

    local_email, local_email_err = _git_get_config_value("user.email", root, scope="local")
    if local_email_err:
        return False, local_email_err

    global_name, global_name_err = _git_get_config_value("user.name", root, scope="global")
    if global_name_err:
        return False, global_name_err

    global_email, global_email_err = _git_get_config_value("user.email", root, scope="global")
    if global_email_err:
        return False, global_email_err

    effective_name = local_name or global_name
    effective_email = local_email or global_email

    if effective_name and effective_email:
        name_source = "local" if local_name else "global"
        email_source = "local" if local_email else "global"

        if name_source == email_source == "local":
            message = f"[GIT] Using local git identity: {effective_name} <{effective_email}>."
        elif name_source == email_source == "global":
            message = f"[GIT] Using global git identity: {effective_name} <{effective_email}>."
        else:
            message = (
                "[GIT] Using git identity from configuration: "
                f"name={effective_name} ({name_source}), email={effective_email} ({email_source})."
            )

        return True, message

    temp_name = "AI Code Assistant"
    temp_email = "ai@example.com"

    name_ok, name_msg = _git_set_config_value("user.name", temp_name, root)
    if not name_ok:
        return False, name_msg

    email_ok, email_msg = _git_set_config_value("user.email", temp_email, root)
    if not email_ok:
        return False, email_msg

    return True, f"[GIT] Temporary local git identity configured: {temp_name} <{temp_email}>."


def _git_working_tree_status(root: Path):
    status_result, status_stdout, status_stderr, error_msg = _run_git_command(
        ["status", "--porcelain"],
        root,
    )

    if error_msg:
        return None, error_msg

    combined = "\n".join(filter(None, [status_stdout, status_stderr]))

    if not status_result:
        return None, f"[ERROR] Failed to query git status.\n{combined}" if combined else "[ERROR] Failed to query git status."

    if status_result.returncode != 0:
        return None, f"[ERROR] Git status failed with code {status_result.returncode}.\n{combined}" if combined else (
            f"[ERROR] Git status failed with code {status_result.returncode}."
        )

    return status_result, combined


def git_init_tool() -> str:
    root = WORK_DIR
    git_dir = root / ".git"

    if git_dir.exists():
        return "[GIT] Repository already initialized."

    result, stdout, stderr, error_msg = _run_git_command(["init"], root)

    if error_msg:
        return error_msg

    output = "\n".join(filter(None, [stdout, stderr]))

    if result and result.returncode == 0:
        gitignore_path = root / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(f"{SCRIPT_NAME}\n")

        return "[GIT] Repository initialized successfully." + (f"\n{output}" if output else "")

    return (
        f"[ERROR] Git init failed with return code {result.returncode if result else 'unknown'}."
        + (f"\n{output}" if output else "")
    )


def git_commit_tool(message: str) -> str:
    if not isinstance(message, str) or not message.strip():
        return "[ERROR] Commit message must be a non-empty string."

    root = WORK_DIR
    git_dir = root / ".git"

    if not git_dir.exists():
        return _git_not_initialized_message()

    identity_ok, identity_message = git_ensure_identity(root)

    if not identity_ok:
        return identity_message

    status_result, status_output = _git_working_tree_status(root)
    if not status_result:
        return status_output

    if not status_output.strip():
        return (
            (identity_message + "\n") if identity_message else ""
            + "[GIT] Nothing to commit: working tree clean. Use write_file before committing."
        )

    add_result, add_stdout, add_stderr, error_msg = _run_git_command(["add", "-A"], root)

    if error_msg:
        return error_msg

    if add_result and add_result.returncode != 0:
        output = "\n".join(filter(None, [add_stdout, add_stderr]))
        return f"[ERROR] git add failed with return code {add_result.returncode}." + (f"\n{output}" if output else "")

    commit_args = ["commit", "-m", message]
    commit_result, commit_stdout, commit_stderr, error_msg = _run_git_command(commit_args, root)

    if error_msg:
        return error_msg

    combined_output = "\n".join(filter(None, [commit_stdout, commit_stderr]))
    commit_text = (commit_stdout + "\n" + commit_stderr).lower()

    if commit_result and commit_result.returncode == 0:
        prefix = (identity_message + "\n") if identity_message else ""
        return prefix + "[GIT] Commit created successfully." + (f"\n{combined_output}" if combined_output else "")

    if "nothing to commit" in commit_text:
        prefix = (identity_message + "\n") if identity_message else ""
        return prefix + "[GIT] Nothing to commit: working tree clean."

    return (
        f"[ERROR] Git commit failed with return code {commit_result.returncode if commit_result else 'unknown'}."
        + (f"\n{combined_output}" if combined_output else "")
    )


def git_rollback_tool(steps: int = 1) -> str:
    try:
        steps_value = int(steps)
    except (TypeError, ValueError):
        return f"[ERROR] Invalid steps value: {steps}. It must be a positive integer."

    if steps_value <= 0:
        return f"[ERROR] Steps must be greater than 0. Got: {steps_value}."

    root = WORK_DIR
    git_dir = root / ".git"

    if not git_dir.exists():
        return _git_not_initialized_message()

    count_result, count_stdout, count_stderr, error_msg = _run_git_command(
        ["rev-list", "--count", "HEAD"],
        root,
    )

    if error_msg:
        return error_msg

    if not count_result or count_result.returncode != 0:
        output = "\n".join(filter(None, [count_stdout, count_stderr]))
        return "[ERROR] Unable to determine commit history." + (f"\n{output}" if output else "")

    try:
        commit_count = int((count_stdout or "0").strip())
    except ValueError:
        commit_count = 0

    if commit_count == 0:
        return "[GIT] No commits available to roll back."

    if steps_value >= commit_count:
        return (
            f"[GIT] Not enough history to rollback {steps_value} step(s). "
            f"Available commits: {commit_count}."
        )

    reset_result, reset_stdout, reset_stderr, error_msg = _run_git_command(
        ["reset", "--hard", f"HEAD~{steps_value}"],
        root,
    )

    if error_msg:
        return error_msg

    output = "\n".join(filter(None, [reset_stdout, reset_stderr]))

    if not reset_result or reset_result.returncode != 0:
        return (
            f"[ERROR] Git rollback failed with return code {reset_result.returncode if reset_result else 'unknown'}."
            + (f"\n{output}" if output else "")
        )

    return (
        f"[GIT] Rolled back {steps_value} commit(s) successfully." + (f"\n{output}" if output else "")
    )


def git_history_tool() -> str:
    root = WORK_DIR
    git_dir = root / ".git"

    if not git_dir.exists():
        return _git_not_initialized_message()

    count_result, count_stdout, count_stderr, error_msg = _run_git_command(
        ["rev-list", "--count", "HEAD"],
        root,
    )

    if error_msg:
        return error_msg

    if not count_result or count_result.returncode != 0:
        output = "\n".join(filter(None, [count_stdout, count_stderr]))
        return "[ERROR] Unable to retrieve git history." + (f"\n{output}" if output else "")

    try:
        commit_count = int((count_stdout or "0").strip())
    except ValueError:
        commit_count = 0

    if commit_count == 0:
        return "[GIT] No git history yet."

    log_result, log_stdout, log_stderr, error_msg = _run_git_command(
        ["log", "-n", "10", "--oneline"],
        root,
    )

    if error_msg:
        return error_msg

    output = "\n".join(filter(None, [log_stdout, log_stderr]))

    if not log_result or log_result.returncode != 0:
        return "[ERROR] Unable to retrieve git history." + (f"\n{output}" if output else "")

    if log_stdout:
        return f"[GIT HISTORY]\n{log_stdout}"

    if output:
        return f"[GIT HISTORY]\n{output}"

    return "[GIT] No git history yet."


def send_pending_tool_response(
    messages: List[Dict],
    function_name: str,
    tool_identifier: str,
    pending_tracker: Set[Tuple[str, str]],
) -> None:
    key = (function_name, tool_identifier)
    if key in pending_tracker:
        log_verbose(
            f"Pending tool response déjà envoyé pour {function_name} -> {tool_identifier}"
        )
        return
    pending_message = {
        "role": "tool",
        "name": function_name,
        "content": "[PENDING] Awaiting user confirmation...",
    }
    messages.append(pending_message)
    pending_tracker.add(key)
    console.print(
        "[dim yellow]⏳ Réponse temporaire envoyée au modèle (en attente de confirmation utilisateur).[/dim yellow]"
    )
    log_verbose(f"Pending tool response envoyé pour {function_name} -> {tool_identifier}")
