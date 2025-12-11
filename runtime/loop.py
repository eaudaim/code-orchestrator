import select
import sys
from typing import Any, Dict, List, Set, Tuple

try:
    import config.settings as cfg

    MODEL_NAME = cfg.MODEL_NAME
    REASONING_LEVEL = cfg.REASONING_LEVEL
    MAX_AUTONOMY_ITERATIONS = cfg.MAX_AUTONOMY_ITERATIONS
    AUTONOMY_TIMEOUT = cfg.AUTONOMY_TIMEOUT
    AUTONOMY = cfg.AUTONOMY
    EXEC_TIMEOUT = cfg.EXEC_TIMEOUT
    MAX_RETRIES = cfg.MAX_RETRIES
except Exception:
    MODEL_NAME = "gpt-oss:20b"
    REASONING_LEVEL = "medium"
    MAX_AUTONOMY_ITERATIONS = 20
    AUTONOMY_TIMEOUT = 5
    AUTONOMY = True
    EXEC_TIMEOUT = 30
    MAX_RETRIES = 3

from core.model import call_model_and_stream
from runtime.executor import (
    create_venv_tool,
    execute_code_tool,
    git_commit_tool,
    git_history_tool,
    git_init_tool,
    git_rollback_tool,
    list_files_tool,
    read_file_tool,
    send_pending_tool_response,
    WORK_DIR,
    write_file_tool,
)

console = None
log_verbose = lambda message: None  # noqa: E731


def set_loop_environment(
    console_obj,
    logger,
    model_name: str,
    reasoning_level: str,
    max_autonomy_iterations: int,
    autonomy_timeout: int,
    autonomy_flag: bool,
    exec_timeout: int,
    max_retries: int,
) -> None:
    global console, log_verbose, MODEL_NAME, REASONING_LEVEL, MAX_AUTONOMY_ITERATIONS
    global AUTONOMY_TIMEOUT, AUTONOMY, EXEC_TIMEOUT, MAX_RETRIES

    console = console_obj
    log_verbose = logger
    MODEL_NAME = model_name
    REASONING_LEVEL = reasoning_level
    MAX_AUTONOMY_ITERATIONS = max_autonomy_iterations
    AUTONOMY_TIMEOUT = autonomy_timeout
    AUTONOMY = autonomy_flag
    EXEC_TIMEOUT = exec_timeout
    MAX_RETRIES = max_retries


def wait_for_manual_override(timeout: int) -> bool:
    console.print(
        f"[cyan]⏳ Autonomie : appuyez sur une touche dans les {timeout}s pour reprendre le contrôle.[/cyan]"
    )
    console.print("[dim]Passé ce délai, l'exécution automatique se poursuit.[/dim]")

    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except Exception as e:
        log_verbose(f"Impossible de surveiller l'entrée utilisateur pendant l'autonomie : {e}")
        return False

    if ready:
        try:
            sys.stdin.readline()
        except Exception:
            pass
        console.print(
            "[green]🎛️ Reprise manuelle détectée : le mode autonomie est désactivé pour cette session.[/green]"
        )
        return True

    console.print("[dim]⏭️  Aucun input détecté, poursuite de l'autonomie.[/dim]")
    return False


def show_autonomy_banner(iteration: int, limit: int) -> None:
    console.print(f"[yellow]🤖 AUTONOMIE ON - Itération {iteration}/{limit}[/yellow]")


def is_affirmative(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"o", "oui", "y", "yes", "true", "1"}
    return False


def autonomy_loop(messages: List[Dict[str, Any]], files_data: Dict[str, str]):
    global AUTONOMY

    execution_count: Dict[str, int] = {}
    pending_tool_responses: Set[Tuple[str, str]] = set()
    tool_call_depth = 0
    autonomy_call_counter = 0
    autonomy_first_successful_write = False

    if AUTONOMY:
        show_autonomy_banner(autonomy_call_counter + 1, MAX_AUTONOMY_ITERATIONS)
    assistant_message, has_tool_calls, tool_calls_data = call_model_and_stream(
        messages, MODEL_NAME, REASONING_LEVEL
    )
    if AUTONOMY:
        autonomy_call_counter += 1
    messages.append(assistant_message)

    while has_tool_calls and tool_call_depth < MAX_AUTONOMY_ITERATIONS:
        log_verbose(f"Traitement de {len(tool_calls_data)} tool call(s)")

        tool_call_depth += 1

        for call in tool_calls_data:
            function_name = call.get("function", {}).get("name", "")
            arguments = call.get("function", {}).get("arguments", {})

            console.print(f"\n[cyan]📋 DEBUG Tool Call:[/cyan]")
            console.print(f"[dim]   Fonction : {function_name}[/dim]")
            console.print(f"[dim]   Arguments bruts : {arguments}[/dim]")
            console.print(f"[dim]   Type arguments : {type(arguments)}[/dim]")

            log_verbose(f"Tool call brut complet : {call}")

            valid_tools = {
                "list_files",
                "read_file",
                "write_file",
                "execute_code",
                "create_venv",
                "git_init",
                "git_commit",
                "git_rollback",
                "git_history",
            }
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

6. git_init()
   - Initializes Git repository if missing
   - Parameters: none (safe)

7. git_commit(message)
   - Runs git add -A then creates a commit with provided message
   - Parameter: message (required, string)

8. git_rollback(steps=1)
   - Hard resets to HEAD~steps (git reset --hard); destructive and discards uncommitted changes
   - Parameter: steps (optional, default: 1)

9. git_history()
   - Displays the last 10 commits
   - Parameters: none (read-only)

 CRITICAL: Do NOT invent tools like 'apply_patch', 'git_diff', 'patch_file', etc.
 ONLY use the tools listed above. No exceptions."""

                result = f"[ERROR] Tool '{function_name}' does not exist!\n{available_tools_list}"

                messages.append({
                    "role": "tool",
                    "content": result,
                    "name": function_name,
                })
                continue

            if isinstance(arguments, str):
                try:
                    import json

                    arguments = json.loads(arguments)
                    console.print(f"[dim]   Arguments après parsing : {arguments}[/dim]")
                except json.JSONDecodeError as e:
                    console.print(f"[red]   ❌ Erreur parsing JSON : {e}[/red]")
                    result = f"[ERROR] Invalid JSON arguments: {e}"
                    messages.append(
                        {"role": "tool", "content": result, "name": function_name}
                    )
                    continue

            if function_name == "read_file":
                file_path = arguments.get("file_path") or arguments.get("path", "")

                invalid_params = [
                    key for key in arguments.keys() if key not in {"file_path", "path"}
                ]
                if invalid_params:
                    console.print(
                        f"[red]   ❌ PARAMÈTRES INCORRECTS pour read_file: {invalid_params}[/red]"
                    )
                    result = (
                        f"[ERROR] read_file called with invalid parameters: {invalid_params}\n"
                        "read_file ONLY accepts 'file_path' parameter.\n"
                        "Do NOT use line_start, line_end, or any other parameters with read_file!\n"
                        "Correct usage: read_file(file_path='filename.py')"
                    )
                    messages.append({"role": "tool", "content": result, "name": function_name})
                    continue

                if not file_path:
                    console.print(
                        f"[red]   ❌ ARGUMENT MANQUANT: file_path est vide ou absent ![/red]"
                    )
                    console.print(f"[red]   Arguments reçus : {arguments}[/red]")
                    result = "[ERROR] Missing required argument 'file_path'. The model did not provide the file path to read."
                    messages.append({"role": "tool", "content": result, "name": function_name})
                    continue

                console.print(f"[green]   ✓ Fichier détecté : {file_path}[/green]")
                console.print(f"[dim]   📂 Lecture de : {file_path}[/dim]")
                file_content = read_file_tool(file_path)
                log_verbose(f"Fichier lu : {len(file_content)} caractères")

                messages.append(
                    {"role": "tool", "content": file_content, "name": function_name}
                )

            elif function_name == "list_files":
                directory_path = arguments.get("directory_path", ".")

                console.print(f"[green]   ✓ Répertoire détecté : {directory_path}[/green]")
                console.print(f"[dim]   📂 Listage de : {directory_path}[/dim]")
                file_list = list_files_tool(directory_path)
                log_verbose(f"Répertoire listé : {len(file_list)} caractères")

                messages.append(
                    {"role": "tool", "content": file_list, "name": function_name}
                )

            elif function_name == "write_file":
                file_path = arguments.get("file_path") or arguments.get("path", "")
                content = arguments.get("content", "")
                raw_line_start = arguments.get("line_start")
                raw_line_end = arguments.get("line_end")

                if not file_path:
                    console.print(
                        f"[red]   ❌ ARGUMENT MANQUANT: file_path/path est vide ou absent ![/red]"
                    )
                    console.print(f"[red]   Arguments reçus : {arguments}[/red]")
                    result = (
                        "[ERROR] Missing required argument 'file_path' or 'path'. "
                        "The model must specify which file to write to."
                    )
                    messages.append({"role": "tool", "content": result, "name": function_name})
                    continue

                if content is None or content.strip() == "":
                    console.print(
                        f"[red]   ❌ Refus d'écrire un contenu vide dans '{file_path}'[/red]"
                    )
                    console.print(f"[red]   Arguments reçus : {arguments}[/red]")
                    result = (
                        "[ERROR] Empty 'content' provided to write_file. "
                        "The model must supply the new file content."
                    )
                    messages.append({"role": "tool", "content": result, "name": function_name})
                    continue

                if raw_line_start is None or raw_line_end is None:
                    console.print(
                        "[red]   ❌ line_start et line_end sont OBLIGATOIRES pour write_file[/red]"
                    )
                    result = (
                        "[ERROR] write_file requires 'line_start' and 'line_end' parameters.\n"
                        "- Insert at line N: line_start=N, line_end=N-1\n"
                        "- Replace lines M-N: line_start=M, line_end=N\n"
                        "- Replace entire file: line_start=1, line_end=-1"
                    )
                    messages.append({"role": "tool", "content": result, "name": function_name})
                    continue

                try:
                    line_start = int(raw_line_start)
                    line_end = int(raw_line_end)
                except (TypeError, ValueError):
                    console.print(
                        f"[red]   ❌ line_start/line_end invalides : {raw_line_start}/{raw_line_end}[/red]"
                    )
                    result = f"[ERROR] Invalid line_start/line_end values: {raw_line_start}/{raw_line_end}."
                    messages.append({"role": "tool", "content": result, "name": function_name})
                    continue

                if line_start < 1 and line_end not in {0, -1}:
                    console.print(
                        f"[red]   ❌ Indices de lignes invalides : start={line_start}, end={line_end}[/red]"
                    )
                    result = (
                        "[ERROR] Invalid line indices. line_start must be >= 1 (or line_end == -1 for full replacement)."
                    )
                    messages.append({"role": "tool", "content": result, "name": function_name})
                    continue

                log_verbose(
                    f"write_file call -> path={file_path}, line_start={line_start}, line_end={line_end}, content_length={len(content)}"
                )
                send_pending_tool_response(
                    messages,
                    function_name,
                    file_path,
                    pending_tool_responses,
                )
                console.print(f"[dim]   ✍️  Écriture dans : {file_path} (lignes {line_start}-{line_end})[/dim]")
                file_was_existing = (WORK_DIR / file_path).exists()
                success, result = write_file_tool(file_path, content, line_start, line_end)
                if AUTONOMY and success and not autonomy_first_successful_write:
                    console.print(
                        "[green]🤖 Première écriture réussie en autonomie : poursuivons avec prudence.[/green]"
                    )
                messages.append({"role": "tool", "content": result, "name": function_name})
                # Auto-commit en autonomie (préservé de l'original)
                if AUTONOMY and success:
                    if not autonomy_first_successful_write:
                        autonomy_first_successful_write = True
                        git_dir = WORK_DIR / ".git"
                        if not git_dir.exists():
                            init_result = git_init_tool()
                            log_verbose(f"Initialisation git automatique : {init_result[:200]}...")
                            messages.append(
                                {
                                    "role": "tool",
                                    "content": init_result,
                                    "name": "git_init",
                                }
                            )

                    commit_message = (
                        f"Auto: {'Modified' if file_was_existing else 'Created'} {file_path}"
                    )
                    commit_result = git_commit_tool(commit_message)
                    console.print(
                        f"[cyan]🤖 Auto-commit executed after write to {file_path}[/cyan]"
                    )
                    log_verbose(f"Commit automatique : {commit_result[:200]}...")
                    messages.append(
                        {
                            "role": "tool",
                            "content": commit_result,
                            "name": "git_commit",
                        }
                    )
                pending_tool_responses.discard((function_name, file_path))

            elif function_name == "execute_code":
                file_path = arguments.get("file_path") or arguments.get("path", "")

                if not file_path:
                    console.print(
                        f"[red]   ❌ ARGUMENT MANQUANT: file_path/path est vide ou absent ![/red]"
                    )
                    console.print(f"[red]   Arguments reçus : {arguments}[/red]")
                    result = "[ERROR] Missing required argument 'file_path' or 'path'. The model must specify which file to execute."
                    messages.append({"role": "tool", "content": result, "name": function_name})
                    continue

                console.print(f"[green]   ✓ Fichier détecté : {file_path}[/green]")

                execution_count[file_path] = execution_count.get(file_path, 0) + 1

                if execution_count[file_path] > MAX_RETRIES:
                    result = (
                        f"[ERROR] Maximum execution attempts ({MAX_RETRIES}) reached for {file_path}. Please review the code manually."
                    )
                    console.print(f"[red]   ❌ Limite de tentatives atteinte ({MAX_RETRIES})[/red]")
                    log_verbose(f"Limite d'exécution atteinte pour {file_path}")
                else:
                    send_pending_tool_response(
                        messages,
                        function_name,
                        file_path,
                        pending_tool_responses,
                    )
                    console.print(
                        f"[dim]   🚀 Exécution de : {file_path} (tentative {execution_count[file_path]}/{MAX_RETRIES})[/dim]"
                    )
                    result = execute_code_tool(file_path)
                    log_verbose(f"Résultat de l'exécution : {result[:200]}...")

                messages.append({"role": "tool", "content": result, "name": function_name})
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
                messages.append({"role": "tool", "content": result, "name": function_name})
                pending_tool_responses.discard((function_name, venv_path))

            elif function_name == "git_init":
                console.print("[dim]   🧰 Initialisation du dépôt Git[/dim]")
                result = git_init_tool()
                log_verbose(f"Résultat git_init : {result[:200]}...")
                messages.append({"role": "tool", "content": result, "name": function_name})

            elif function_name == "git_commit":
                message = arguments.get("message", "")
                console.print("[dim]   🧰 Commit Git demandé[/dim]")
                git_dir = WORK_DIR / ".git"
                if not git_dir.exists():
                    result = "[ERROR] Git repository is not initialized. Run git_init first." + " Suggestion: run git_init first."
                    messages.append({"role": "tool", "content": result, "name": function_name})
                    continue

                pending_key = message or "<commit-message>"
                if AUTONOMY or is_affirmative(arguments.get("confirm")):
                    result = git_commit_tool(message)
                else:
                    send_pending_tool_response(
                        messages,
                        function_name,
                        pending_key,
                        pending_tool_responses,
                    )
                    console.print("[yellow]⚠️ Confirmation requise pour appliquer le commit.[/yellow]")
                    confirmation = input(
                        f"Confirmer le commit git avec le message \"{message}\" ? (o/n) : "
                    )
                    if is_affirmative(confirmation):
                        result = git_commit_tool(message)
                    else:
                        result = "[CANCELLED] Git commit cancelled by user."
                pending_tool_responses.discard((function_name, pending_key))
                log_verbose(f"Résultat git_commit : {result[:200]}...")
                messages.append({"role": "tool", "content": result, "name": function_name})

            elif function_name == "git_rollback":
                steps = arguments.get("steps", 1)
                console.print(f"[dim]   🧰 Rollback Git de {steps} étape(s) demandé[/dim]")
                git_dir = WORK_DIR / ".git"
                if not git_dir.exists():
                    result = "[ERROR] Git repository is not initialized. Run git_init first." + " Suggestion: run git_init first."
                    messages.append({"role": "tool", "content": result, "name": function_name})
                    continue

                pending_key = f"rollback-{steps}"
                if AUTONOMY or is_affirmative(arguments.get("confirm")):
                    result = git_rollback_tool(steps)
                else:
                    send_pending_tool_response(
                        messages,
                        function_name,
                        pending_key,
                        pending_tool_responses,
                    )
                    console.print("[yellow]⚠️ Rollback Git destructif mis en attente de confirmation utilisateur.[/yellow]")
                    confirmation = input("Confirmer le rollback git ? (o/n) : ")
                    if is_affirmative(confirmation):
                        result = git_rollback_tool(steps)
                    else:
                        result = "[CANCELLED] Git rollback cancelled by user."
                pending_tool_responses.discard((function_name, pending_key))
                log_verbose(f"Résultat git_rollback : {result[:200]}...")
                messages.append({"role": "tool", "content": result, "name": function_name})

            elif function_name == "git_history":
                console.print("[dim]   🧰 Historique Git demandé[/dim]")
                result = git_history_tool()
                log_verbose(f"Résultat git_history : {result[:200]}...")
                messages.append({"role": "tool", "content": result, "name": function_name})

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
                if AUTONOMY:
                    show_autonomy_banner(autonomy_call_counter + 1, MAX_AUTONOMY_ITERATIONS)
                assistant_message, has_tool_calls, tool_calls_data = call_model_and_stream(messages)
                if AUTONOMY:
                    autonomy_call_counter += 1
                messages.append(assistant_message)
                if not has_tool_calls:
                    break
            else:
                log_verbose("L'utilisateur a confirmé la poursuite des actions.")

        if AUTONOMY:
            if wait_for_manual_override(AUTONOMY_TIMEOUT):
                AUTONOMY = False
                has_tool_calls = False
                console.print(
                    "[yellow]↩️ Retour à l'invite manuelle : en attente de votre prochaine instruction.[/yellow]"
                )
                break

        if AUTONOMY:
            show_autonomy_banner(autonomy_call_counter + 1, MAX_AUTONOMY_ITERATIONS)
        assistant_message, has_tool_calls, tool_calls_data = call_model_and_stream(messages)
        if AUTONOMY:
            autonomy_call_counter += 1
        messages.append(assistant_message)

        if not (AUTONOMY and has_tool_calls and tool_call_depth < MAX_AUTONOMY_ITERATIONS):
            if AUTONOMY and has_tool_calls and tool_call_depth >= MAX_AUTONOMY_ITERATIONS:
                console.print(
                    "[yellow]⚠️  Arrêt automatique: limite d'itérations atteinte avant nouveaux tool calls.[/yellow]"
                )
            break

    log_verbose("Réponse de l'assistant ajoutée à l'historique")
