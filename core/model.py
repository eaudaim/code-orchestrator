import json
from typing import Any, Callable, Dict, List, Tuple

import ollama
from rich.console import Console

console = Console(force_terminal=True)

MODEL_NATIVE_TOOLS = True
MODEL_JSON_FALLBACK = False
TOOLS: List[Dict[str, Any]] = []
log_verbose: Callable[[str], None] = lambda message: None
parse_json_tool_calls: Callable[[str], List[Dict[str, Any]]] = lambda text: []


def validate_messages_json(messages: List[Dict[str, Any]]) -> Tuple[bool, str | None]:
    """Valide que les messages peuvent être sérialisés en JSON avant l'envoi à Ollama."""

    try:
        json.dumps(messages)
        return True, None
    except (TypeError, ValueError) as e:
        return False, f"Invalid JSON in messages: {e}"


def set_model_context(
    tools: List[Dict[str, Any]],
    model_native_tools: bool,
    model_json_fallback: bool,
    debug_logger: Callable[[str], None] | None = None,
    shared_console: Console | None = None,
    json_parser: Callable[[str], List[Dict[str, Any]]] | None = None,
) -> None:
    """Configure les dépendances nécessaires au module modèle."""

    global TOOLS, MODEL_NATIVE_TOOLS, MODEL_JSON_FALLBACK, log_verbose, console, parse_json_tool_calls

    TOOLS = tools or []
    MODEL_NATIVE_TOOLS = model_native_tools
    MODEL_JSON_FALLBACK = model_json_fallback

    if debug_logger is not None:
        log_verbose = debug_logger

    if shared_console is not None:
        console = shared_console

    if json_parser is not None:
        parse_json_tool_calls = json_parser


def call_model_and_stream(
    messages: List[Dict[str, Any]],
    model_name: str,
    reasoning_level: str,
) -> Tuple[Dict[str, Any], bool, List[Dict[str, Any]]]:
    """Appelle le modèle et gère le streaming, retourne le message assistant et les tool calls."""

    console.print("[dim]🤖 Le modèle réfléchit...[/dim]")
    log_verbose(f"Nombre de messages dans l'historique : {len(messages)}")

    valid, error = validate_messages_json(messages)
    if not valid:
        console.print(f"[yellow]⚠️ Messages JSON invalides : {error}[/yellow]")
        error_content = f"[JSON_ERROR] Invalid message format: {error}"
        assistant_message = {"role": "assistant", "content": error_content}
        return assistant_message, False, []

    try:
        log_verbose(f"Appel à ollama.chat() avec modèle {model_name}, reasoning={reasoning_level}")
        ollama_params: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "num_ctx": 16384,  # Contexte maximum pour gpt-oss
                "temperature": 0.2,  # Encore plus strict
                "repeat_penalty": 1.3,  # Anti-répétition
                "reasoning": reasoning_level,
            },
        }
        if MODEL_NATIVE_TOOLS:
            ollama_params["tools"] = TOOLS

        response = ollama.chat(**ollama_params)
        log_verbose("Réponse du modèle reçue, début du streaming")
    except Exception as e:
        console.print(f"[red]❌ Erreur lors de l'appel au modèle : {e}[/red]")
        log_verbose(f"Exception complète : {e}")
        error_content = f"[MODEL_ERROR] Error during model initialization: {str(e)}"
        assistant_message = {"role": "assistant", "content": error_content}
        return assistant_message, False, []

    assistant_content = ""
    thinking_content = ""
    has_tool_calls = False
    tool_calls_data: List[Dict[str, Any]] = []

    chunk_count = 0
    is_thinking = False

    console.print("[bold blue]Assistant> [/bold blue]")
    try:
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
    except ollama.ResponseError as e:
        console.print(f"[yellow]⚠️ Tool call malformé ignoré : {e}[/yellow]")
        log_verbose(f"Ollama ResponseError détaillée : {e}")
        error_content = f"[TOOL_CALL_ERROR] Malformed JSON detected by Ollama: {str(e)}"
        assistant_message = {"role": "assistant", "content": error_content}
        return assistant_message, False, []
    except Exception as e:
        console.print(f"[yellow]⚠️ Erreur streaming capturée : {e}[/yellow]")
        log_verbose(f"Exception streaming détaillée : {e}")
        error_content = f"[STREAMING_ERROR] Unexpected error during model response: {str(e)}"
        assistant_message = {"role": "assistant", "content": error_content}
        return assistant_message, False, []

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

    assistant_message: Dict[str, Any] = {"role": "assistant", "content": assistant_content}
    if has_tool_calls:
        assistant_message["tool_calls"] = tool_calls_data

    return assistant_message, has_tool_calls, tool_calls_data
