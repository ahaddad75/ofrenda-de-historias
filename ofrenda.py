#!/usr/bin/env python3
"""
Ofrenda de Historias: turn a spoken memory into a written ofrenda, fully offline.

Pipeline:
  1. Whisper (mlx-whisper, runs locally on Apple Silicon) transcribes the recording.
  2. Gemma 4 (via Ollama, local) writes a remembrance and extracts the details
     that made the person who they were: dishes, songs, places, sayings, objects.

Nothing leaves the machine. No API key, no cloud.

Usage:
  python ofrenda.py recuerdo.m4a
  python ofrenda.py --text recuerdo.txt          # skip Whisper, start from a transcript
  python ofrenda.py recuerdo.m4a --model gemma4:e4b
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma4:12b"
DEFAULT_WHISPER = "mlx-community/whisper-large-v3-turbo"

# Structured output: Ollama constrains Gemma's answer to this JSON schema.
SCHEMA = {
    "type": "object",
    "properties": {
        "nombre": {"type": "string"},
        "remembranza": {"type": "string"},
        "platillos": {"type": "array", "items": {"type": "string"}},
        "canciones": {"type": "array", "items": {"type": "string"}},
        "lugares": {"type": "array", "items": {"type": "string"}},
        "frases": {"type": "array", "items": {"type": "string"}},
        "objetos": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["nombre", "remembranza", "platillos", "canciones", "lugares", "frases", "objetos"],
}

SYSTEM_PROMPT = """You help families build an ofrenda for Día de Muertos out of the stories they tell.
You receive the transcript of someone telling a memory about a loved one.

Rules:
- Write in the SAME language as the transcript.
- Use ONLY what the transcript says. Never invent a dish, a song, a place or a saying.
  If a category is not mentioned, return an empty list for it.
- "nombre": how the speaker calls the person (e.g. "mi abuela Lupita"). Empty string if unknown.
- "remembranza": 4 to 6 warm, simple sentences addressed to the person, in the spirit of an ofrenda.
  Keep the speaker's own words and details where you can. No clichés, no religious claims the speaker did not make.
- "frases": things the person used to say, quoted as the speaker reported them.
"""


def transcribe(audio_path: Path, whisper_model: str) -> tuple[str, str]:
    try:
        import mlx_whisper
    except ImportError:
        sys.exit("mlx_whisper is not installed: pip install mlx-whisper")
    print(f"Transcribing {audio_path.name} with Whisper (local)…", file=sys.stderr)
    result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=whisper_model)
    return result["text"].strip(), result.get("language", "?")


def _stream_chat(body: dict, deadline_s: int) -> tuple[str, str]:
    """Call Ollama in streaming mode, echo progress live, return (content, done_reason)."""
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    content, done_reason, start = [], "", time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            chunk = json.loads(raw)
            msg = chunk.get("message", {})
            piece = msg.get("content", "")
            content.append(piece)
            # Show thinking (dim) and content live so a stall is visible.
            if msg.get("thinking"):
                print(f"\033[2m{msg['thinking']}\033[0m", end="", file=sys.stderr, flush=True)
            if piece:
                print(piece, end="", file=sys.stderr, flush=True)
            if chunk.get("done"):
                done_reason = chunk.get("done_reason", "")
                break
            if time.time() - start > deadline_s:
                done_reason = "timeout"
                break
    print(file=sys.stderr)
    return "".join(content), done_reason


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def ask_gemma(transcript: str, model: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n\n{transcript}"},
    ]
    options = {"temperature": 0.4, "num_ctx": 8192, "num_predict": 1500}

    # Attempt 1: schema-constrained output. Attempt 2: plain JSON instructions,
    # in case constrained decoding stalls or loops on this model build.
    attempts = [
        {"model": model, "messages": messages, "format": SCHEMA, "stream": True, "think": False, "options": options},
        {
            "model": model,
            "messages": messages[:-1]
            + [{"role": "user", "content": messages[-1]["content"]
                + "\n\nAnswer with ONE JSON object only, with exactly these keys: "
                + ", ".join(SCHEMA["required"]) + ". No markdown, no comments."}],
            "stream": True,
            "think": False,
            "options": options,
        },
    ]

    for i, body in enumerate(attempts, 1):
        print(f"Writing the ofrenda with {model} (local), attempt {i}…", file=sys.stderr)
        try:
            text, reason = _stream_chat(body, deadline_s=180)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "ignore")
            if "think" in err.lower():
                body.pop("think")
                text, reason = _stream_chat(body, deadline_s=180)
            else:
                sys.exit(f"Ollama error: {err}")
        except urllib.error.URLError:
            sys.exit("Ollama is not reachable on localhost:11434. Open the Ollama app and retry.")
        try:
            data = _extract_json(text)
            missing = [k for k in SCHEMA["required"] if k not in data]
            for k in missing:
                data[k] = "" if k in ("nombre", "remembranza") else []
            return data
        except ValueError as e:
            print(f"Attempt {i} failed ({reason or e}).", file=sys.stderr)

    sys.exit("Gemma did not return a usable ofrenda. Run `ollama ps` and share the output.")


def to_markdown(ofrenda: dict, transcript: str) -> str:
    title = ofrenda["nombre"] or "Ofrenda"
    lines = [f"# {title}", "", ofrenda["remembranza"], ""]
    sections = [
        ("Platillos", "platillos"),
        ("Canciones", "canciones"),
        ("Lugares", "lugares"),
        ("Sus frases", "frases"),
        ("Objetos", "objetos"),
    ]
    for label, key in sections:
        if ofrenda[key]:
            lines.append(f"## {label}")
            lines += [f"- {item}" for item in ofrenda[key]]
            lines.append("")
    lines += ["---", "", "## Transcripción", "", transcript, ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a spoken memory into an ofrenda, offline.")
    parser.add_argument("audio", nargs="?", type=Path, help="Recording (m4a, mp3, wav…)")
    parser.add_argument("--text", type=Path, help="Use an existing transcript instead of audio")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    parser.add_argument("--whisper", default=DEFAULT_WHISPER, help="Whisper model for mlx-whisper")
    parser.add_argument("--out", type=Path, default=Path("ofrendas"), help="Output folder")
    args = parser.parse_args()

    if args.text:
        transcript, lang = args.text.read_text(encoding="utf-8").strip(), "?"
        stem = args.text.stem
    elif args.audio:
        transcript, lang = transcribe(args.audio, args.whisper)
        stem = args.audio.stem
    else:
        parser.error("give an audio file or --text transcript.txt")

    ofrenda = ask_gemma(transcript, args.model)
    ofrenda["idioma"] = lang
    ofrenda["transcripcion"] = transcript

    args.out.mkdir(exist_ok=True)
    (args.out / f"{stem}.json").write_text(json.dumps(ofrenda, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / f"{stem}.md").write_text(to_markdown(ofrenda, transcript), encoding="utf-8")

    print(to_markdown(ofrenda, transcript))
    print(f"\nSaved to {args.out / stem}.json and .md", file=sys.stderr)


if __name__ == "__main__":
    main()
