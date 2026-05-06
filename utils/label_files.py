#!/usr/bin/env python3
"""
Interactively label WAV files in a directory tree and save results to JSON.

Files ending in _partXXX are treated as chunks of a parent file.
The parent is labelled once and all its chunks inherit that label automatically.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# ── Presets ──────────────────────────────────────────────────────────────────
SEARCH_DIR  = "./train_diffusion/"  # Directory to scan for WAV files
OUTPUT_FILE = "train_labels.json"          # Output JSON filename
# ─────────────────────────────────────────────────────────────────────────────

PART_RE = re.compile(r"^(.+)_part\d{3}$")


def is_chunk(stem: str) -> bool:
    return bool(PART_RE.match(stem))


def parent_stem(stem: str) -> str:
    """Return the stem without the _partXXX suffix."""
    m = PART_RE.match(stem)
    return m.group(1) if m else stem


def collect_wav_files(root: Path) -> list[Path]:
    """Return all .wav files under *root*, sorted."""
    return sorted(root.rglob("*.wav"))


def group_files(root: Path, wav_files: list[Path]):
    """
    Separate files into:
      - originals: files that are NOT chunks (no _partXXX suffix)
      - chunks: files that ARE chunks, keyed by (parent_dir, parent_stem)
    Returns (originals, chunks_map).
    """
    originals: list[Path] = []
    chunks_map: dict[tuple[Path, str], list[Path]] = defaultdict(list)

    for wav in wav_files:
        if is_chunk(wav.stem):
            key = (wav.parent, parent_stem(wav.stem))
            chunks_map[key].append(wav)
        else:
            originals.append(wav)

    return originals, chunks_map


def prompt_label(relative: Path, index: int, total: int) -> str:
    """Display the file path and prompt the user for a label."""
    print(f"\n[{index}/{total}] {relative}")
    try:
        label = input("  Label: ").strip()
    except EOFError:
        label = ""
    return label


def build_output(root: Path, labelled: list[tuple[Path, str]]) -> list[dict]:
    """
    Group labelled files by their immediate parent directory (relative to root)
    and return the JSON-ready structure.
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for abs_path, label in labelled:
        rel = abs_path.relative_to(root)
        dir_key = str(rel.parent)
        groups[dir_key].append({"file": str(rel), "label": label})

    return [
        {"directory": directory, "labels": files}
        for directory, files in sorted(groups.items())
    ]


def main():
    root = Path(SEARCH_DIR).expanduser().resolve()

    if not root.is_dir():
        print(f"Error: '{root}' is not a valid directory.")
        return

    wav_files = collect_wav_files(root)

    if not wav_files:
        print(f"No WAV files found under '{root}'.")
        return

    originals, chunks_map = group_files(root, wav_files)

    # Files to prompt: originals + any chunk group whose parent wasn't found
    to_prompt: list[tuple[Path, list[Path]]] = []  # (representative file, [all files to label])
    orphan_chunks: list[Path] = []

    for wav in originals:
        key = (wav.parent, wav.stem)
        related_chunks = chunks_map.pop(key, [])
        to_prompt.append((wav, [wav] + sorted(related_chunks)))

    # Remaining chunks_map entries have no matching original — prompt them individually
    for (parent_dir, stem), chunks in sorted(chunks_map.items()):
        # Use first chunk as representative, label all of them together
        to_prompt.append((chunks[0], sorted(chunks)))

    total = len(to_prompt)
    print(f"Found {len(wav_files)} WAV file(s) ({total} label prompt(s)) under '{root}'")
    print("Chunk files (_partXXX) automatically inherit their parent's label.")
    print("Press Enter with no input to leave a label blank.\n")
    print("─" * 60)

    labelled: list[tuple[Path, str]] = []

    for i, (representative, all_files) in enumerate(to_prompt, start=1):
        rel = representative.relative_to(root)
        label = prompt_label(rel, i, total)

        if len(all_files) > 1:
            chunk_rels = [f.relative_to(root) for f in all_files[1:]]
            print(f"  ↳ Applying to {len(chunk_rels)} chunk(s): {', '.join(str(r) for r in chunk_rels)}")

        for f in all_files:
            labelled.append((f, label))

    output_data = build_output(root, labelled)

    output_path = Path(OUTPUT_FILE)
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    print(f"\n✓ Labels saved to '{output_path.resolve()}'")


if __name__ == "__main__":
    main()