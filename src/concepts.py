"""Per-grade concept vocabulary from a stored, hand-curated JSON table.

The vocabulary is read from CONFIG["concept_vocab_path"], keyed by grade — no LLM is used.
Each grade's concepts are used VERBATIM: every listed concept becomes a basis column, in
order, with no lexical/CLIP filtering and no fixed size. Concepts may be given as plain
strings or as {"label": ...} objects (extra fields such as id / category / description are
kept in the table but ignored here).
"""

import json
import os

from config import CONFIG, cache_name, resolve_vocab_key


def _flatten_concepts(entry):
    """Lowercased, de-duplicated concept labels from one vocab entry (order preserved).

    Supports both vocab-entry shapes:
      - a flat list: ["dot hemorrhage", {"label": "..."}, ...]
      - a grouped dict: {"description": <str>, "<category>": [concepts...], "notes": {...}}
        -> every LIST-valued field is a concept group; non-list fields (the description
           text, a notes dict) are metadata and skipped.
    """
    groups = entry.values() if isinstance(entry, dict) else [entry]
    out, seen = [], set()
    for group in groups:
        if not isinstance(group, (list, tuple)):
            continue                                  # skip description / notes metadata
        for c in group:
            label = (c["label"] if isinstance(c, dict) else c)
            label = str(label).strip().lower()
            if label and label not in seen:
                seen.add(label)
                out.append(label)
    return out


def _load_vocab(cls):
    """Load the stored candidate concepts for grade `cls` (lowercased strings).

    The vocab table is keyed by snake_case grade identifiers; `cls` is the dataset folder
    name (e.g. '2_moderate_npdr'), matched by canonical key (see config.resolve_vocab_key).
    Each entry may be a flat list or a grouped dict (see _flatten_concepts).
    """
    path = CONFIG["concept_vocab_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Concept vocabulary table not found at {path}. "
            f"Create it (see concept_vocab.json) keyed by grade name."
        )
    with open(path) as f:
        table = json.load(f)
    key = resolve_vocab_key(cls, table.keys())   # exact canonical match, then aliases
    if key is None:
        raise KeyError(
            f"No concept vocabulary for grade '{cls}' in {path}. Available keys: "
            f"{sorted(table)}. If the folder name differs from its key (e.g. a bare "
            f"'2' or 'Moderate NPDR' folder vs. key '2_moderate_npdr'), add it to "
            f"CONCEPT_ALIASES in config.py."
        )
    return _flatten_concepts(table[key])


def get_concepts(vlm=None, images=None):
    """Return the target grade's curated concept list, cached to JSON.

    Every concept listed for the grade becomes a basis column, in order, verbatim. `vlm`
    and `images` are accepted for call-site symmetry with the rest of the pipeline but are
    unused (there is no CLIP/lexical filtering step).
    """
    path = os.path.join(CONFIG["cache_dir"], cache_name("concepts", ".json", "con"))
    if os.path.exists(path):
        with open(path) as f:
            print("[cache] loaded concepts.json")
            return json.load(f)

    cls = CONFIG["class_name"]
    concepts = list(dict.fromkeys(_load_vocab(cls)))       # verbatim, de-duplicated, ordered
    if not concepts:
        raise ValueError(
            f"No concepts listed for grade '{cls}' in {CONFIG['concept_vocab_path']}.")
    if len(concepts) < 5:
        print(f"[warn] only {len(concepts)} concepts for '{cls}' — the concept basis will "
              f"be very thin (reconstruction / C-Insertion may be degenerate); consider "
              f"adding a few more to concept_vocab.json.")
    with open(path, "w") as f:
        json.dump(concepts, f, indent=2)
    print(f"[cache] saved concepts.json ({len(concepts)} curated concepts for '{cls}')")
    return concepts
