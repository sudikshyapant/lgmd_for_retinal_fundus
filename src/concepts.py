"""Class-specific concept vocabulary from a stored table + two-stage filtering.

The vocabulary is read from a stored JSON table (CONFIG["concept_vocab_path"]),
keyed by class name — no external LLM API is used. Each class maps to a list of
candidate concepts, given either as plain strings or as {"label": ...} objects
(extra fields such as id / category / description are kept in the table but ignored
here). The table is over-provided on purpose so the filters have headroom to land
on exactly r concepts.

Filtering mirrors the supplementary material:
  Stage 1 — lexical (suppl. A1.3): keep 2-3 word concepts; drop generic filler
    terms (rule i) and concepts overlapping the class name (rule ii), with the
    exception that concepts carrying a visual-attribute word are preserved.
  Stage 2 — CLIP semantic (suppl. A1.4): rank surviving concepts by cosine
    similarity to the mean CLIP embedding of class images (when images are
    available), then greedily keep diverse concepts, enforcing pairwise CLIP-text
    cosine similarity below CONFIG["dedup_threshold"]. Result: exactly r concepts.
"""

import json
import os
import re

from config import CONFIG, cache_name


def _canonical_key(name):
    """Canonical snake_case key for a class name (ML-standard identifier).

    Lowercases and collapses runs of whitespace/hyphens to single underscores,
    e.g. 'Great White Shark' -> 'great_white_shark'. Keys in concept_vocab.json
    use this form; CONFIG['class_name'] may be the human-readable variant.
    """
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _load_vocab(cls):
    """Load the stored candidate concepts for `cls` (lowercased strings).

    The vocab table is keyed by snake_case identifiers, but `cls` is typically the
    human-readable class name (e.g. 'tabby cat'); we match by canonical key so
    either form resolves.
    """
    path = CONFIG["concept_vocab_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Concept vocabulary table not found at {path}. "
            f"Create it (see concept_vocab.json) keyed by class name."
        )
    with open(path) as f:
        table = json.load(f)
    canon = {_canonical_key(k): k for k in table}
    key = canon.get(_canonical_key(cls))
    if key is None:
        raise KeyError(
            f"No concept vocabulary for class '{cls}' in {path}. "
            f"Available classes: {sorted(table)}"
        )
    raw = []
    for c in table[key]:
        label = (c["label"] if isinstance(c, dict) else c).strip().lower()
        if label:
            raw.append(label)
    return raw


def _lexical_filter(concepts, cls):
    """Stage-1 lexical filter (suppl. A1.3).

    Keep concepts that are 2-3 words; drop generic filler terms (rule i) and
    concepts overlapping the class name (rule ii) — except concepts that carry a
    visual-attribute word, which are preserved despite partial overlap. Exact
    duplicates are removed, original order preserved.
    """
    cls_words = set(re.findall(r"\w+", cls.lower()))
    filler = set(CONFIG["concept_filler_terms"])
    attr = set(CONFIG["concept_attribute_terms"])
    wmin, wmax = CONFIG["concept_word_min"], CONFIG["concept_word_max"]
    kept, seen = [], set()
    for c in concepts:
        words = re.findall(r"\w+", c)
        wset = set(words)
        if not (wmin <= len(words) <= wmax):          # word-count constraint
            continue
        if wset & filler:                             # (i) generic filler
            continue
        if (wset & cls_words) and not (wset & attr):  # (ii) class-name overlap (attribute-exempt)
            continue
        if c in seen:
            continue
        seen.add(c)
        kept.append(c)
    return kept


def _clip_select(concepts, clip, images, threshold, r):
    """Stage-2 CLIP semantic filter (suppl. A1.4).

    Rank concepts by cosine similarity to the mean CLIP embedding of up to
    CONFIG["concept_proto_images"] class images (when `images` is given); without
    images the lexical order is kept. Then greedily select diverse concepts,
    rejecting any whose CLIP-text similarity to an already-kept concept exceeds
    `threshold`. Returns at most r concepts.
    """
    if not concepts:
        return []
    prompts = [CONFIG["prompt_template"].format(c) for c in concepts]
    text_emb = clip.embed_text(prompts)                 # (n, d), L2-normalized

    order = list(range(len(concepts)))
    if images:
        from data_utils import clip_preprocess       # lazy: avoids torch import when unused
        sample = images[:CONFIG["concept_proto_images"]]
        proto = clip.embed_images([clip_preprocess(im) for im in sample]).mean(0)
        proto = proto / proto.norm()                    # class image prototype mu_I
        scores = text_emb @ proto                       # s_i = <t_i, mu_I>
        order.sort(key=lambda i: float(scores[i]), reverse=True)

    kept, kept_emb = [], []
    for i in order:
        emb = text_emb[i]
        if any(float(emb @ e) > threshold for e in kept_emb):   # near-duplicate in CLIP-text space
            continue
        kept.append(concepts[i])
        kept_emb.append(emb)
        if len(kept) == r:
            break
    return kept


def _shared_bank():
    """Flattened, de-duplicated shared concept bank used for every class (shared mode).

    Reads concept_vocab.json and flattens ALL of its values into one ordered list,
    regardless of how the file groups them (anatomical groups, a flat list, etc.) — the
    grouping is purely for human readability. Concepts are used verbatim (curated):
    no lexical/CLIP reduction, so r = bank size and the basis columns are identical and
    comparable across classes.
    """
    path = CONFIG["concept_vocab_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept vocabulary not found at {path}.")
    with open(path) as f:
        table = json.load(f)
    bank, seen = [], set()
    values = table.values() if isinstance(table, dict) else [table]
    for group in values:
        for c in group:
            label = (c["label"] if isinstance(c, dict) else c).strip().lower()
            if label and label not in seen:
                seen.add(label)
                bank.append(label)
    return bank


def get_concepts(clip, images=None):
    """Return the concept list for the target class, cached to JSON.

    In "shared" mode (default) returns the fixed shared bank — same for every class.
    In "per_class" mode returns exactly r concepts via the two-stage lexical+CLIP filter.
    `images`: optional class images used to build the CLIP relevance prototype for
    per-class stage-2 ranking (ignored in shared mode).
    """
    path = os.path.join(CONFIG["cache_dir"], cache_name("concepts", ".json", "con"))
    if os.path.exists(path):
        with open(path) as f:
            print("[cache] loaded concepts.json")
            return json.load(f)

    if CONFIG["concept_mode"] == "shared":
        concepts = _shared_bank()
        with open(path, "w") as f:
            json.dump(concepts, f, indent=2)
        print(f"[cache] saved concepts.json (shared bank: {len(concepts)} concepts)")
        return concepts

    cls, r = CONFIG["class_name"], CONFIG["r"]
    raw = _load_vocab(cls)                                  # stored vocabulary (over-provided)
    lexical = _lexical_filter(raw, cls)                     # stage 1 (suppl. A1.3)
    concepts = _clip_select(lexical, clip, images,         # stage 2 (suppl. A1.4)
                            CONFIG["dedup_threshold"], r)
    if len(concepts) < r:
        raise RuntimeError(f"Only {len(concepts)}/{r} concepts survived filtering; "
                           f"add more candidates for '{cls}' in {CONFIG['concept_vocab_path']}.")

    with open(path, "w") as f:
        json.dump(concepts, f, indent=2)
    print(f"[cache] saved concepts.json ({len(concepts)} concepts)")
    return concepts
