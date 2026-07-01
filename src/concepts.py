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
    cosine similarity below CONFIG["dedup_threshold"]. Result: at most r concepts
    (fewer if dedup collapses the pool — get_concepts then flags the short class).
"""

import hashlib
import json
import os
import re

from config import CONFIG, cache_name, resolve_vocab_key


class InsufficientConcepts(RuntimeError):
    """Raised when fewer than r concepts survive per-class filtering.

    Subclasses RuntimeError so runner.run_all's ``except Exception`` skips the class
    rather than aborting the whole run. Recover by adding candidates for the class
    (``add_concepts``) and re-evaluating.
    """


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
    key = resolve_vocab_key(cls, table.keys())   # exact canonical match, then aliases
    if key is None:
        raise KeyError(
            f"No concept vocabulary for class '{cls}' in {path}. Available keys: "
            f"{sorted(table)}. If the folder name differs from its key (e.g. AMD / "
            f"normal variants), add it to CONCEPT_ALIASES in config.py."
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

    # Greedy diverse selection, no backfill: near-duplicates are dropped outright. If the
    # diversity dedup leaves fewer than r, we return short on purpose so get_concepts raises
    # InsufficientConcepts — run_all then skips the class and the recovery cell (add_concepts)
    # supplies more distinct candidates for it.
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
    if not concepts or len(concepts) < r:
        short = len(concepts)
        msg = (f"only {short}/{r} concepts survived filtering for '{cls}' "
               f"(lexical pool was {len(lexical)})")
        # An empty pool is always fatal. A short-but-nonempty pool is skippable: when
        # concept_skip_if_short is set (default), raise so a multi-class run records the
        # class as a failure and moves on — then recover via add_concepts + re-evaluate.
        if short == 0 or CONFIG.get("concept_skip_if_short", True):
            raise InsufficientConcepts(
                f"{msg[0].upper()}{msg[1:]}. Add more candidates for '{cls}' in "
                f"{CONFIG['concept_vocab_path']} (e.g. concepts.add_concepts('{cls}', [...])) "
                f"and re-evaluate.")
        print(f"[warn] {msg}; proceeding with {short} "
              f"(set CONFIG['concept_skip_if_short']=True to skip & recover instead).")

    with open(path, "w") as f:
        json.dump(concepts, f, indent=2)
    print(f"[cache] saved concepts.json ({len(concepts)} concepts)")
    return concepts


def add_concepts(cls, new_concepts, vocab_path=None):
    """Append candidate concepts for `cls` to the vocab table and refresh the cache key.

    Recovery path for a class skipped with InsufficientConcepts: supply extra 2-3 word
    clinical phrases (distinct enough to survive the CLIP dedup at CONFIG["dedup_threshold"]),
    then re-run just that class with runner.run_all(['<cls>']).

    New concepts are lowercased and de-duplicated against the class's existing entry, then
    written back to concept_vocab.json (the entry is created if the class is new). The vocab
    content hash in CONFIG is recomputed so every concept-keyed cache (concepts.json, S, W,
    metrics) rebuilds on the next run instead of serving the stale, too-short result.
    Returns the class's full updated candidate list.
    """
    path = vocab_path or CONFIG["concept_vocab_path"]
    with open(path) as f:
        table = json.load(f)

    key = resolve_vocab_key(cls, table.keys()) or _canonical_key(cls)
    existing = list(table.get(key, []))
    seen = {(c["label"] if isinstance(c, dict) else c).strip().lower() for c in existing}

    added = []
    for c in new_concepts:
        label = (c["label"] if isinstance(c, dict) else c).strip().lower()
        if label and label not in seen:
            seen.add(label)
            existing.append(label)
            added.append(label)
    table[key] = existing

    with open(path, "wb") as f:
        blob = json.dumps(table, indent=2, ensure_ascii=False).encode()
        f.write(blob)

    # Recompute the content hash of the file we just wrote (cache_name's "con" group
    # depends on it) so concept caches invalidate. Hashing `blob` directly — rather than
    # config._vocab_hash(), which always reads the module-level default path — keeps this
    # correct even when CONFIG["concept_vocab_path"] points elsewhere.
    CONFIG["concept_vocab_hash"] = hashlib.md5(blob).hexdigest()[:8]
    print(f"[vocab] '{key}': +{len(added)} new concept(s) (now {len(existing)} candidates); "
          f"cache key refreshed. Re-evaluate with runner.run_all(['{cls}']).")
    if not added:
        print("[vocab] note: nothing added — all supplied concepts were already present.")
    return existing
