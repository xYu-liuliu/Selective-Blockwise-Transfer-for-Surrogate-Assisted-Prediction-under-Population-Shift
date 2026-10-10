"""Prepare the CheXpert-Plus pneumothorax cohort and its surrogate blocks.

Data preparation of "Selective Blockwise Transfer for Surrogate-Assisted
Prediction under Population Shift". CheXpert-Plus links each CheXpert
radiograph to its de-identified report, split into sections, and to
demographic and administrative fields.

Labels
------
The endpoint is pneumothorax. Labels come from the original CheXpert
``train.csv`` rather than from the label columns shipped with CheXpert-Plus,
mapped positive-versus-rest: canonical 1 is a case, and 0, -1 and blank are
controls. ``row_audit.csv`` keeps each row's unmodified canonical state.

Cohort and domains
------------------
An image enters if it is a frontal view, its narrative text is nonempty after
cleaning, and it decodes under both OpenCV and Pillow. White is the source
domain and Asian the target. Race enters nowhere else: neither surrogate block
carries race, ethnicity, vital status or a split indicator, and the metadata
column names are checked for them.

Source split
------------
Made once, at the patient level. ``N_SL`` = 1200 distinct source patients,
one image each, form SL; the other images of those patients are discarded;
all images of the remaining source patients form SU. No patient is in both,
so the density ratio, fitted on SU against TU and applied to SL, never scores
a patient it was fitted on. The target rows are written as one TARGET block,
from which the pipeline redraws TL, VALID and TU in every repeat.

Images
------
Each image is resized once to 224 x 224 by area interpolation and stored as
single-channel uint8, with no histogram equalisation or intensity
normalisation, because each image branch applies its own. BioViL reads the
released file instead of this array.

Narrative surrogate
-------------------
The History and Clinical History sections, coalesced. Both are written before
the radiograph is read; the Findings and Impression sections record the read
itself and are never used (the original CheXpert labeler also extracts its
labels from the Impression). Text processing normalises Unicode (NFKC),
lowercases, strips quoting artefacts and the underscore runs left by
de-identification, and collapses whitespace. Slashes and hyphens survive the
cleaning, because s/p, r/o and w/o mean something in this register. The token
pattern keeps those slash abbreviations whole, where the scikit-learn default
(runs of two or more word characters) would drop them; hyphenated words still
split into their parts.

Unigrams and bigrams in at least ``BOW_MIN_DF`` = 10 documents of the pooled
corpus are the candidates, and counts enter as log(1 + count). Three rules
then cut the candidate set, none of them reading an outcome:

1. a term must clear a document-frequency floor of 1% in both domains;
2. boilerplate goes: any term containing a bare numeral or a token that
   names age or sex (age and sex being already in the structured block), and
   any term made only of function words;
3. a unigram goes when a retained bigram containing it occurs in at least
   0.9 of its documents, since near-duplicate columns let the penalty split a
   coefficient between them arbitrarily.

Negation and uncertainty markers are exempt from the last two. A bag of words
tells an affirmative mention from a ruled-out one only through "no",
"without", "r/o" and their kin, which rule 2 would take for function words and
rule 3 would delete, "no" occurring almost only inside the phrase it negates;
their survival is checked at the end of the build. Nothing rotates or reduces
the block, so every coefficient of theta_S reads back to its term, and
``vocabulary_filter_audit.csv`` records which rule removed each discarded
candidate. The vocabulary is fixed once on the pooled cohort from document
frequencies, so it is transductive but label-free.

Why the floor is two-sided
--------------------------
delta is estimated on the nT rows of TL alone. A term rare in the target is
absent from those rows often enough (at a target prevalence of 0.005 and
nT = 200, with probability about 0.37) that delta has no data with which to
remove a coefficient theta_tilde put on it. A term frequent in one domain only
fails the same way, and is also what makes the two domains easy to tell apart.

Structured fields
-----------------
Age and body mass index, each standardised with a missingness indicator and
thresholds (65 and 75; 30 and 35), sex, whether an interpreter was required,
and payer category, the last three as indicators with an explicit unknown
level. Constant columns are dropped here (two of sixteen in the paper's
cohort, leaving 14), and the surviving count goes to the manifest so the
pipeline never hard-codes it.

Outputs
-------
``SL.npz``      X_image_uint8, S_text_bow, S_meta, Y, prepared_row_index
``SU.npz``      X_image_uint8, S_text_bow, S_meta, prepared_row_index
``TARGET.npz``  X_image_uint8, S_text_bow, S_meta, Y, prepared_row_index
``surrogate_vocabulary.csv``, ``vocabulary_filter_audit.csv``,
``metadata_feature_names.csv``,
``row_audit.csv``, ``prepare_config.json``

Inputs are read from, and outputs written to, the locations in
``project_paths.py``.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
import re
import unicodedata
import warnings
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

from project_paths import CHEXPERT_DIR, CHEXPERT_PLUS_CSV, PREP_DIR

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

# Locations come from project_paths.py (or the STRIFLE_* environment variables).
CSV_PATH = CHEXPERT_PLUS_CSV          # CheXpert-Plus table
IMAGE_BASE = CHEXPERT_DIR             # downsampled CheXpert: train.csv + train/
CANONICAL_LABELS_PATH = os.path.join(IMAGE_BASE, "train.csv")
CANONICAL_PATH_COLUMN = "Path"

SAVE_DIR = PREP_DIR

TARGET_LABEL = "Pneumothorax"
SOURCE_RACE = "White"
TARGET_RACE = "Asian"

N_SL = 1_200
RANDOM_SEED = 42

SOURCE_ALLOWED_CANONICAL_LABELS = (1, 0, -1)
TARGET_ALLOWED_CANONICAL_LABELS = (1, 0, -1)
INCLUDE_CANONICAL_BLANK_AS_ZERO = True

IMG_SIZE = 224
IMAGE_INTERPOLATION = cv2.INTER_AREA
PAPER_IMAGE_MEAN = (0.485, 0.485, 0.485)
PAPER_IMAGE_STD = (0.286, 0.286, 0.286)

HISTORY_TEXT_FIELD = "section_history"
CLINICAL_HISTORY_TEXT_FIELD = "section_clinical_history"

# Bag-of-words surrogate. BOW_MIN_DF only bounds the candidate vocabulary; the
# operative filter is rule 1, the two-sided prevalence floor below (see the
# module docstring for why it is two-sided).
BOW_TOKEN_PATTERN = r"(?u)\b[\w/]+\b"
BOW_NGRAM_RANGE = (1, 2)
BOW_MIN_DF = 10
BOW_TARGET_PREVALENCE_MIN = 0.01
BOW_SOURCE_PREVALENCE_MIN = 0.01
BOW_COUNT_TRANSFORM = "log1p"   # counts enter as log(1 + count)

# ---------------------------------------------------------------------------
# Rules 2 and 3 of the vocabulary filter, pre-specified and label-free
# ---------------------------------------------------------------------------
# Rule 2, function words: a term made up entirely of these carries no
# clinical content. Negation and uncertainty markers are deliberately absent
# from this list: "no", "not", "without", "rule", "out", "negative",
# "denies", "possible", "concern", "evaluate" and the slash abbreviations r/o
# and w/o are the only way a bag of words can separate an affirmative mention
# from a ruled-out one.
BOW_STOPWORDS = frozenset("""
a an the of in on at to for with and or is are was were be been being am
this that these those it its as by from into onto per which who whom whose
""".split())

# Rule 2, age and sex: a term containing one of these tokens restates a
# structured field. Age and sex already enter S as meta_age_scaled,
# meta_age_ge_65, meta_age_ge_75 and meta_sex_male; admitting them again as
# text puts the same variable twice into theta_S, which lets the Lasso split
# the coefficient arbitrarily between near-duplicate columns and duplicates
# the same covariate shift inside the density-ratio design.
BOW_DEMOGRAPHIC_TOKENS = frozenset("""
year years yr yrs yo old aged age male female man woman men women
gentleman lady boy girl mr mrs ms
""".split())

# Negation and uncertainty markers, exempt from rules 2 and 3. They are the
# only way a bag of words can tell an affirmative mention from a ruled-out or
# merely suspected one. Rule 2 would take them for function words, and rule 3
# would delete them: "no" and "out" occur almost only inside "no
# pneumothorax" and "rule out", which is exactly the pattern it removes.
BOW_PROTECTED_TOKENS = frozenset("""
no not without negative denies resolved
r/o w/o rule out ruled possible possibly question questionable concern
concerning evaluate eval assess suspect suspected versus vs
""".split())

# Rule 2, numerals: a term containing a bare numeral token goes as a whole.
# "65 years" is caught by the age tokens, but the unigram "65" is the same age
# fragment and duplicates meta_age_ge_65.
BOW_NUMERIC_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")

# Rule 3: a unigram that appears almost only inside a retained bigram carries
# no information the bigram does not already have; keeping both leaves the
# Lasso free to split the coefficient between two near-identical columns, so
# the surviving term differs from repeat to repeat and the coefficient table
# becomes unreadable. Purely a document-frequency rule, no outcome involved.
BOW_DROP_BOILERPLATE = True
BOW_DROP_SUBSUMED_UNIGRAMS = True
BOW_SUBSUMPTION_THRESHOLD = 0.9

# The smaller of the paper's two arms (NT = 200 and 400). Only recorded in the
# manifest, so the prevalence floor can be audited against the labelled target
# sample it protects.
NT_REFERENCE = 200

SAVE_MMAP_IMAGE_NPY_COPY = False

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# =============================================================================
# General helpers
# =============================================================================

def clean_string_col(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def robust_read_csv(path: str) -> pd.DataFrame:
    """Read a CSV containing occasionally malformed free-text quotation."""
    attempts = [
        {"engine": "c", "encoding": "utf-8-sig", "low_memory": False},
        {"engine": "python", "encoding": "utf-8-sig"},
        {"engine": "python", "encoding": "utf-8-sig", "quoting": csv.QUOTE_NONE},
    ]
    errors = []
    for kwargs in attempts:
        try:
            print(f"Reading CSV with {kwargs}")
            return pd.read_csv(path, **kwargs)
        except Exception as exc:  # pragma: no cover - data-dependent fallback
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
    raise RuntimeError("All CSV parsing strategies failed:\n" + "\n".join(errors))


def normalize_canonical_chexpert_path(values: Iterable) -> pd.Series:
    """Normalize original CheXpert paths to ``train/patient...`` paths."""
    return (
        pd.Series(values, dtype="string")
        .fillna("")
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.replace(r"/+", "/", regex=True)
        .str.extract(r"(train/.*)$", expand=False)
    )


def load_canonical_labels(labels_path: str) -> pd.DataFrame:
    """Load and validate original CheXpert image-level canonical labels."""
    labels = robust_read_csv(labels_path)
    required = {CANONICAL_PATH_COLUMN, TARGET_LABEL}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise KeyError(
            f"Canonical CheXpert CSV is missing columns {missing}; "
            f"available columns={labels.columns.tolist()}"
        )
    if len(labels) < 200_000:
        raise ValueError(
            f"Canonical label CSV has only {len(labels):,} rows. "
            "Expected the original CheXpert train.csv, not valid.csv."
        )

    result = labels[[CANONICAL_PATH_COLUMN, TARGET_LABEL]].copy()
    result["path_to_image"] = normalize_canonical_chexpert_path(
        result[CANONICAL_PATH_COLUMN]
    )
    bad_paths = result["path_to_image"].isna() | result["path_to_image"].eq("")
    if bad_paths.any():
        examples = result.loc[bad_paths, CANONICAL_PATH_COLUMN].head(10).tolist()
        raise ValueError(
            f"Could not normalize canonical CheXpert paths; examples={examples}"
        )

    duplicate_paths = result["path_to_image"].duplicated(keep=False)
    if duplicate_paths.any():
        examples = result.loc[duplicate_paths, "path_to_image"].head(10).tolist()
        raise ValueError(
            f"Duplicate normalized paths in canonical train.csv; examples={examples}"
        )

    raw_label_name = f"{TARGET_LABEL}_canonical_raw"
    result.rename(columns={TARGET_LABEL: raw_label_name}, inplace=True)
    raw_values = result[raw_label_name]
    parsed_values = pd.to_numeric(raw_values, errors="coerce")
    malformed = raw_values.notna() & parsed_values.isna()
    if malformed.any():
        examples = raw_values.loc[malformed].head(10).tolist()
        raise ValueError(
            "Canonical Pneumothorax labels contain nonnumeric nonblank values; "
            f"examples={examples}"
        )
    return result[["path_to_image", raw_label_name]]


def positive_vs_rest_label(values: Iterable) -> np.ndarray:
    """Map canonical 1 to 1 and 0/-1/blank/no-mention to 0."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
    return (numeric == 1).astype(np.float32)


def canonical_label_state(values: Iterable) -> pd.Categorical:
    """Return auditable names for canonical 1/0/-1/blank label states."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    states = np.select(
        [numeric.eq(1), numeric.eq(0), numeric.eq(-1)],
        ["positive_1", "explicit_negative_0", "uncertain_-1"],
        default="no_mention_blank",
    )
    return pd.Categorical(
        states,
        categories=[
            "positive_1", "explicit_negative_0", "uncertain_-1", "no_mention_blank",
        ],
    )


def select_history_text(history, clinical_history) -> tuple[str, str]:
    """Coalesce History and Clinical History; both are pre-imaging text."""
    history = "" if pd.isna(history) else str(history).strip()
    clinical_history = "" if pd.isna(clinical_history) else str(clinical_history).strip()
    if history and clinical_history:
        return f"{history} {clinical_history}", "section_history+section_clinical_history"
    if history:
        return history, HISTORY_TEXT_FIELD
    if clinical_history:
        return clinical_history, CLINICAL_HISTORY_TEXT_FIELD
    return "", "missing_history"


def paper_clean_text(text) -> str:
    """The paper's text processing: NFKC, lowercase, quoting artefacts and
    de-identification underscore runs removed, whitespace collapsed."""
    text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Quote/backslash removal only. Slashes and hyphens are preserved because
    # s/p, r/o and w/o mean something in this register.
    text = re.sub(r'["\\]+', " ", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"_{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ordered_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8", errors="replace"))
        digest.update(b"\n")
    return digest.hexdigest()


def installed_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "unknown"


def parse_patient_id(paths: Iterable) -> pd.Series:
    ids = (
        pd.Series(paths, dtype="string")
        .astype(str)
        .str.extract(r"(?i)(patient[_-]?\d+)", expand=False)
        .str.lower()
    )
    if ids.isna().any():
        bad = pd.Series(paths).loc[ids.isna()].head(10).tolist()
        raise ValueError(f"Could not parse patient ID from paths: {bad}")
    return ids


# =============================================================================
# Image helpers
# =============================================================================

def resolve_image_path(path_to_image) -> str | None:
    """Resolve a relative path under IMAGE_BASE.

    Every candidate is checked to actually end with the requested relative
    path. Without that check a short suffix candidate could match a different
    patient's file and silently misalign an image with its label.
    """
    raw = str(path_to_image)
    normalized = raw.replace("\\", "/")
    candidates = [
        raw,
        os.path.join(IMAGE_BASE, raw),
        os.path.join(IMAGE_BASE, raw.lstrip("/\\")),
    ]
    parts = Path(raw).parts
    candidates.extend(os.path.join(IMAGE_BASE, *parts[i:]) for i in range(len(parts)))

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        if not str(candidate).replace("\\", "/").endswith(normalized):
            continue
        return candidate
    return None


def validate_image_file(path: str) -> tuple[bool, str]:
    """Require both OpenCV and full Pillow pixel decoding to succeed."""
    image_cv = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image_cv is None:
        return False, "OpenCV could not decode the image"
    try:
        with Image.open(path) as image_pil:
            image_pil.verify()
        with Image.open(path) as image_pil:
            image_pil.load()
            _ = np.asarray(image_pil)
    except Exception as exc:
        return False, f"Pillow {type(exc).__name__}: {exc}"
    return True, ""


def load_images_uint8(
    frame: pd.DataFrame, row_indices: np.ndarray, split_name: str
) -> np.ndarray:
    """Resize one split to N x 224 x 224 uint8 by area interpolation, with no
    histogram equalisation (CLAHE) or intensity normalisation, because each
    image branch applies its own."""
    row_indices = np.asarray(row_indices, dtype=np.int64)
    output = np.empty((len(row_indices), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for position, row_index in enumerate(row_indices):
        if position % 2000 == 0:
            print(f"  Loading {split_name} image {position}/{len(row_indices)}")
        path = frame.iloc[int(row_index)]["resolved_image_path"]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(
                f"Image failed after validation: split={split_name}, "
                f"row_index={row_index}, path={path}"
            )
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=IMAGE_INTERPOLATION)
        output[position] = image.astype(np.uint8, copy=False)
    return output


# =============================================================================
# Bag-of-words surrogate
# =============================================================================

def is_boilerplate_term(term: str) -> bool:
    """Rule 2: true when a term carries no clinical content of its own.

    Three cases: any term containing a bare numeral token ("65", an age
    fragment duplicating meta_age_ge_65, or "day 3"), any term containing a
    demographic token ("68 years old male"), and terms built entirely from
    function words ("of the"). A term containing a negation or uncertainty
    marker is never boilerplate, so "no", "rule out" and "r/o" all survive.
    """
    tokens = term.split()
    if any(t in BOW_PROTECTED_TOKENS for t in tokens):
        return False
    if any(BOW_NUMERIC_PATTERN.match(t) for t in tokens):
        return True
    if any(t in BOW_DEMOGRAPHIC_TOKENS for t in tokens):
        return True
    return all(t in BOW_STOPWORDS for t in tokens)


def drop_subsumed_unigrams(vocabulary, document_frequency, keep, threshold):
    """Rule 3: remove a unigram when a retained bigram containing it occurs in
    at least ``threshold`` of its documents.

    "shortness of breath" generates shortness, breath, shortness of and of
    breath. They are near-duplicate columns, so the Lasso splits a coefficient
    between them arbitrarily and the selected term changes from repeat to
    repeat, which makes the coefficient table unreadable. The rule compares
    document frequencies only, so no outcome is involved.

    Protected tokens are exempt. A negation marker naturally occurs almost
    only inside the phrase it negates, so this rule would delete precisely the
    markers the surrogate cannot afford to lose.
    """
    keep = np.asarray(keep, dtype=bool).copy()
    subsumed = np.zeros(len(vocabulary), dtype=bool)
    kept_indices = np.flatnonzero(keep)
    unigram_index = {
        vocabulary[j]: j for j in kept_indices if " " not in vocabulary[j]
    }
    bigram_indices = [j for j in kept_indices if " " in vocabulary[j]]

    for j in bigram_indices:
        for token in vocabulary[j].split():
            if token in BOW_PROTECTED_TOKENS:
                continue
            i = unigram_index.get(token)
            if i is None or not keep[i] or document_frequency[i] <= 0:
                continue
            if document_frequency[j] / document_frequency[i] >= threshold:
                keep[i] = False
                subsumed[i] = True
    return keep, subsumed


def build_bow_surrogate(
    texts: Sequence[str], is_source: np.ndarray, is_target: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Candidate unigrams and bigrams, the three rules, log(1 + count).

    Candidates are the terms in at least BOW_MIN_DF documents of the pooled
    corpus. Rule 1 keeps a term only if its document frequency reaches the
    floor in both domains, rule 2 drops boilerplate and rule 3 subsumed
    unigrams; negation and uncertainty markers are exempt from rules 2 and 3.
    No outcome label is read, so the vocabulary is fixed once on the pooled
    cohort: transductive but label-free.
    """
    vectorizer = CountVectorizer(
        ngram_range=BOW_NGRAM_RANGE,
        min_df=BOW_MIN_DF,
        token_pattern=BOW_TOKEN_PATTERN,
        lowercase=False,      # paper_clean_text already lowercased
        dtype=np.int32,
    )
    counts = vectorizer.fit_transform(list(texts))
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    print(f"  candidate vocabulary at min_df={BOW_MIN_DF}: {len(vocabulary):,} terms")

    present = counts > 0
    prevalence_source = np.asarray(present[is_source].mean(axis=0)).ravel()
    prevalence_target = np.asarray(present[is_target].mean(axis=0)).ravel()
    prevalence_all = np.asarray(present.mean(axis=0)).ravel()
    document_frequency = np.asarray(present.sum(axis=0)).ravel()

    passes_prevalence = (
        (prevalence_target >= BOW_TARGET_PREVALENCE_MIN)
        & (prevalence_source >= BOW_SOURCE_PREVALENCE_MIN)
    )
    boilerplate = np.array(
        [is_boilerplate_term(t) for t in vocabulary], dtype=bool
    ) if BOW_DROP_BOILERPLATE else np.zeros(len(vocabulary), dtype=bool)

    keep = passes_prevalence & ~boilerplate
    subsumed = np.zeros(len(vocabulary), dtype=bool)
    if BOW_DROP_SUBSUMED_UNIGRAMS:
        keep, subsumed = drop_subsumed_unigrams(
            vocabulary, document_frequency, keep, BOW_SUBSUMPTION_THRESHOLD
        )

    if keep.sum() < 10:
        raise ValueError(
            f"Only {int(keep.sum())} terms survive the filters. Lower "
            "BOW_TARGET_PREVALENCE_MIN / BOW_SOURCE_PREVALENCE_MIN or disable "
            "BOW_DROP_SUBSUMED_UNIGRAMS."
        )

    # Full audit trail for every candidate, so the paper can state exactly
    # which rule removed which term and reviewers can verify that no negation
    # or uncertainty marker was discarded.
    reason = np.where(
        keep, "kept",
        np.where(boilerplate, "boilerplate",
                 np.where(~passes_prevalence, "below_prevalence_floor",
                          np.where(subsumed, "subsumed_by_bigram", "unknown"))),
    )
    pd.DataFrame({
        "term": vocabulary,
        "document_frequency": document_frequency,
        "prevalence_all": prevalence_all,
        "prevalence_source": prevalence_source,
        "prevalence_target": prevalence_target,
        "expected_TL_support": prevalence_target * NT_REFERENCE,
        "boilerplate": boilerplate,
        "passes_prevalence": passes_prevalence,
        "subsumed_by_bigram": subsumed,
        "kept": keep,
        "exclusion_reason": reason,
    }).sort_values(
        ["kept", "prevalence_all"], ascending=[False, False]
    ).to_csv(os.path.join(SAVE_DIR, "vocabulary_filter_audit.csv"), index=False)

    kept_counts = counts[:, keep]
    dense = np.asarray(kept_counts.todense(), dtype=np.float32)
    if BOW_COUNT_TRANSFORM == "log1p":
        features = np.log1p(dense).astype(np.float32)
    elif BOW_COUNT_TRANSFORM == "binary":
        features = (dense > 0).astype(np.float32)
    else:
        features = dense
    del dense, kept_counts
    gc.collect()

    if not np.isfinite(features).all():
        raise ValueError("Non-finite value in the bag-of-words surrogate.")

    spec = pd.DataFrame({
        "term": vocabulary[keep],
        "prevalence_all": prevalence_all[keep],
        "prevalence_source": prevalence_source[keep],
        "prevalence_target": prevalence_target[keep],
        "expected_TL_support": prevalence_target[keep] * NT_REFERENCE,
    }).sort_values("prevalence_all", ascending=False).reset_index(drop=True)

    dropped_target_only = int(
        ((prevalence_target >= BOW_TARGET_PREVALENCE_MIN)
         & (prevalence_source < BOW_SOURCE_PREVALENCE_MIN)).sum()
    )
    dropped_source_only = int(
        ((prevalence_source >= BOW_SOURCE_PREVALENCE_MIN)
         & (prevalence_target < BOW_TARGET_PREVALENCE_MIN)).sum()
    )
    thin_support = int((spec["expected_TL_support"] < 5).sum())
    info = {
        "candidate_vocabulary": int(len(vocabulary)),
        "kept_vocabulary": int(keep.sum()),
        "dropped_boilerplate": int((boilerplate & passes_prevalence).sum()),
        "dropped_subsumed_unigrams": int(subsumed.sum()),
        "dropped_frequent_in_target_only": dropped_target_only,
        "dropped_frequent_in_source_only": dropped_source_only,
        "token_pattern": BOW_TOKEN_PATTERN,
        "ngram_range": list(BOW_NGRAM_RANGE),
        "min_df": BOW_MIN_DF,
        "target_prevalence_min": BOW_TARGET_PREVALENCE_MIN,
        "source_prevalence_min": BOW_SOURCE_PREVALENCE_MIN,
        "boilerplate_filter": BOW_DROP_BOILERPLATE,
        "subsumed_unigram_filter": BOW_DROP_SUBSUMED_UNIGRAMS,
        "subsumption_threshold": BOW_SUBSUMPTION_THRESHOLD,
        "count_transform": BOW_COUNT_TRANSFORM,
        "nT_reference": NT_REFERENCE,
        "min_expected_TL_support": float(spec["expected_TL_support"].min()),
        "median_expected_TL_support": float(spec["expected_TL_support"].median()),
        "n_terms_with_TL_support_below_5": thin_support,
        "frac_terms_with_TL_support_below_5": float(thin_support / len(spec)),
        "fitted_with_y": False,
        "sklearn_version": installed_version("scikit-learn"),
    }

    print(
        f"  prevalence floor    : {int(passes_prevalence.sum()):,} pass "
        f"(dropped {dropped_source_only:,} frequent only in source, "
        f"{dropped_target_only:,} frequent only in target)"
    )
    print(
        f"  boilerplate filter  : -{info['dropped_boilerplate']:,} "
        "(function words, age/sex restatements, bare numerals)"
    )
    print(f"  subsumed unigrams   : -{info['dropped_subsumed_unigrams']:,}")
    print(f"  kept                : {info['kept_vocabulary']:,} clinical terms")
    print(
        f"  TL support: min={info['min_expected_TL_support']:.1f}, "
        f"median={info['median_expected_TL_support']:.1f}, "
        f"below 5 = {thin_support}/{len(spec)} "
        f"({info['frac_terms_with_TL_support_below_5']:.1%})"
    )

    # The negation and uncertainty markers are the only way a bag of words can
    # tell an affirmative mention from a ruled-out one, so their survival is
    # checked explicitly rather than left to inspection.
    marker_status = {}
    kept_terms = set(vocabulary[keep].tolist())
    joined = " ".join(sorted(kept_terms))
    for marker in ("no", "not", "without", "w/o", "r/o", "rule", "out",
                   "rule out", "negative", "denies", "possible", "concern",
                   "evaluate", "eval"):
        in_corpus = bool(document_frequency[vocabulary == marker].sum()) \
            if marker in set(vocabulary.tolist()) else False
        marker_status[marker] = (
            "kept" if marker in kept_terms
            else ("absent_from_candidates" if not in_corpus else "DROPPED")
        )
    dropped_markers = [m for m, s in marker_status.items() if s == "DROPPED"]
    info["negation_marker_status"] = marker_status
    if dropped_markers:
        print(
            f"  [WARN] negation/uncertainty markers removed: {dropped_markers}. "
            "They are how the bag of words tells an affirmative mention from a "
            "ruled-out one; check BOW_STOPWORDS and the prevalence floor before "
            "continuing."
        )
    else:
        print("  negation/uncertainty markers: all survivors or absent from corpus")
    return features, spec, info


# =============================================================================
# Metadata block
# =============================================================================

MISSING_CATEGORY_TOKENS = {
    "", "unknown", "unk", "nan", "none", "null", "not available", "n/a",
}


def normalize_category(series: pd.Series) -> pd.Series:
    return (
        series.fillna("Unknown").astype(str).str.strip().str.lower()
        .replace({"": "unknown", "nan": "unknown", "none": "unknown"})
    )


def standardize_numeric_with_missing(series: pd.Series):
    """Median-impute then z-score.

    The downstream pipeline re-standardises every metadata column on the
    repeat-specific reference pool, so the affine part of this transform is
    undone there. Only the imputation constant persists, and it is a
    label-free cohort median.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    missing = numeric.isna()
    if numeric.notna().any():
        median = float(numeric.median())
        filled = numeric.fillna(median)
        mean = float(filled.mean())
        std = float(filled.std())
        if np.isfinite(std) and std >= 1e-6:
            scaled = ((filled - mean) / std).astype(np.float32)
        else:
            scaled = pd.Series(0.0, index=numeric.index, dtype=np.float32)
    else:
        median = 0.0
        scaled = pd.Series(0.0, index=numeric.index, dtype=np.float32)
    return numeric, scaled, missing.astype(np.float32), median


def build_metadata_surrogate(frame: pd.DataFrame):
    """The structured-field block, with constant columns dropped.

    Sixteen columns: age and body mass index (median-imputed z-score,
    missingness indicator, thresholds at 65 and 75, and at 30 and 35), sex,
    interpreter required and payer category, the last three as indicators
    against a reference level with an explicit unknown level. Constant
    columns would leave nothing to standardise downstream, so they are removed
    here (two in the paper's cohort) and the surviving count goes to the
    manifest. No column may name race, ethnicity, vital status or a split.
    """
    meta = pd.DataFrame(index=frame.index)
    encodings = {}

    age_col = "age" if "age" in frame.columns else (
        "patient_age" if "patient_age" in frame.columns else None
    )
    age_input = (
        frame[age_col] if age_col
        else pd.Series(np.nan, index=frame.index, dtype=np.float32)
    )
    age_raw, age_scaled, age_missing, _ = standardize_numeric_with_missing(age_input)
    age_valid = age_raw.notna()
    meta["meta_age_scaled"] = age_scaled
    meta["meta_age_missing"] = age_missing
    meta["meta_age_ge_65"] = (age_valid & (age_raw >= 65)).astype(np.float32)
    meta["meta_age_ge_75"] = (age_valid & (age_raw >= 75)).astype(np.float32)

    bmi_col = next(
        (n for n in ["recent_bmi", "bmi", "BMI", "patient_bmi"] if n in frame.columns),
        None,
    )
    bmi_input = (
        frame[bmi_col] if bmi_col
        else pd.Series(np.nan, index=frame.index, dtype=np.float32)
    )
    bmi_raw, bmi_scaled, bmi_missing, _ = standardize_numeric_with_missing(bmi_input)
    bmi_valid = bmi_raw.notna()
    meta["meta_recent_bmi_scaled"] = bmi_scaled
    meta["meta_recent_bmi_missing"] = bmi_missing
    meta["meta_recent_bmi_ge_30"] = (bmi_valid & (bmi_raw >= 30)).astype(np.float32)
    meta["meta_recent_bmi_ge_35"] = (bmi_valid & (bmi_raw >= 35)).astype(np.float32)

    sex_col = "sex" if "sex" in frame.columns else None
    sex = normalize_category(
        frame[sex_col] if sex_col else pd.Series("unknown", index=frame.index)
    )
    male = sex.isin({"male", "m"})
    female = sex.isin({"female", "f"})
    meta["meta_sex_male"] = male.astype(np.float32)
    meta["meta_sex_unknown"] = (~(male | female)).astype(np.float32)

    interpreter_col = (
        "interpreter_needed" if "interpreter_needed" in frame.columns else None
    )
    interpreter = normalize_category(
        frame[interpreter_col] if interpreter_col
        else pd.Series("unknown", index=frame.index)
    )
    interpreter_yes = interpreter.isin({"yes", "y", "true", "1"})
    interpreter_no = interpreter.isin({"no", "n", "false", "0"})
    meta["meta_interpreter_needed_yes"] = interpreter_yes.astype(np.float32)
    meta["meta_interpreter_needed_unknown"] = (
        ~(interpreter_yes | interpreter_no)
    ).astype(np.float32)

    insurance_col = "insurance_type" if "insurance_type" in frame.columns else None
    insurance = normalize_category(
        frame[insurance_col] if insurance_col
        else pd.Series("unknown", index=frame.index)
    )
    insurance_unknown = insurance.isin(MISSING_CATEGORY_TOKENS)
    insurance_private = insurance.str.contains("private", regex=False) & ~insurance_unknown
    insurance_medicare = insurance.str.contains("medicare", regex=False) & ~insurance_unknown
    insurance_medicaid = insurance.str.contains("medicaid", regex=False) & ~insurance_unknown
    insurance_other = ~(
        insurance_unknown | insurance_private | insurance_medicare | insurance_medicaid
    )
    meta["meta_insurance_medicare"] = insurance_medicare.astype(np.float32)
    meta["meta_insurance_medicaid"] = insurance_medicaid.astype(np.float32)
    meta["meta_insurance_other"] = insurance_other.astype(np.float32)
    meta["meta_insurance_unknown"] = insurance_unknown.astype(np.float32)

    encodings = {
        "meta_age_scaled": (age_col or "missing", "median imputation; cohort z-score"),
        "meta_age_missing": (age_col or "missing", "1 if missing"),
        "meta_age_ge_65": (age_col or "missing", "1 if observed age >= 65"),
        "meta_age_ge_75": (age_col or "missing", "1 if observed age >= 75"),
        "meta_recent_bmi_scaled": (bmi_col or "missing", "median imputation; cohort z-score"),
        "meta_recent_bmi_missing": (bmi_col or "missing", "1 if missing"),
        "meta_recent_bmi_ge_30": (bmi_col or "missing", "1 if observed BMI >= 30"),
        "meta_recent_bmi_ge_35": (bmi_col or "missing", "1 if observed BMI >= 35"),
        "meta_sex_male": (sex_col or "missing", "Male vs Female reference"),
        "meta_sex_unknown": (sex_col or "missing", "Unknown sex vs Female reference"),
        "meta_interpreter_needed_yes": (interpreter_col or "missing", "Yes vs No reference"),
        "meta_interpreter_needed_unknown": (interpreter_col or "missing", "Unknown vs No reference"),
        "meta_insurance_medicare": (insurance_col or "missing", "Medicare vs Private reference"),
        "meta_insurance_medicaid": (insurance_col or "missing", "Medicaid vs Private reference"),
        "meta_insurance_other": (insurance_col or "missing", "Other vs Private reference"),
        "meta_insurance_unknown": (insurance_col or "missing", "Unknown vs Private reference"),
    }

    meta = meta.astype(np.float32)
    if not np.isfinite(meta.to_numpy()).all():
        raise ValueError("Non-finite metadata feature found.")

    standard_deviations = meta.std(axis=0)
    constant_columns = standard_deviations[standard_deviations < 1e-8].index.tolist()
    if constant_columns:
        print(f"  dropping constant metadata columns: {constant_columns}")
        meta = meta.drop(columns=constant_columns)
    if meta.shape[1] < 4:
        raise AssertionError(f"Only {meta.shape[1]} metadata columns survived.")

    forbidden = [
        name for name in meta.columns
        if any(tok in name.lower() for tok in ["race", "ethnicity", "deceased", "split"])
    ]
    if forbidden:
        raise AssertionError(f"Forbidden metadata columns: {forbidden}")

    spec = pd.DataFrame([
        {
            "feature": feature,
            "source_column": encodings[feature][0],
            "encoding": encodings[feature][1],
            "prevalence_or_mean": float(meta[feature].mean()),
            "sd": float(meta[feature].std()),
            "type": "metadata_always_kept",
        }
        for feature in meta.columns
    ])
    return meta.to_numpy(np.float32), meta.columns.tolist(), spec, constant_columns


# =============================================================================
# Patient-level split helper
# =============================================================================

def patient_level_source_split(
    patient_ids: np.ndarray, candidate_rows: np.ndarray, n_sl: int, rng
):
    """One image from each of n_sl distinct source patients forms SL.

    Every other image belonging to a selected patient is excluded, so no
    source patient appears in both SL and SU. This mirrors the target-side
    construction and keeps the density ratio, which is fit on SU versus TU
    and applied to SL, from scoring patients it has already seen.
    """
    candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
    table = pd.DataFrame({
        "row": candidate_rows,
        "patient": patient_ids[candidate_rows],
    })
    grouped = {pid: g["row"].to_numpy() for pid, g in table.groupby("patient", sort=False)}
    patients = np.asarray(sorted(grouped.keys()), dtype=str)
    if len(patients) < n_sl:
        raise ValueError(
            f"Only {len(patients)} source patients available for N_SL={n_sl}."
        )
    permuted = rng.permutation(patients)
    sl_patients = permuted[:n_sl]
    su_patients = permuted[n_sl:]

    sl_rows = np.array([int(rng.choice(grouped[p])) for p in sl_patients], dtype=np.int64)
    excluded = np.array(
        [r for p in sl_patients for r in grouped[p] if r not in set(sl_rows.tolist())],
        dtype=np.int64,
    )
    su_rows = np.concatenate([grouped[p] for p in su_patients]).astype(np.int64)

    assert len(sl_rows) == n_sl
    assert len(set(patient_ids[sl_rows]).intersection(set(patient_ids[su_rows]))) == 0
    assert not np.intersect1d(sl_rows, su_rows).size
    assert not np.intersect1d(sl_rows, excluded).size
    return np.sort(sl_rows), np.sort(su_rows), np.sort(excluded)


# =============================================================================
# Output helpers
# =============================================================================

def save_split_npz(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    split_name: str,
    text_all: np.ndarray,
    metadata_all: np.ndarray,
    labels_all: np.ndarray,
) -> dict:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    images = load_images_uint8(frame, row_indices, split_name)
    text = text_all[row_indices].astype(np.float32, copy=False)
    metadata = metadata_all[row_indices].astype(np.float32, copy=False)

    if not (len(images) == len(text) == len(metadata) == len(row_indices)):
        raise AssertionError(f"{split_name} modality row-count mismatch.")

    output_path = os.path.join(SAVE_DIR, f"{split_name}.npz")
    payload = {
        "X_image_uint8": images,
        "S_text_bow": text,
        "S_meta": metadata,
        "prepared_row_index": row_indices,
    }
    if split_name in {"SL", "TARGET"}:
        payload["Y"] = labels_all[row_indices].astype(np.float32, copy=False)

    print(f"Saving {split_name} NPZ: {output_path}")
    np.savez_compressed(output_path, **payload)
    if SAVE_MMAP_IMAGE_NPY_COPY:
        np.save(os.path.join(SAVE_DIR, f"X_{split_name}.npy"), images)

    result = {
        "file": os.path.basename(output_path),
        "n_rows": int(len(row_indices)),
        "image_shape": list(images.shape),
        "image_dtype": str(images.dtype),
        "text_shape": list(text.shape),
        "metadata_shape": list(metadata.shape),
        "has_y": split_name in {"SL", "TARGET"},
        "n_patients": int(frame.iloc[row_indices]["patient_id_parsed"].nunique()),
        "ordered_path_sha256": ordered_sha256(
            frame.iloc[row_indices]["path_to_image"].astype(str).tolist()
        ),
        "ordered_clean_text_sha256": ordered_sha256(
            frame.iloc[row_indices]["surrogate_text_clean"].astype(str).tolist()
        ),
    }
    del images, text, metadata, payload
    gc.collect()
    return result


# =============================================================================
# Main preparation
# =============================================================================

def main() -> None:
    print("=" * 88)
    print("Prepare canonical blank-as-zero CheXpert Pneumothorax")
    print("Coalesced History bag-of-words surrogate + 224x224 image NPZ files")
    print("=" * 88)

    missing_inputs = [p for p in (CSV_PATH, CANONICAL_LABELS_PATH)
                      if not os.path.isfile(p)]
    if missing_inputs:
        raise FileNotFoundError(
            "Input file(s) not found:\n  " + "\n  ".join(missing_inputs)
            + "\nPlace the data as described in project_paths.py, or set "
            "STRIFLE_CHEXPERT_PLUS_CSV / STRIFLE_CHEXPERT_DIR."
        )
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"CheXpert-Plus table : {CSV_PATH}")
    print(f"CheXpert images     : {IMAGE_BASE}")
    print(f"Output directory    : {SAVE_DIR}")

    frame_raw = robust_read_csv(CSV_PATH)
    label_table = load_canonical_labels(CANONICAL_LABELS_PATH)

    required_table = {"path_to_image", "frontal_lateral", "race"}
    missing_table = sorted(required_table - set(frame_raw.columns))
    if missing_table:
        raise KeyError(f"Missing CheXpert-Plus table columns: {missing_table}")

    frame = pd.merge(
        frame_raw, label_table, on="path_to_image", how="inner", validate="one_to_one"
    )
    plus_match_rate = len(frame) / len(frame_raw) if len(frame_raw) else 0.0
    canonical_match_rate = len(frame) / len(label_table) if len(label_table) else 0.0
    print(
        f"Canonical merge: {frame.shape}; CheXpert-Plus match={plus_match_rate:.4%}; "
        f"canonical match={canonical_match_rate:.4%}"
    )
    if canonical_match_rate < 0.95:
        raise ValueError(
            "Fewer than 95% of canonical CheXpert training paths matched the "
            "CheXpert-Plus table. Check path normalization before continuing."
        )

    missing_history_cols = [
        c for c in (HISTORY_TEXT_FIELD, CLINICAL_HISTORY_TEXT_FIELD)
        if c not in frame.columns
    ]
    if missing_history_cols:
        raise KeyError(
            f"Required raw History column(s) {missing_history_cols} are missing; "
            f"available columns={frame.columns.tolist()}"
        )
    frame[HISTORY_TEXT_FIELD] = clean_string_col(frame[HISTORY_TEXT_FIELD])
    frame[CLINICAL_HISTORY_TEXT_FIELD] = clean_string_col(
        frame[CLINICAL_HISTORY_TEXT_FIELD]
    )

    selected_text = [
        select_history_text(h, c)
        for h, c in zip(frame[HISTORY_TEXT_FIELD], frame[CLINICAL_HISTORY_TEXT_FIELD])
    ]
    frame["surrogate_text_raw"] = [item[0] for item in selected_text]
    frame["surrogate_text_source"] = [item[1] for item in selected_text]
    frame["surrogate_text_clean"] = frame["surrogate_text_raw"].map(paper_clean_text)

    coverage_breakdown = (
        frame.groupby(["race", "surrogate_text_source"]).size().unstack(fill_value=0)
    )
    print("Pre-filter text-source breakdown by race:")
    print(coverage_breakdown)

    frame["frontal_lateral"] = clean_string_col(frame["frontal_lateral"])
    frame["race"] = clean_string_col(frame["race"])
    print("Race value counts before the race filter:")
    print(frame["race"].value_counts().head(20))

    raw_label_name = f"{TARGET_LABEL}_canonical_raw"
    raw_label_numeric = pd.to_numeric(frame[raw_label_name], errors="coerce")
    blank_label = raw_label_numeric.isna()
    source_state_ok = raw_label_numeric.isin(SOURCE_ALLOWED_CANONICAL_LABELS) | (
        INCLUDE_CANONICAL_BLANK_AS_ZERO & blank_label
    )
    target_state_ok = raw_label_numeric.isin(TARGET_ALLOWED_CANONICAL_LABELS) | (
        INCLUDE_CANONICAL_BLANK_AS_ZERO & blank_label
    )
    frame = frame[
        frame["frontal_lateral"].eq("Frontal")
        & frame["surrogate_text_clean"].ne("")
        & (
            (frame["race"].eq(SOURCE_RACE) & source_state_ok)
            | (frame["race"].eq(TARGET_RACE) & target_state_ok)
        )
    ].copy()
    frame["canonical_label_state"] = canonical_label_state(frame[raw_label_name])
    frame[TARGET_LABEL] = positive_vs_rest_label(frame[raw_label_name])
    frame.reset_index(drop=True, inplace=True)

    print(f"After frontal/text/race-specific canonical-label filter: {frame.shape}")
    print(frame.groupby("race")[TARGET_LABEL].agg(["count", "sum", "mean"]))
    print("Canonical label-state counts before binary mapping:")
    print(pd.crosstab(frame["race"], frame["canonical_label_state"], dropna=False))

    print("Resolving image paths ...")
    frame["resolved_image_path"] = frame["path_to_image"].map(resolve_image_path)
    unresolved = int(frame["resolved_image_path"].isna().sum())
    print(f"Unresolved paths: {unresolved}")
    frame = frame[frame["resolved_image_path"].notna()].copy().reset_index(drop=True)

    print("Strict OpenCV/Pillow validation ...")
    valid_rows, failures = [], []
    for row_index, row in frame.iterrows():
        if row_index % 5000 == 0:
            print(f"  Checking candidate image {row_index}/{len(frame)}")
        valid, error = validate_image_file(row["resolved_image_path"])
        if valid:
            valid_rows.append(row_index)
        else:
            failures.append({
                "candidate_row_index": int(row_index),
                "path_to_image": row["path_to_image"],
                "resolved_image_path": row["resolved_image_path"],
                "race": row["race"],
                "validation_error": error,
            })
    if not valid_rows:
        raise RuntimeError("No images passed strict validation.")
    pd.DataFrame(failures, columns=[
        "candidate_row_index", "path_to_image", "resolved_image_path",
        "race", "validation_error",
    ]).to_csv(os.path.join(SAVE_DIR, "failed_image_audit.csv"), index=False)

    frame = frame.iloc[valid_rows].copy().reset_index(drop=True)
    print(f"Strictly readable rows: {frame.shape}")

    # Patient identity, cross-checked against any declared column.
    frame["patient_id_parsed"] = parse_patient_id(frame["path_to_image"])
    for declared in ("patient_id", "subject_id"):
        if declared in frame.columns and frame[declared].notna().all():
            agree = pd.crosstab(frame[declared].astype(str), frame["patient_id_parsed"])
            one_to_one = bool(
                (agree > 0).sum(axis=1).max() == 1 and (agree > 0).sum(axis=0).max() == 1
            )
            print(
                f"  patient id column {declared!r} is "
                f"{'consistent' if one_to_one else 'NOT consistent'} with the "
                "id parsed from path_to_image"
            )
            if not one_to_one:
                print(
                    "  >>> using the parsed id; every patient-disjoint guarantee "
                    "downstream depends on it"
                )
            break
    patient_ids_all = frame["patient_id_parsed"].to_numpy(dtype=str)

    # ------------------------------------------------------------------
    # Patient-level source split
    # ------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    source_rows = np.flatnonzero(frame["race"].to_numpy() == SOURCE_RACE)
    target_rows = np.flatnonzero(frame["race"].to_numpy() == TARGET_RACE)
    if len(target_rows) == 0:
        raise ValueError("No eligible target rows remain.")

    sl_rows, su_rows, excluded_source_rows = patient_level_source_split(
        patient_ids_all, source_rows, N_SL, rng
    )
    print(
        f"Source split: SL={len(sl_rows)} images / {len(sl_rows)} patients, "
        f"SU={len(su_rows)} images / "
        f"{len(np.unique(patient_ids_all[su_rows]))} patients, "
        f"excluded SL siblings={len(excluded_source_rows)}"
    )
    target_rows = rng.permutation(target_rows)

    frame["split_group"] = "unused"
    frame.loc[sl_rows, "split_group"] = "SL"
    frame.loc[su_rows, "split_group"] = "SU"
    frame.loc[target_rows, "split_group"] = "TARGET"
    frame.loc[excluded_source_rows, "split_group"] = "excluded_source_sibling"

    selected = np.concatenate([sl_rows, su_rows, target_rows])
    frame = frame.iloc[selected].copy().reset_index(drop=True)
    frame["prepared_row_index"] = np.arange(len(frame), dtype=np.int64)
    labels_all = frame[TARGET_LABEL].to_numpy(dtype=np.float32, copy=True)
    patient_ids_all = frame["patient_id_parsed"].to_numpy(dtype=str)

    sl_idx = np.flatnonzero(frame["split_group"].to_numpy() == "SL")
    su_idx = np.flatnonzero(frame["split_group"].to_numpy() == "SU")
    target_idx = np.flatnonzero(frame["split_group"].to_numpy() == "TARGET")

    sl_patients = set(patient_ids_all[sl_idx].tolist())
    su_patients = set(patient_ids_all[su_idx].tolist())
    assert sl_patients.isdisjoint(su_patients), "SL and SU share a patient."

    print("Final split summary:")
    print(frame.groupby("split_group")[TARGET_LABEL].agg(["count", "sum", "mean"]))

    # ------------------------------------------------------------------
    # Surrogate blocks
    # ------------------------------------------------------------------
    print("Building the bag-of-words surrogate ...")
    is_source_mask = np.zeros(len(frame), dtype=bool)
    is_source_mask[np.concatenate([sl_idx, su_idx])] = True
    is_target_mask = np.zeros(len(frame), dtype=bool)
    is_target_mask[target_idx] = True

    text_all, vocabulary_spec, bow_info = build_bow_surrogate(
        frame["surrogate_text_clean"].tolist(), is_source_mask, is_target_mask
    )
    vocabulary_spec.to_csv(
        os.path.join(SAVE_DIR, "surrogate_vocabulary.csv"), index=False
    )

    metadata_all, metadata_names, metadata_spec, dropped_metadata = (
        build_metadata_surrogate(frame)
    )
    metadata_spec.to_csv(
        os.path.join(SAVE_DIR, "metadata_feature_names.csv"), index=False
    )
    print(f"Surrogate dimensions: text={text_all.shape[1]}, "
          f"metadata={metadata_all.shape[1]}, "
          f"q={text_all.shape[1] + metadata_all.shape[1]}")

    split_manifests = {}
    for split_name, indices in [("SL", sl_idx), ("SU", su_idx), ("TARGET", target_idx)]:
        split_manifests[split_name] = save_split_npz(
            frame=frame,
            row_indices=indices,
            split_name=split_name,
            text_all=text_all,
            metadata_all=metadata_all,
            labels_all=labels_all,
        )

    frame["surrogate_text_sha256"] = frame["surrogate_text_clean"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    audit_candidates = [
        "prepared_row_index", "split_group", "path_to_image", "resolved_image_path",
        "race", TARGET_LABEL, raw_label_name, "canonical_label_state",
        "patient_id_parsed", "patient_id", "subject_id", "study_id",
        "surrogate_text_source", "surrogate_text_sha256",
    ]
    audit_columns = [c for c in audit_candidates if c in frame.columns]
    frame[audit_columns].to_csv(os.path.join(SAVE_DIR, "row_audit.csv"), index=False)

    q_total = int(text_all.shape[1] + metadata_all.shape[1])
    # p = 128 is the PCA cap of the image block downstream, realised by SimCLR
    # and Supervised; BioViL keeps fewer components, so its d is smaller.
    d_total = 128 + q_total
    config = {
        "target_label": TARGET_LABEL,
        "label_source": {
            "dataset": "original CheXpert canonical training labels",
            "file": CANONICAL_LABELS_PATH,
            "path_column": CANONICAL_PATH_COLUMN,
            "normalized_join_column": "path_to_image",
            "report_section_used_by_original_labeler": "impression",
            "labeler": "original rule-based CheXpert labeler",
        },
        "label_policy": (
            "both races keep canonical 1/0/-1/blank; map 1 -> 1 and "
            "0/-1/blank (unmentioned) -> 0; History text must be nonempty"
        ),
        "source_allowed_canonical_labels": list(SOURCE_ALLOWED_CANONICAL_LABELS),
        "target_allowed_canonical_labels": list(TARGET_ALLOWED_CANONICAL_LABELS),
        "canonical_blank_included_as_zero": INCLUDE_CANONICAL_BLANK_AS_ZERO,
        "binary_mapping": {
            "positive_1": 1, "explicit_negative_0": 0,
            "uncertain_-1": 0, "no_mention_blank": 0,
        },
        "source_race": SOURCE_RACE,
        "target_race": TARGET_RACE,
        "random_seed": RANDOM_SEED,
        "source_split_policy": (
            "patient-level: one image from each of 1200 distinct source "
            "patients forms SL; every other image of a selected patient is "
            "excluded; all images of remaining patients form SU"
        ),
        "n_sl": int(len(sl_idx)),
        "n_su": int(len(su_idx)),
        "n_target": int(len(target_idx)),
        "n_sl_patients": int(len(sl_patients)),
        "n_su_patients": int(len(su_patients)),
        "n_target_patients": int(len(np.unique(patient_ids_all[target_idx]))),
        "n_excluded_source_siblings": int(len(excluded_source_rows)),
        "n_rejected_images": int(len(failures)),
        "text": {
            "representation": "bag_of_words",
            "fields": [HISTORY_TEXT_FIELD, CLINICAL_HISTORY_TEXT_FIELD],
            "selection": (
                "section_history coalesced with section_clinical_history; "
                "require nonempty cleaned result; no Findings or Impression"
            ),
            "role": (
                "training-only privileged surrogate for theta/imputation; "
                "the final beta model remains image-only"
            ),
            "prohibited_as_surrogate": ["section_impression", "section_summary"],
            "cleaning": (
                "NFKC; lowercase; remove quote/backslash/NUL/repeated "
                "underscore; collapse whitespace; slashes preserved"
            ),
            "bow": bow_info,
            "dimension": int(text_all.shape[1]),
            "pca_applied_in_prepare": False,
            "note": (
                "no dimension reduction: variance-ordered components and "
                "signal-carrying directions are close to orthogonal in this "
                "corpus, so PCA/SVD discards signal; terms are selected by a "
                "two-sided document-frequency floor instead"
            ),
        },
        "image": {
            "stored_key": "X_image_uint8",
            "shape_per_row": [IMG_SIZE, IMG_SIZE],
            "channels_stored": 1,
            "dtype": "uint8",
            "resize_interpolation": "cv2.INTER_AREA",
            "clahe": False,
            "normalization_in_prepare": False,
            "paper_resnet_downstream_mean": list(PAPER_IMAGE_MEAN),
            "paper_resnet_downstream_std": list(PAPER_IMAGE_STD),
        },
        "metadata": {
            "n_features": int(metadata_all.shape[1]),
            "feature_names": metadata_names,
            "dropped_constant": dropped_metadata,
            "excluded": ["race", "ethnicity", "deceased", "split"],
        },
        "dimensions": {
            "q_text": int(text_all.shape[1]),
            "q_metadata": int(metadata_all.shape[1]),
            "q_total": q_total,
            "p_image_downstream": 128,
            "d_total": d_total,
            "nT_reference": NT_REFERENCE,
            "log_d_over_nT": float(np.log(d_total) / NT_REFERENCE),
            "l1_budget_sqrt_nT_over_log_d": float(
                np.sqrt(NT_REFERENCE / np.log(d_total))
            ),
        },
        "split_npz": split_manifests,
        "supervised_feature_selection_in_prepare": False,
        "patient_id_source": "parsed_from_path_to_image",
    }
    with open(os.path.join(SAVE_DIR, "prepare_config.json"), "w", encoding="utf-8") as h:
        json.dump(config, h, indent=2)

    print("=" * 88)
    print(f"Prepared data saved to: {SAVE_DIR}")
    print(f"SL={len(sl_idx)}, SU={len(su_idx)}, TARGET={len(target_idx)}")
    print(f"BoW shape: {text_all.shape}   metadata shape: {metadata_all.shape}")
    print(f"q = {q_total},  d = {d_total},  log(d)/nT = "
          f"{np.log(d_total) / NT_REFERENCE:.4f},  "
          f"L1 budget = {np.sqrt(NT_REFERENCE / np.log(d_total)):.2f}")
    print("No Y was used for vocabulary selection, metadata, or any surrogate fit.")


if __name__ == "__main__":
    main()
