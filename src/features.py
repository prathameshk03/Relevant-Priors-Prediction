"""Feature extraction for current/prior radiology study pairs."""

from __future__ import annotations

from datetime import date, datetime

from .preprocess import normalize_description


BODY_PART_MAP: dict[str, tuple[str, ...]] = {
    "BRAIN": ("BRAIN", "HEAD", "HEADBRAIN", "INTRACRANIAL", "SKULL"),
    "CHEST": ("CHEST", "LUNG", "LUNGS", "THORAX", "PULMONARY", "RIB", "RIBS"),
    "ABDOMEN": ("ABDOMEN", "ABDOMINAL", "ABD"),
    "SPINE": (
        "SPINE",
        "CERVICAL",
        "C",
        "LUMBAR",
        "L",
        "THORACIC",
        "T",
        "SACRUM",
        "SACRAL",
    ),
    "PELVIS": ("PELVIS", "PELVIC", "HIP", "HIPS"),
    "EXTREMITIES": (
        "EXTREMITY",
        "EXTREMITIES",
        "ARM",
        "FOREARM",
        "HUMERUS",
        "ELBOW",
        "WRIST",
        "HAND",
        "FINGER",
        "FINGERS",
        "LEG",
        "FEMUR",
        "KNEE",
        "TIBIA",
        "FIBULA",
        "ANKLE",
        "FOOT",
        "FEET",
        "TOE",
        "TOES",
        "SHOULDER",
        "CLAVICLE",
    ),
    "HEART": ("CARDIAC", "CARDIO", "CARDIOLITE", "CORONARY", "HEART", "ECHO", "TTE"),
    "NECK": ("NECK", "SOFTTISSUE"),
    "BREAST": ("BREAST", "MAMMO", "MAMMOGRAM", "MAMMOGRAPHY"),
    "VASCULAR": ("VASCULAR", "VAS", "ARTERY", "ARTERIAL", "VEIN", "VENOUS", "DOPPLER"),
    "KIDNEY": ("KIDNEY", "KIDNEYS", "RENAL"),
}

STOPWORDS = {
    "AND",
    "BI",
    "BILATERAL",
    "CONTRAST",
    "FRONTAL",
    "LEFT",
    "ONLY",
    "PORTABLE",
    "RIGHT",
    "SINGLE",
    "VIEW",
    "WITH",
    "WITHOUT",
    "WO",
}


def extract_modality(description: str | None) -> str:
    """Map a study description to a coarse modality."""
    normalized = normalize_description(description)
    tokens = set(normalized.split())

    if "MRI" in tokens or "MR" in tokens:
        return "MRI"
    if "CT" in tokens:
        return "CT"
    if "X" in tokens and "RAY" in tokens:
        return "XRAY"
    if "XR" in tokens or "XRAY" in tokens:
        return "XRAY"
    if "ULTRASOUND" in tokens or "US" in tokens:
        return "ULTRASOUND"
    return "UNKNOWN"


def extract_body_part(description: str | None) -> str:
    """Return the first coarse body part whose keywords appear in the description."""
    normalized = normalize_description(description)
    tokens = set(normalized.split())

    for body_part, keywords in BODY_PART_MAP.items():
        if any(keyword in tokens for keyword in keywords):
            return body_part
    return "UNKNOWN"


def keyword_overlap_score(current_description: str | None, prior_description: str | None) -> float:
    """Return a Jaccard-style token overlap score for normalized descriptions."""
    current_tokens = _meaningful_tokens(current_description)
    prior_tokens = _meaningful_tokens(prior_description)
    if not current_tokens or not prior_tokens:
        return 0.0

    overlap = current_tokens & prior_tokens
    union = current_tokens | prior_tokens
    return len(overlap) / len(union)


def recency_score(current_date: str | None, prior_date: str | None) -> float:
    """Compute a bucketed recency score from YYYY-MM-DD date strings."""
    current = _parse_date(current_date)
    prior = _parse_date(prior_date)
    if current is None or prior is None:
        return 0.1

    years_diff = abs((current - prior).days) / 365
    if years_diff <= 1:
        return 1.0
    if years_diff <= 5:
        return 0.7
    if years_diff <= 10:
        return 0.4
    return 0.1


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _meaningful_tokens(description: str | None) -> set[str]:
    return {
        token
        for token in normalize_description(description).split()
        if len(token) > 1 and token not in STOPWORDS
    }
