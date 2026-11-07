"""Logistic regression classifier over CLIP embeddings."""

import hashlib
import logging
import random
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import data
import embedder as emb

log = logging.getLogger("classifier")


def classifier_fingerprint(pet_names: list[str], refs_per_pet: dict, negatives: list[dict]) -> str:
    """Stable hash of the inputs that define a trained classifier."""
    parts = []
    for name in sorted(pet_names):
        parts.append(name + ":" + ",".join(sorted(r["asset_id"] for r in refs_per_pet[name])))
    # A whole-photo entry's token is its bare asset id, so legacy lists fingerprint as before.
    parts.append("neg:" + ",".join(sorted(data.crop_entry_token(e) for e in negatives)))
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


def fit_logreg(clf: LogisticRegression, X, y) -> None:
    """clf.fit, silencing one benign warning. scikit-learn 1.5.2's lbfgs solver passes
    `iprint` to scipy, which scipy>=1.15 no longer accepts, emitting an OptimizeWarning on
    every fit; the optimization itself is unaffected. Drop this once scikit-learn is bumped
    to >=1.6 (which stopped passing it). Kept in one place so every training path (here, the
    scan subprocess, and the ref-analysis diagnostic) shares the same suppression."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Unknown solver options: iprint", OptimizeWarning)
        clf.fit(X, y)


def subsample_negatives(negatives: list[dict], total_refs: int) -> list[dict]:
    """At most 3x total_refs negatives, chosen deterministically."""
    target = total_refs * 3
    if len(negatives) > target:
        return random.Random(0).sample(sorted(negatives, key=data.crop_entry_key), target)
    return negatives


def embed_negative(entry: dict) -> list:
    """A crop entry adds only that crop, a whole-photo entry adds every crop of the photo."""
    if entry.get("bbox"):
        vec = emb.embed_crop_by_bbox(entry["asset_id"], entry["bbox"])
        return [vec] if vec is not None else []
    return emb.embed_asset_crops(entry["asset_id"])


def build_classifier(
    pet_names: list[str],
    refs_per_pet: dict[str, list[dict]],
    negatives: list[dict] | None = None,
) -> tuple[list[str], LogisticRegression, StandardScaler] | None:
    all_vecs = []
    all_labels = []
    unknown_idx = len(pet_names)
    names = pet_names + ["unknown"]

    for i, name in enumerate(pet_names):
        refs = refs_per_pet.get(name, [])
        log.info(f"Embedding {len(refs)} refs for '{name}'...")
        for ref in refs:
            asset_id = ref["asset_id"]
            bbox = ref.get("bbox")
            if bbox:
                vec = emb.embed_crop_by_bbox(asset_id, bbox)
                if vec is not None:
                    all_vecs.append(vec)
                    all_labels.append(i)
                else:
                    log.warning(f"  Skipped ref {asset_id} for '{name}' (could not embed crop)")
            else:
                vecs = emb.embed_asset_crops(asset_id, require_animal=True)
                if not vecs:
                    vecs = emb.embed_asset_crops(asset_id, require_animal=False)
                if vecs:
                    all_vecs.extend(vecs)
                    all_labels.extend([i] * len(vecs))
                else:
                    log.warning(f"  Skipped ref {asset_id} for '{name}' (thumbnail unavailable)")

    total_refs = sum(len(refs) for refs in refs_per_pet.values())
    if negatives:
        if len(negatives) > total_refs * 3:
            negatives = subsample_negatives(negatives, total_refs)
            log.info(f"Subsampled negatives to {len(negatives)} (3x {total_refs} refs)")

        log.info(f"Embedding {len(negatives)} negative samples...")
        for neg in negatives:
            for vec in embed_negative(neg):
                all_vecs.append(vec)
                all_labels.append(unknown_idx)

    if not all_vecs:
        log.warning("No embeddings computed, skipping classifier training.")
        return None

    X = np.array(all_vecs, dtype=np.float64)
    y = np.array(all_labels, dtype=np.intp)

    if unknown_idx not in y:
        X = np.vstack([X, np.zeros((1, X.shape[1]))])
        y = np.append(y, unknown_idx)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    fit_logreg(clf, X_scaled, y)
    log.info(f"Classifier trained on {len(y)} samples, classes: {names} ({sum(y==unknown_idx)} unknown)")
    return names, clf, scaler


def classify(vec, names, clf, scaler) -> tuple[str, float]:
    v = np.asarray(vec, dtype=np.float64).reshape(1, -1)
    probs = clf.predict_proba(scaler.transform(v))[0]
    i = int(np.argmax(probs))
    return names[i], float(probs[i])
