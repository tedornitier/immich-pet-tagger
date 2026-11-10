"""Classify one or more assets and show the full probability breakdown.

Usage:
    docker compose exec immich-pet-tagger python debug_asset.py <asset_id> [asset_id ...]

The classifier is cached to disk and only rebuilt when refs or negatives change.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

import classifier as clf_mod
import data
import embedder as emb

DATA_DIR = Path("/data")
CACHE_FILE = DATA_DIR / "debug_clf_cache.pkl"


def load_or_build_classifier():
    config = data.load_config(DATA_DIR)
    pet_names = list(config.keys())
    refs_per_pet = {
        n: data.load_pet_refs(config[n].get("person_id") or n, DATA_DIR)
        for n in pet_names
    }
    pet_names = [n for n in pet_names if refs_per_pet.get(n)]
    negatives = data.load_negatives(DATA_DIR)

    fp = clf_mod.classifier_fingerprint(pet_names, refs_per_pet, negatives)

    if CACHE_FILE.exists():
        cached = pickle.loads(CACHE_FILE.read_bytes())
        if cached.get("fingerprint") == fp:
            print("Loaded classifier from cache.")
            return cached["names"], cached["clf"], cached["scaler"]

    print("Building classifier...")
    result = clf_mod.build_classifier(pet_names, refs_per_pet, negatives)
    if result is None:
        print("Could not build classifier.")
        sys.exit(1)
    names, clf, scaler = result

    CACHE_FILE.write_bytes(pickle.dumps({"fingerprint": fp, "names": names, "clf": clf, "scaler": scaler}))
    print("Classifier built and cached.")
    return names, clf, scaler


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_asset.py <asset_id> [asset_id ...]")
        sys.exit(1)

    names, clf, scaler = load_or_build_classifier()

    for aid in sys.argv[1:]:
        print(f"\n--- {aid} ---")
        vec = emb.embed_asset(aid)
        if vec is None:
            print("  no embedding (thumbnail unavailable)")
            continue
        probs = clf.predict_proba(scaler.transform(np.asarray(vec).reshape(1, -1)))[0]
        for name, prob in sorted(zip(names, probs), key=lambda x: -x[1]):
            bar = "#" * int(prob * 40)
            print(f"  {name:20s} {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()
