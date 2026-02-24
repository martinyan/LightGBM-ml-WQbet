#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    lock_path = Path('prod/manifests/artifacts.lock.json')
    if not lock_path.exists():
        print(f"Missing lockfile: {lock_path}", file=sys.stderr)
        return 2

    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    bad = 0
    for a in lock.get('artifacts', []):
        p = Path(a['path'])
        if not p.exists():
            print(f"MISSING {p}")
            bad += 1
            continue
        got = sha256_file(p)
        exp = a['sha256']
        if got != exp:
            print(f"BAD {p} sha256 expected={exp} got={got}")
            bad += 1
        else:
            print(f"OK  {p}")

    if bad:
        print(f"\nFAILED: {bad} artifact(s) mismatched", file=sys.stderr)
        return 1

    print("\nAll artifacts verified.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
