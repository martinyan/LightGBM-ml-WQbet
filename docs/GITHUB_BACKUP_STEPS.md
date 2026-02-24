# Back up this HKJC production repo to your GitHub (step-by-step)

This guide assumes you have a local machine with git configured.

## 0) Pre-flight

- Ensure you are **not committing secrets** (MATON_API_KEY, tokens).
- This repo includes pinned model artifacts under `models/`.
- `hkjc.sqlite` should stay untracked (data is rebuildable).

## 1) Create a new repo on GitHub

1. Go to https://github.com/new
2. Repository name suggestion: `hkjc-prod-golden`
3. Choose **Private** (recommended)
4. Do **not** initialize with README (we already have one)
5. Click **Create repository**

Copy the repo URL, e.g.
- HTTPS: `https://github.com/<user>/hkjc-prod-golden.git`
- SSH: `git@github.com:<user>/hkjc-prod-golden.git`

## 2) Add the GitHub remote

From your repo folder:

```bash
git remote -v
# if no origin yet:
git remote add origin <YOUR_GITHUB_REPO_URL>
# if origin exists but is wrong:
git remote set-url origin <YOUR_GITHUB_REPO_URL>
```

## 3) Push all commits

```bash
git branch --show-current
# usually main

git push -u origin main
```

## 4) Git LFS? (optional)

Your model artifacts are ~10–30MB each. GitHub usually accepts this without LFS.
If your repo grows large later, consider Git LFS:

```bash
git lfs install
git lfs track "models/**/*.pkl" "models/**/*.bin" "models/**/ranker.txt"
git add .gitattributes

git add models

git commit -m "Track large model artifacts with Git LFS"
git push
```

## 5) Verify on GitHub

- Go to your repo page → Actions tab
- CI should run `scripts/run_smoke_test.py` and pass.

## 6) Reproduce elsewhere

Clone and run:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <repo>
python3 scripts/verify_artifacts.py
```

Then follow `docs/PROD_BOOTSTRAP.md`.
