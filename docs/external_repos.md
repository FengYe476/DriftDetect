# External Repositories

**Last updated:** April 28, 2026

This document tracks the external third-party repositories used by DriftDetect. These repos are cloned into `external/` but NOT committed to version control (excluded via `.gitignore`).

---

## Repositories

### 1. NM512/dreamerv3-torch

**Purpose:** PyTorch implementation of DreamerV3 for checkpoint loading and inference.

- **GitHub:** https://github.com/NM512/dreamerv3-torch
- **Cloned commit:** `6ef8646` (main branch, April 27, 2026)
- **Clone command:**
```bash
  cd external
  git clone https://github.com/NM512/dreamerv3-torch.git
```

**Usage in DriftDetect:**
- Load pre-trained DreamerV3 checkpoints (if available from maintainer)
- Extract model architecture for rollout generation (`src/diagnostics/extract_rollout.py`)
- Reference implementation for understanding latent dynamics

**Known issues:**
- Maintainer may not provide pre-trained checkpoints - check repo Releases page or Issues
- If no checkpoints available, we'll need to train from scratch (500k steps, ~25-30 GPU-hours per task)

**License:** MIT (as of commit 6ef8646)

---

### 2. nicklashansen/dreamer4

**Purpose:** DreamerV4 implementation for potential future cross-architecture analysis (Month 3-4).

- **GitHub:** https://github.com/nicklashansen/dreamer4
- **Cloned commit:** `bdeddfe` (main branch, April 27, 2026)
- **Clone command:**
```bash
  cd external
  git clone https://github.com/nicklashansen/dreamer4.git
```

**Usage in DriftDetect:**
- *Not used in Month 1-2* (setup and V3 diagnostics phase)
- Month 3-4: Compare V3 vs V4 drift patterns across architectures
- Reference for understanding architectural differences that might affect drift

**Known issues:**
- API may differ significantly from V3 - adapter code will need separate implementation
- Training requirements unknown (check repo documentation before Month 3)

**License:** MIT (as of commit bdeddfe)

---

## Updating External Repos

If you need to update to a newer commit of an external repo:

1. **Navigate to the repo:**
```bash
   cd external/<repo-name>
```

2. **Pull latest changes:**
```bash
   git fetch origin
   git log --oneline -10  # Review recent commits
   git checkout <new-commit-hash>
```

3. **Update this document:**
   - Change the "Cloned commit" field above
   - Update "Last updated" date at the top
   - Note the reason for updating (e.g., "Updated for bug fix in rollout extraction")

4. **Commit the documentation change:**
```bash
   cd ../..  # Back to DriftDetect root
   git add docs/external_repos.md
   git commit -m "Update external repo: <repo-name> to commit <hash>"
```

**Important:** Never commit the contents of `external/` itself - only this tracking document.

---

## Verification

To verify the external repos are at the correct commits:

```bash
cd external/dreamerv3-torch && git log --oneline -1
cd ../dreamer4 && git log --oneline -1
cd ../..
```

Expected output:
````
6ef8646 <commit message>
bdeddfe <commit message>
````

If the commit hashes don't match the documented versions, someone has modified the repos. Re-clone from scratch or reset:

```bash
cd external/<repo-name>
git fetch origin
git reset --hard <documented-commit-hash>
```

---

## Contributing

If you add a new external dependency:

1. Clone it to `external/`
2. Add an entry to this document with:
   - Repository URL
   - Purpose in DriftDetect
   - Cloned commit hash
   - Clone command
   - Known issues (if any)
3. Ensure it's excluded in `.gitignore` (check the `external/*` rule)
4. Commit this document with message: `"Add external repo: <repo-name>"`
