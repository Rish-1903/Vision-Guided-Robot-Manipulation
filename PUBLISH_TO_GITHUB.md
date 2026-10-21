# Publishing your *actual* tested project to GitHub

These instructions are intended for the latest working **local Mac folder**. If you changed a destination, TCP geometry or motion code after V4, retain those local files instead of replacing them with this older source snapshot.

## Recommended repository

`Rish-1903/Vision-Guided-Robotic-Manipulation`

Start private until you finish checking third-party attribution, source headers, video/image permissions and the Task 2 calibration discrepancy.

## Steps in macOS Terminal

Install GitHub CLI if not already installed:

```bash
brew install gh
gh auth login
```

Change into your actual working pipeline directory (not an outdated copy). Copy over the documentation README/.gitignore/.github/workflows and `docs/` from this kit, without overwriting your fixed Python code.

```bash
git init
git branch -M main
git add .
git status --short
git commit -m "Initial vision-guided robotic manipulation pipeline"
gh repo create Rish-1903/Vision-Guided-Robotic-Manipulation \
  --private --source=. --remote=origin --push
```

The `gh repo create` command creates the private remote repository and pushes committed files. If you prefer a public repository, change `--private` to `--public` **only after** reviewing the tracked contents and third-party licenses.

If you already have a local Git repository, do not blindly rerun `git init`, rename your branch, or add a duplicate `origin`; inspect `git remote -v` first.

## Verify the upload

```bash
gh repo view Rish-1903/Vision-Guided-Robotic-Manipulation --web
```

If you're not ready to commit public research code, keep the repository private and upload only non-sensitive source and docs. The generated `runs/` directory, `.ttt` workcell backups, calibration `.npz` matrices and large screenshots/videos are gitignored by default.
