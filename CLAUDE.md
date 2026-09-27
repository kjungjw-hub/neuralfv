# Working conventions for this repo

- **Always commit and push.** This is a public portfolio/resume repo
  (github.com/kjungjw-hub/neuralfv). Every change — code, docs, configs,
  figures, CI — gets committed and pushed to `origin/main` as part of doing
  the work, not as a separate step to remember later. GitHub should always
  reflect the current state of the project.
- If a push is ever blocked (missing token scope, merge conflict, etc.),
  say so explicitly and name what's needed to unblock it rather than
  leaving the change uncommitted without flagging it.
- Training checkpoints (`outputs/`) are intentionally gitignored — they're
  reproducible from the committed configs (fixed seeds), not artifacts to
  version.
