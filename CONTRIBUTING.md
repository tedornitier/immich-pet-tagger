# Contributing

Thanks for wanting to help! A few things make a PR much easier to review and merge:

- **Open an issue first** for new features or bigger changes, so we can agree on the approach before you spend time on it. Small fixes can go straight to a PR.
- **Target the `dev` branch**, from a feature branch in your fork (not your fork's `main`), and keep it rebased on `dev` so it has no conflicts.
- **Keep it scoped.** One change per PR, no unrelated refactors or reformatting. New behavior should be opt-in, with defaults that keep things working as they do today.
- **Test it.** Run `python -m pytest`, add tests for new logic, and say how you tested it on a real Immich library (versions, hardware, GPU or CPU).
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org) (`fix:`, `feat:`, `docs:`, ...). Reference issues with `part of #N` instead of `fixes #N`.
- **Docs:** a new or changed env var goes in the README table, and changes to the getting started steps must match in `README.md` and the in-app guide (`showGuide()` in `app/static/app.js`).
- All Immich API calls go through the backend (`app/api.py` / `app/immich.py`). The browser never talks to Immich directly.
