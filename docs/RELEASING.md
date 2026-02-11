# Releasing Izwi

This repository publishes release artifacts with GitHub Actions when a tag starting with `v` is pushed.

For the first alpha, use:

```bash
git tag v0.1.0-alpha
git push origin v0.1.0-alpha
```

## What the release workflow builds

For each OS runner, the workflow builds:

- Terminal bundle:
  - `izwi` (CLI),
  - `izwi-server`,
  - `izwi-desktop` (desktop shell binary used by `izwi serve --mode desktop`).
- Desktop installer bundle:
  - Linux: `.deb`
  - Windows: `NSIS .exe`
  - macOS: `.dmg`

All artifacts are attached to the GitHub Release for the tag.

## Signing and notarization

macOS signing is intentionally disabled by default. This keeps release builds working while no Apple certificate is configured.

When you are ready to sign/notarize, add Apple signing secrets and set the relevant environment variables in `.github/workflows/release.yml`.
The workflow is already prepared to use these optional secrets on macOS:

- `APPLE_CERTIFICATE`
- `APPLE_CERTIFICATE_PASSWORD`
- `APPLE_SIGNING_IDENTITY`
- `APPLE_ID`
- `APPLE_PASSWORD`
- `APPLE_TEAM_ID`

## Runtime logging defaults

Production defaults now use `warn` level logs:

- `izwi serve` defaults to `--log-level warn`
- `izwi-server` falls back to `warn` filters when `RUST_LOG` is not set

Users can still override at runtime:

```bash
RUST_LOG=info izwi serve
izwi serve --log-level debug
```

## Basic user startup paths

Terminal startup:

```bash
izwi serve
```

Desktop startup from terminal:

```bash
izwi serve --mode desktop
```
