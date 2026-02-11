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

Linux `.deb` installs terminal commands into `/usr/bin`:
- `izwi`
- `izwi-server`

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

## macOS CLI behavior (DMG install)

The macOS app bundle includes `izwi` and `izwi-server` in app resources.
On app startup, Izwi tries to create PATH symlinks:

- `/opt/homebrew/bin` (Apple Silicon fallback) or `/usr/local/bin`

If permissions prevent automatic setup, the app requests admin privileges via macOS and runs the link command. If that is denied, it prints fallback `sudo ln -sf ...` commands.

### macOS install steps

1. Download the macOS `.dmg` from the GitHub release.
2. Open the `.dmg`.
3. Drag `Izwi Desktop.app` into `/Applications`.
4. Launch `Izwi Desktop` from Applications.
5. Verify terminal access:

```bash
izwi --version
izwi serve --help
```

### macOS uninstall steps

Quick uninstall script:

```bash
./scripts/uninstall-macos.sh
```

Or manual full uninstall:

```bash
pkill -f "izwi-server|izwi serve|izwi-desktop|Izwi Desktop" >/dev/null 2>&1 || true
sudo rm -rf "/Applications/Izwi Desktop.app"
sudo rm -f /opt/homebrew/bin/izwi /opt/homebrew/bin/izwi-server
sudo rm -f /usr/local/bin/izwi /usr/local/bin/izwi-server
rm -rf "$HOME/Library/Application Support/izwi"
rm -rf "$HOME/Library/Application Support/com.agentem.izwi.desktop"
rm -rf "$HOME/Library/Saved Application State/com.agentem.izwi.desktop.savedState"
rm -rf "$HOME/Library/Caches/com.agentem.izwi.desktop"
rm -rf "$HOME/Library/WebKit/com.agentem.izwi.desktop"
```

Detailed user-facing instructions are also in `docs/MACOS.md`.

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
