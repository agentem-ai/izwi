# Izwi on macOS

## Install

1. Download `Izwi` for macOS from the latest GitHub release.
2. Open the downloaded `.dmg`.
3. Drag `Izwi.app` into `/Applications`.
4. Launch `Izwi` from Applications.
5. On first launch, Izwi will try to link `izwi` and `izwi-server` into your PATH and may ask for admin permission.
6. Verify in Terminal:

```bash
izwi --version
izwi serve --help
```

## Uninstall

### One-shot uninstall script

From a cloned repo:

```bash
./scripts/uninstall-macos.sh
```

Or from release assets:

```bash
chmod +x uninstall-izwi-macos.sh
./uninstall-izwi-macos.sh
```

### Manual full uninstall

```bash
pkill -f "izwi-server|izwi serve|izwi-desktop|/Applications/Izwi\.app|Izwi\.app" >/dev/null 2>&1 || true
sudo rm -rf "/Applications/Izwi.app"
sudo rm -f /opt/homebrew/bin/izwi /opt/homebrew/bin/izwi-server
sudo rm -f /usr/local/bin/izwi /usr/local/bin/izwi-server
rm -rf "$HOME/Library/Application Support/izwi"
rm -rf "$HOME/Library/Application Support/com.agentem.izwi.desktop"
rm -rf "$HOME/Library/Saved Application State/com.agentem.izwi.desktop.savedState"
rm -rf "$HOME/Library/Caches/com.agentem.izwi.desktop"
rm -rf "$HOME/Library/WebKit/com.agentem.izwi.desktop"
```

Verify cleanup:

```bash
command -v izwi || echo "izwi removed"
pgrep -af "izwi|izwi-server|izwi-desktop" || echo "no izwi processes"
```
