#!/usr/bin/env bash
# Izwi macOS full uninstall helper.
# Removes app bundle, CLI links, and local user data.

set -euo pipefail

APP_PATH="/Applications/Izwi Desktop.app"
IDENTIFIER="com.agentem.izwi.desktop"

PATH_LINKS=(
  "/opt/homebrew/bin/izwi"
  "/opt/homebrew/bin/izwi-server"
  "/usr/local/bin/izwi"
  "/usr/local/bin/izwi-server"
)

DATA_PATHS=(
  "$HOME/Library/Application Support/izwi"
  "$HOME/Library/Application Support/$IDENTIFIER"
  "$HOME/Library/Saved Application State/$IDENTIFIER.savedState"
  "$HOME/Library/Caches/$IDENTIFIER"
  "$HOME/Library/WebKit/$IDENTIFIER"
)

echo "Stopping running Izwi processes..."
pkill -f "izwi-server|izwi serve|izwi-desktop|Izwi Desktop" >/dev/null 2>&1 || true

if [ -d "$APP_PATH" ]; then
  echo "Removing app bundle: $APP_PATH"
  sudo rm -rf "$APP_PATH"
fi

echo "Removing CLI links..."
for link in "${PATH_LINKS[@]}"; do
  if [ -L "$link" ]; then
    sudo rm -f "$link"
  fi
done

echo "Removing user data..."
for path in "${DATA_PATHS[@]}"; do
  rm -rf "$path"
done

echo "Done. Izwi has been removed from this macOS user account."
