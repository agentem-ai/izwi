# Installation

Choose your platform for detailed installation instructions.

---

## Platforms

| Platform | Guide |
|----------|-------|
| **macOS** | [macOS Installation](./macos.md) |
| **Linux** | [Linux Installation](./linux.md) |
| **Windows** | [Windows Installation](./windows.md) |
| **From Source** | [Build from Source](./from-source.md) |

---

## Quick Install

### macOS

```bash
# Download and install the .dmg from GitHub Releases
open https://github.com/agentem-ai/izwi/releases
```

### Linux (Debian/Ubuntu)

```bash
# Download the .deb package
wget https://github.com/agentem-ai/izwi/releases/latest/download/izwi_amd64.deb
sudo dpkg -i izwi_amd64.deb
```

### Windows

Download and run the installer from [GitHub Releases](https://github.com/agentem-ai/izwi/releases).

---

## Verify Installation

After installation, verify everything is working:

```bash
izwi --version
izwi status
```

You should see the Izwi version number and server status.

---

## Next Steps

- [Getting Started](../getting-started.md) — Run your first commands
- [Download Models](../models/index.md) — Get AI models for inference
