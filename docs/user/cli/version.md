# izwi version

Show version information.

---

## Synopsis

```bash
izwi version [OPTIONS]
```

---

## Description

Displays the Izwi version and optionally detailed build information.

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --full` | Show detailed version info | — |

---

## Examples

### Basic version

```bash
izwi version
```

Output:
```
izwi 0.1.0
```

### Full version info

```bash
izwi version --full
```

Output:
```
izwi 0.1.0
Build: release
Target: aarch64-apple-darwin
Rust: 1.83.0
Features: metal
```

---

## See Also

- [`izwi status`](./status.md) — Server status
- [`izwi --version`](./index.md) — Quick version check
