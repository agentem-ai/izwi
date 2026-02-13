# Manual Download: Gemma 3 1B Instruct

`Gemma-3-1b-it` must be downloaded manually from Hugging Face.

## 1. Install Hugging Face CLI

Choose one option:

```bash
pipx install huggingface_hub
```

or

```bash
python3 -m pip install --upgrade "huggingface_hub[cli]"
```

## 2. Authenticate with Hugging Face

```bash
hf auth login
```

Use a Hugging Face token with permission to read model repositories.

## 3. Download Gemma 3 1B to Izwi's model directory

```bash
hf download google/gemma-3-1b-it \
  --repo-type model \
  --local-dir "/Users/lennex/Library/Application Support/izwi/models/Gemma-3-1b-it"
```

After download completes, restart Izwi (or refresh model list) so the model is detected.
