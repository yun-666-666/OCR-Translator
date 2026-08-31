# Contributing

Thank you for helping improve this modified fork of Game-Changing Translator.

## Source project and attribution

This repository is based on [Game-Changing Translator](https://github.com/tomkam1702/OCR-Translator), created by [Tomasz Kamiński](https://github.com/tomkam1702). We thank Tomasz Kamiński for the original project and require contributors and redistributors to preserve the existing copyright, licence, and original-author notices.

Read [ATTRIBUTION.md](ATTRIBUTION.md) before redistributing a modified version or a binary release.

## Pull requests

1. Keep each change focused and include tests when practical.
2. Do not commit API keys, provider profiles, logs, caches, local databases, screenshots containing private data, or generated local configuration.
3. Preserve the GPL licence, `LICENSE`, `ATTRIBUTION.md`, and all original-author notices.
4. Explain user-visible changes and validation in the pull request.

## Development checks

The authoritative test tree is `tests/`. Install `pytest` in the development environment and run the canonical command from the repository root:

```bash
python -m pytest
```

Do not add duplicate `test_*.py` modules at the repository root. `pytest` is not a runtime dependency and is not required for normal application use.

## AI-assisted work

This fork credits GPT as the primary AI collaborator, with Claude and Grok as collaborating assistants. AI-assisted contributions must still be reviewed, tested, and submitted under the project licence.
