# Troubleshooting Guide

The supported runtime is RapidOCR or PaddleOCR for local OCR, optional Custom AI OCR, and Custom AI translation through an OpenAI-compatible endpoint.

## Application does not start

- Use Python 3.9 through 3.12.
- From the repository root, run `python -m pip install -r requirements.txt`.
- Start with `python main.py` so import errors are visible.
- Install the Microsoft Visual C++ Redistributable if Windows reports a missing native DLL.

`pytest` is not required to run the application. It is only the development test runner.

## RapidOCR problems

RapidOCR is the default local OCR engine.

- Keep the capture region tight around the text.
- Adjust the RapidOCR minimum score if faint text is being rejected or noise is being accepted.
- Verify the application can read its bundled model resources.
- RapidOCR's source-language selection is translation context; it does not dynamically switch the bundled OCR model.

## PaddleOCR unavailable

The supported CPU source installation is defined in `requirements.txt` and includes PaddlePaddle, PaddleOCR, and PaddleX.

- Re-run `python -m pip install -r requirements.txt` in the same Python environment used to start the app.
- If you intentionally use CUDA, replace the CPU `paddlepaddle` package with the matching `paddlepaddle-gpu` build from PaddlePaddle's official installation matrix.
- If a custom PaddleOCR source directory is configured, treat it as trusted executable Python code and verify that the directory exists.
- Select PaddleOCR and use OCR Preview to trigger its first lazy initialization. The application no longer preloads PaddleOCR at startup.

## PaddleOCR accuracy or speed

The Settings tab exposes the effective PaddleOCR language/model selection, OCR version, model size, device, minimum score, upscale, detection limit and type, text-line orientation, and optional source directory.

- Select a language supported by the chosen PaddleOCR model family.
- Prefer smaller models and CPU for lower memory use; larger or GPU models can improve throughput but consume more memory.
- Avoid large capture regions combined with high upscale values because image memory grows with pixel area.
- Increase the minimum score to reject noise; decrease it if valid faint text is missing.

## Local OpenAI-compatible endpoint

- Set the base URL and model name in a Custom AI profile.
- Leave API Key blank when the local server does not require authentication. The app then omits the `Authorization` header.
- Choose the protocol expected by the server: Chat Completions or Responses API.
- If model discovery is unsupported, enter the model name manually.
- Confirm the local server is already listening; the application does not start or manage external model servers.

## Remote endpoint authentication errors

- Confirm the base URL, model, protocol, and API Key all belong to the same provider.
- Check expiry, quota, and account permissions.
- API keys are stored through the credential store and are not written in plaintext to the profile JSON during normal operation.

## Slow translation or high resource use

- Reduce the capture area and avoid unnecessary PaddleOCR upscale.
- Use RapidOCR when its recognition quality is sufficient.
- Increase scan interval if OCR is consuming too much CPU.
- Use a faster endpoint/model or the speed optimization mode.
- Race mode is globally bounded, but each winning request may still leave an already-started losing HTTP call running until its network timeout.

## Application exit is delayed

The app stops accepting new work, waits for tracked OCR/translation/Race futures for a bounded period, then continues shutdown without waiting indefinitely. A remote HTTP call that has already started cannot be forcibly cancelled by Python; check endpoint latency and request timeouts if exits repeatedly reach the bound.

## Settings or profiles differ between launch methods

Relative Custom AI profile paths, `custom_prompt.txt`, and the translation cache are resolved from the application/project root, not the process working directory. Confirm you are launching the intended checkout or packaged application directory.

## Development test failures

Run the canonical suite from the repository root:

```bash
python -m pytest
```

Pytest is configured to collect only `tests/`. Do not restore the removed duplicate root-level test modules.
