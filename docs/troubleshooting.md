# Troubleshooting Guide

> **Live path note:** Supported OCR is PaddleOCR or Custom AI OCR; supported translation is Custom AI profiles (OpenAI-compatible endpoints). Sections that mention MarianMT, DeepL, Google Translate, Gemini, or OpenAI standalone providers are **archived legacy reference** only and are not active product paths.

This guide addresses common issues you might encounter when using Game-Changing Translator.

## Application Startup Issues

### Application fails to start

**Symptoms:** Error message or application immediately closes after startup.

**Possible solutions:**
- Verify you have Python 3.9 through 3.12 installed
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check your Python environment if using a virtual environment
- Try running the application from command line to see error messages: `python main.py`

### Missing DLL error

**Symptoms:** Error about missing DLL files.

**Solution:** Install the Microsoft Visual C++ Redistributable package from the [Microsoft website](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).

## OCR Issues

### PaddleOCR unavailable

**Symptoms:** Local OCR returns an error or the preview cannot run PaddleOCR.

**Solutions:**
- Verify the Python dependencies are installed with `pip install -r requirements.txt`
- In the Settings tab, check the PaddleOCR source directory, language, model size, and device settings
- If you use a local PaddleOCR source checkout, make sure the configured folder exists
- Switch the OCR model to a Custom AI OCR profile if local PaddleOCR is unavailable on the machine

### Poor OCR accuracy

**Symptoms:** Text is not recognized correctly or contains many errors.

**Solutions:**
- Adjust the PaddleOCR minimum score, model size, upscale, and detection settings in the Settings tab
- Increase the size of the captured area to include more context
- Set the correct source language in the Settings tab
- For non-Latin languages, set the correct PaddleOCR language in Settings
- Use a clearer font or increase text size in the source application if possible
- Adjust the stability threshold if text is flickering

## Translation Issues

### Language selection issues

**Symptoms:** Translation fails, wrong language detected, or OCR performs poorly.

**Solutions:**
- Make sure you've selected the correct source language for OCR
- Verify the language pair is supported by your selected Custom AI profile or OpenAI-compatible endpoint
- For non-Latin languages, verify the selected PaddleOCR language or use a Custom AI OCR profile
- Check that the CSV language files are properly installed in the application directory

### API key errors

**Symptoms:** "API key missing" or "Authentication error" messages.

**Solutions:**
- Verify you've entered the correct Custom AI API key and base URL in the Settings tab
- Check if your API key has expired or reached its limit
- Confirm the active translation profile is a Custom AI profile (legacy Google/DeepL/Gemini/OpenAI/MarianMT keys are not used on the live path)

### Legacy MarianMT translation errors *(archived)*

**Status:** MarianMT is not part of the current live path. If you still see MarianMT wording, switch the translation model to a Custom AI profile.

### Slow or intermittent translations

**Symptoms:** Translations appear slowly or inconsistently.

**Solutions:**
- Reduce scanning interval in Settings for faster detection
- Decrease stability threshold if text is stable
- Increase clear translation timeout to keep translations visible longer
- Enable file caching to improve performance for repeated text
- If using a legacy MarianMT path, lower the beam search value for faster performance
- If using Custom AI translation, check endpoint latency, model choice, and request timeout settings
- For better performance, use a computer with more RAM or CPU cores

## Interface Issues

### Overlay windows disappear

**Symptoms:** Source or target overlay windows are not visible.

**Solutions:**
- Press Alt+1 to toggle source window visibility
- Reselect source or target areas from the main interface
- Restart the application if overlays don't respond
- Check if another application is running in fullscreen mode that might be hiding the overlays

### Hotkeys not working

**Symptoms:** Keyboard shortcuts like ~ (toggle translation) don't respond.

**Solutions:**
- Make sure the keyboard module is installed: `pip install keyboard`
- Run the application as administrator (some systems restrict keyboard hooks)
- Check if another application is intercepting the same hotkeys
- Try restarting the application

## Application Performance

### High CPU/Memory Usage

**Symptoms:** Computer becomes slow when translation is running.

**Solutions:**
- Increase scan interval in Settings to reduce CPU usage
- Use a smaller source capture area
- If using a legacy MarianMT path, lower the beam search value
- Close unnecessary applications to free up system resources
- For MarianMT, offloading to GPU can improve performance (requires compatible GPU and proper setup)

### Application crashes

**Symptoms:** Application unexpectedly closes during operation.

**Solutions:**
- Check translator_debug.log for error details
- Update all dependencies to the latest versions
- Restart your computer to free up resources
- If using large legacy MarianMT models, ensure you have sufficient RAM

## File-related Issues

### Config file errors

**Symptoms:** Settings are not saved between sessions or load incorrectly.

**Solutions:**
- Delete ocr_translator_config.ini and let the application create a new one
- Check file permissions in the application directory
- Run the application as administrator to ensure write access

### Cache File Issues

**Symptoms:** Warnings about cache files or repetitive API calls.

**Solutions:**
- Use the "Clear Cache" button to reset translation caches
- Check file permissions for the cache files
- Disable and re-enable file caching in Settings

## Still Having Problems?

If you're still experiencing issues:

1. Check the debug log file (translator_debug.log) for specific error messages
2. Try running the application with OCR Debugging enabled to see exactly what's being captured
