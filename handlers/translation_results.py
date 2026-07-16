"""Provider result formatting, logs, and legacy provider adapters."""

import html
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta

from logger import log_debug_coalesced, summarize_text_for_log
from translation_utils import is_translation_error_result

REQUESTS_AVAILABLE = False


def _log_debug(message):
    facade = sys.modules.get("handlers.translation_handler")
    if facade is not None:
        return facade.log_debug(message)


def _append_rotating_text(*args, **kwargs):
    facade = sys.modules.get("handlers.translation_handler")
    if facade is not None:
        return facade.append_rotating_text(*args, **kwargs)


class TranslationResultsMixin:
    def _initialize_deepl_log_file(self):
        """Initialize DeepL translation log file with header if it doesn't exist."""
        try:
            if not os.path.exists(self.deepl_log_file):
                with open(self.deepl_log_file, 'w', encoding='utf-8-sig') as f:
                    f.write("=== DEEPL TRANSLATION API CALL LOG ===\n")
                    f.write(f"Log initialized: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                _log_debug(f"DeepL translation log file initialized: {self.deepl_log_file}")
        except Exception as e:
            _log_debug(f"Error initializing DeepL log file: {e}")
    
    def _is_deepl_logging_enabled(self):
        """Check if DeepL API logging is enabled in settings."""
        # For now, always return True. You can add a setting later if needed.
        return True
    
    def _log_deepl_translation_call(self, original_text, source_lang, target_lang, 
                                   context_string, context_size, translated_text, model_type, 
                                   call_start_time, call_duration):
        """Log DeepL translation API call with context information."""
        if not self._is_deepl_logging_enabled():
            return
        
        try:
            with self.deepl_log_lock:
                # Format timestamps
                start_timestamp = call_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                end_time = call_start_time + timedelta(seconds=call_duration)
                end_timestamp = end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                
                # Convert model type to display name
                model_display = "Next-gen" if model_type == "quality_optimized" else "Classic"
                
                # Build the log entry
                log_entry = f"""=== DEEPL TRANSLATION API CALL ===
Timestamp: {start_timestamp}
Language Pair: {source_lang} -> {target_lang}
Original Text: {original_text}

MESSAGE SENT TO DEEPL:
"""
                
                # Add context section if context was used
                if context_string:
                    subtitle_count = min(context_size, len(self.deepl_context_window))
                    count_text = f" ({subtitle_count} subtitle{'s' if subtitle_count != 1 else ''})" if context_size > 0 and subtitle_count > 0 else ""
                    log_entry += f"""
CONTEXT{count_text}:
{context_string}
"""
                # Add text to translate
                log_entry += f"""
TEXT TO TRANSLATE:
{original_text}

RESPONSE RECEIVED:
Model: {model_display}
Timestamp: {end_timestamp}
Call Duration: {call_duration:.3f} seconds

---BEGIN RESPONSE---
{translated_text}
---END RESPONSE---

========================================

"""
                
                # Write to log file
                with open(self.deepl_log_file, 'a', encoding='utf-8-sig') as f:
                    f.write(log_entry)
                
                _log_debug(f"DeepL translation call logged: {source_lang}->{target_lang}, Duration={call_duration:.3f}s")
        
        except Exception as e:
            _log_debug(f"Error logging DeepL translation call: {e}")

    # === UNIFIED TRANSLATE METHOD ===
    def _log_custom_short_call(self, call_type, profile, result_text, usage, duration):
        try:
            with self._custom_log_state_lock:
                log_executor = self._custom_log_executor
                if log_executor is None:
                    _log_debug("Custom AI short log skipped after handler close")
                    return

                log_file = (
                    "CustomAI_OCR_Short_Log.txt"
                    if call_type == "ocr"
                    else "CustomAI_Translation_Short_Log.txt"
                )
                session_header = ""
                if call_type not in self._custom_session_started:
                    session_header = (
                        f"\nSESSION 1 STARTED "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                    )
                header = (
                    "========= OCR CALL ==========="
                    if call_type == "ocr"
                    else "===== TRANSLATION CALL ======="
                )
                cost = self._custom_usage_number(usage, "cost_usd")
                prompt_tokens = int(self._custom_usage_number(
                    usage,
                    "input_tokens",
                    "prompt_tokens",
                ))
                completion_tokens = int(self._custom_usage_number(
                    usage,
                    "output_tokens",
                    "completion_tokens",
                ))
                reasoning_tokens_line = ""
                if isinstance(usage, dict) and "reasoning_tokens" in usage:
                    reasoning_tokens = int(self._custom_usage_number(
                        usage,
                        "reasoning_tokens",
                    ))
                    reasoning_tokens_line = (
                        f"Reasoning Tokens: {reasoning_tokens}\n"
                    )
                cached_prompt_tokens = int(self._custom_usage_number(
                    usage,
                    "cached_input_tokens",
                    "cached_prompt_tokens",
                ))
                cached_input_ratio = self._record_custom_prompt_cache_usage(
                    call_type,
                    usage,
                    profile=profile,
                )
                if call_type == "translation":
                    self._record_custom_ai_latency_observation(
                        duration,
                        success=True,
                        profile=profile,
                    )
                block = (
                    f"{session_header}"
                    f"{header}\n"
                    f"Provider: {profile.get('name')}\n"
                    f"Model: {profile.get('model')}\n"
                    f"Duration: {duration:.3f}s\n"
                    f"Input Tokens: {prompt_tokens}\n"
                    f"Cached Input Tokens: {cached_prompt_tokens}\n"
                    f"cached_input_ratio={cached_input_ratio:.2f}\n"
                    f"Output Tokens: {completion_tokens}\n"
                    f"{reasoning_tokens_line}"
                    f"Cost: ${cost:.8f}\n"
                    f"Result:\n--------------------\n{result_text}\n--------------------\n\n"
                )
                log_executor.submit(
                    self._write_custom_short_log,
                    log_file,
                    block,
                )
                self._custom_session_started.add(call_type)
        except Exception as e:
            _log_debug(f"Custom AI short log scheduling failed: {e}")

    def _legacy_translate_disabled(self):
        """Keep the removed legacy route inert for old callers."""
        return None
        
    # === NON-LLM PROVIDER METHODS (UNCHANGED) ===
    def _google_translate(self, text_to_translate_gt, source_lang_gt, target_lang_gt):
        _log_debug(
            "Google Translate API call "
            f"{summarize_text_for_log(text_to_translate_gt)}"
        )
        api_key_google = self.app.google_api_key_var.get().strip()
        if not api_key_google: return "Google Translate API key missing"
        if not REQUESTS_AVAILABLE: return "Requests library not available for Google Translate"
        try:
            url = "https://translation.googleapis.com/language/translate/v2"
            params = {'key': api_key_google, 'q': text_to_translate_gt, 'target': target_lang_gt, 'format': 'text'}
            if source_lang_gt and source_lang_gt.lower() != 'auto': params['source'] = source_lang_gt
            response = requests.post(url, data=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result and 'data' in result and 'translations' in result['data']:
                return html.unescape(result['data']['translations'][0]['translatedText'])
            return f"Google Translate API returned unexpected result: {result}"
        except requests.exceptions.RequestException as e_req:
            return f"Google Translate API request error: {str(e_req)}"
        except Exception as e_cgt:
            return f"Google Translate API error: {type(e_cgt).__name__} - {str(e_cgt)}"

    def _deepl_translate(self, text_to_translate_dl, source_lang_dl, target_lang_dl):
        """Translate using DeepL API with context support and logging."""
        
        # Check for language change and clear context if needed
        if (self.deepl_current_source_lang is not None and 
            self.deepl_current_target_lang is not None and
            (self.deepl_current_source_lang != source_lang_dl or 
             self.deepl_current_target_lang != target_lang_dl)):
            _log_debug(f"Language pair changed from {self.deepl_current_source_lang}->{self.deepl_current_target_lang} "
                     f"to {source_lang_dl}->{target_lang_dl}, clearing DeepL context")
            self._clear_deepl_context()
        
        # Track current language pair
        self.deepl_current_source_lang = source_lang_dl
        self.deepl_current_target_lang = target_lang_dl
        
        # Get context window size from settings
        context_size = getattr(self.app, 'deepl_context_window_var', None)
        context_size = context_size.get() if context_size else 0
        
        # Build context string (source language only)
        context_string = self._build_deepl_context(context_size)
        
        # Add custom prompt prefix if available
        custom_prompt = getattr(self.app, 'custom_prompt_text', '').strip()
        if custom_prompt:
            if context_string:
                context_string = f"[{custom_prompt}]\n{context_string}"
            else:
                context_string = f"[{custom_prompt}]\n"
        
        model_type = self.app.deepl_model_type_var.get()
        _log_debug(
            "DeepL API call "
            f"{summarize_text_for_log(text_to_translate_dl)} "
            f"model_type={model_type}"
        )
        
        if context_string:
            _log_debug(
                "DeepL context prepared "
                f"{summarize_text_for_log(context_string)} "
                f"subtitles={context_size}"
            )
        
        if not self.app.deepl_api_client:
            return "DeepL API client not initialized"
        
        try:
            deepl_source_param = source_lang_dl if source_lang_dl and source_lang_dl.lower() != 'auto' else None
            
            # Check if source or target language is a beta language
            is_beta_translation = False
            if source_lang_dl and source_lang_dl.upper() in DEEPL_BETA_LANGUAGES:
                is_beta_translation = True
            if target_lang_dl and target_lang_dl.upper() in DEEPL_BETA_LANGUAGES:
                is_beta_translation = True
            
            # Log beta language detection
            if is_beta_translation:
                _log_debug(f"Beta language detected - source: {source_lang_dl}, target: {target_lang_dl}")
            
            # Prepare translation parameters
            translate_params = {
                'text': text_to_translate_dl,
                'target_lang': target_lang_dl,
                'model_type': model_type
            }
            
            # Add source language if not auto-detect
            if deepl_source_param:
                translate_params['source_lang'] = deepl_source_param
            
            # Enable beta languages flag if using beta languages
            # Must be passed via extra_body_parameters dictionary (note: plural)
            if is_beta_translation:
                translate_params['extra_body_parameters'] = {'enable_beta_languages': True}
                _log_debug("enable_beta_languages=True added to API call via extra_body_parameters")
            
            # Add context if available (IMPORTANT: context not counted for billing!)
            if context_string and deepl_source_param:  # Only send context with explicit source language
                translate_params['context'] = context_string
            
            # Record start time for logging
            call_start_time = datetime.now()
            
            try:
                result_dl = self.app.deepl_api_client.translate_text(**translate_params)
                
                # Calculate call duration
                call_duration = (datetime.now() - call_start_time).total_seconds()
                
                if result_dl and hasattr(result_dl, 'text') and result_dl.text:
                    translated_text = result_dl.text
                    
                    # Log the API call
                    self._log_deepl_translation_call(
                        original_text=text_to_translate_dl,
                        source_lang=source_lang_dl,
                        target_lang=target_lang_dl,
                        context_string=context_string,
                        context_size=context_size,
                        translated_text=translated_text,
                        model_type=model_type,
                        call_start_time=call_start_time,
                        call_duration=call_duration
                    )
                    
                    # Update context window with source text (not translation!)
                    self._update_deepl_context(text_to_translate_dl)
                    return translated_text
                return "DeepL API returned empty or invalid result"
            except Exception as quality_error:
                if (model_type == "quality_optimized" and 
                    ("language pair" in str(quality_error).lower() or 
                     "not supported" in str(quality_error).lower() or 
                     "unsupported" in str(quality_error).lower())):
                    _log_debug(f"DeepL quality_optimized failed, falling back to latency_optimized: {quality_error}")
                    
                    # Prepare fallback parameters
                    fallback_params = {
                        'text': text_to_translate_dl,
                        'target_lang': target_lang_dl,
                        'model_type': "latency_optimized"
                    }
                    
                    # Add source language if available
                    if deepl_source_param:
                        fallback_params['source_lang'] = deepl_source_param
                    
                    # Enable beta languages flag if using beta languages
                    # Note: Beta languages cannot use latency_optimized, so this fallback might fail
                    # Must be passed via extra_body_parameters dictionary (note: plural)
                    if is_beta_translation:
                        fallback_params['extra_body_parameters'] = {'enable_beta_languages': True}
                        _log_debug("Warning: Beta languages require quality_optimized model, fallback may fail")
                    
                    # Include context in fallback attempt as well
                    if context_string and deepl_source_param:
                        fallback_params['context'] = context_string
                    
                    # Record start time for fallback call
                    fallback_start_time = datetime.now()
                    
                    result_dl_fallback = self.app.deepl_api_client.translate_text(**fallback_params)
                    
                    # Calculate fallback call duration
                    fallback_duration = (datetime.now() - fallback_start_time).total_seconds()
                    
                    if result_dl_fallback and hasattr(result_dl_fallback, 'text') and result_dl_fallback.text:
                        translated_text = result_dl_fallback.text
                        
                        # Log the fallback API call (with latency_optimized model)
                        self._log_deepl_translation_call(
                            original_text=text_to_translate_dl,
                            source_lang=source_lang_dl,
                            target_lang=target_lang_dl,
                            context_string=context_string,
                            context_size=context_size,
                            translated_text=translated_text,
                            model_type="latency_optimized",
                            call_start_time=fallback_start_time,
                            call_duration=fallback_duration
                        )
                        
                        # Update context window with source text (not translation!)
                        self._update_deepl_context(text_to_translate_dl)
                        return translated_text
                    return "DeepL API fallback returned empty or invalid result"
                else:
                    raise quality_error
        except Exception as e_cdl:
            _log_debug(f"DeepL API error: {type(e_cdl).__name__} - {str(e_cdl)}")
            return f"DeepL API error: {type(e_cdl).__name__} - {str(e_cdl)}"

    def get_deepl_usage(self):
        """Get DeepL API usage statistics from the usage endpoint."""
        if not self.app.DEEPL_API_AVAILABLE:
            _log_debug("DeepL API libraries not available for usage check")
            return None
            
        api_key = self.app.deepl_api_key_var.get().strip()
        if not api_key:
            _log_debug("DeepL API key missing for usage check")
            return None
        
        if not REQUESTS_AVAILABLE:
            _log_debug("Requests library not available for DeepL usage check")
            return None
        
        try:
            url = "https://api-free.deepl.com/v2/usage"
            headers = {
                "Authorization": f"DeepL-Auth-Key {api_key}",
                "User-Agent": "OCR-Translator/1.1.0"
            }
            
            _log_debug("Checking DeepL API usage...")
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                usage_data = response.json()
                _log_debug(f"DeepL usage retrieved: {usage_data}")
                return usage_data
            elif response.status_code == 403:
                _log_debug("DeepL API: Invalid API key or unauthorized access")
                return None
            elif response.status_code == 456:
                _log_debug("DeepL API: Quota exceeded")
                return None
            else:
                _log_debug(f"DeepL API error: HTTP {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            _log_debug(f"DeepL usage API network error: {e}")
            return None
        except Exception as e:
            _log_debug(f"DeepL usage API error: {e}")
            return None

    def _marian_translate(self, text_to_translate_mm, source_lang_mm, target_lang_mm, beam_value_mm):
        _log_debug(
            "MarianMT translation call "
            f"{summarize_text_for_log(text_to_translate_mm)} "
            f"beam={beam_value_mm}"
        )
        if self.app.marian_translator is None: return "MarianMT translator not initialized"
        text_to_translate_cleaned = re.sub(r'\s+', ' ', text_to_translate_mm).strip()
        if not text_to_translate_cleaned: return ""
        try:
            self.app.marian_translator.num_beams = beam_value_mm
            result_mm = self.app.marian_translator.translate(text_to_translate_cleaned, source_lang_mm, target_lang_mm)
            return result_mm
        except Exception as e_cmm:
            return f"MarianMT translation error: {type(e_cmm).__name__} - {str(e_cmm)}"

    # === UTILITY METHODS (UNCHANGED) ===
    def _is_error_message(self, text):
        return is_translation_error_result(text)
    
    def is_placeholder_text(self, text_content):
        if not text_content: return True
        text_lower = text_content.lower().strip()
        placeholders = ["source text will appear here", "translation will appear here", "translation...", "ocr source", "source text", "loading...", "translating...", "", "translation", "...", "translation error:"]
        return text_lower in placeholders or text_lower.startswith("translation error:")

    def clear_cache(self):
        """Clear the unified translation cache."""
        self.unified_cache.clear_all()
        _log_debug("Cleared unified translation cache")

    def update_marian_active_model(self, model_name_uam, source_lang_uam=None, target_lang_uam=None):
        if self.app.marian_translator is None:
            self.initialize_marian_translator()
            if self.app.marian_translator is None:
                _log_debug("Cannot update MarianMT model - translator not initialized and init failed.")
                return False
        
        try:
            final_source_lang = source_lang_uam if source_lang_uam else self.app.marian_source_lang
            final_target_lang = target_lang_uam if target_lang_uam else self.app.marian_target_lang

            if not final_source_lang or not final_target_lang:
                _log_debug(f"Cannot update MarianMT model '{model_name_uam}': source/target language not determined.")
                return False
            
            _log_debug(f"Attempting to make MarianMT model active: {model_name_uam} for {final_source_lang}->{final_target_lang}")

            if hasattr(self.app.marian_translator, '_unload_current_model'):
                self.app.marian_translator._unload_current_model()
            
            if hasattr(self.app.marian_translator, 'direct_pairs'):
                self.app.marian_translator.direct_pairs[(final_source_lang, final_target_lang)] = model_name_uam
            
            self.unified_cache.clear_provider('marianmt')

            if hasattr(self.app.marian_translator, '_try_load_direct_model'):
                load_success = self.app.marian_translator._try_load_direct_model(final_source_lang, final_target_lang)
                if load_success:
                    _log_debug(f"Successfully loaded MarianMT model for {final_source_lang}->{final_target_lang}")
                    return True
                else:
                    _log_debug(f"Failed to load MarianMT model for {final_source_lang}->{final_target_lang}")
                    return False
            return False
        except Exception as e_umam:
            _log_debug(f"Error updating MarianMT active model: {e_umam}")
            return False

    def update_marian_beam_value(self):
        if self.app.marian_translator is not None:
            try:
                beam_value_clamped = max(1, min(50, self.app.num_beams_var.get()))
                if beam_value_clamped != self.app.num_beams_var.get():
                    self.app.num_beams_var.set(beam_value_clamped)
                self.app.marian_translator.num_beams = beam_value_clamped
                _log_debug(f"Updated MarianMT beam search value in translator to: {beam_value_clamped}")
            except Exception as e_umbv:
                _log_debug(f"Error updating MarianMT beam value: {e_umbv}")

    def calculate_text_similarity(self, text1_sim, text2_sim):
        if not text1_sim or not text2_sim: return 0.0
        if len(text1_sim) < 10 or len(text2_sim) < 10: 
            return 1.0 if text1_sim == text2_sim else 0.0
        
        words1_set = set(text1_sim.lower().split())
        words2_set = set(text2_sim.lower().split())
        intersection_len = len(words1_set.intersection(words2_set))
        union_len = len(words1_set.union(words2_set))
        return intersection_len / union_len if union_len > 0 else 0.0

    def initialize_marian_translator(self):
        if self.app.marian_translator is not None: return
        if not hasattr(self.app, 'MARIANMT_AVAILABLE') or not self.app.MARIANMT_AVAILABLE: return
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir_name = "marian_models_cache"
            if getattr(sys, 'frozen', False):
                 executable_dir = os.path.dirname(sys.executable)
                 cache_dir = os.path.join(executable_dir, "_internal", cache_dir_name)
            else:
                 cache_dir = os.path.join(base_dir, cache_dir_name)
            os.makedirs(cache_dir, exist_ok=True)
            current_beam_value = self.app.num_beams_var.get()
            self.app.marian_translator = MarianMTTranslator(cache_dir=cache_dir, num_beams=current_beam_value)
            _log_debug(f"MarianMT translator initialized (cache: {cache_dir}, beams: {current_beam_value})")
        except Exception as e:
            _log_debug(f"Error initializing MarianMT translator: {e}\n{traceback.format_exc()}")
