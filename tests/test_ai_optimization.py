import unittest

from ai_optimization import (
    AI_OPTIMIZATION_AUTO,
    AI_OPTIMIZATION_QUALITY,
    AI_OPTIMIZATION_SPEED,
    AiOcrImageCapabilityMemory,
    migrate_legacy_ai_optimization_settings,
    migrate_legacy_ocr_defaults,
    normalize_ai_optimization_mode,
    resolve_ai_ocr_image_policy,
    resolve_ai_response_mode,
)


class AiOptimizationConfigTests(unittest.TestCase):
    def test_invalid_mode_normalizes_to_auto(self):
        self.assertEqual(normalize_ai_optimization_mode("unknown"), AI_OPTIMIZATION_AUTO)

    def test_legacy_quality_settings_migrate_to_quality(self):
        settings = {
            "custom_ai_latency_mode": "safe",
            "custom_ai_ocr_image_format": "png",
            "custom_ai_ocr_image_mode": "lossless_webp",
            "custom_ai_ocr_image_quality": "95",
            "custom_ai_ocr_image_detail": "high",
            "capture_backend": "pyautogui",
        }

        mode = migrate_legacy_ai_optimization_settings(settings)

        self.assertEqual(mode, AI_OPTIMIZATION_QUALITY)
        self.assertEqual(settings["ai_optimization_mode"], AI_OPTIMIZATION_QUALITY)
        self.assertNotIn("custom_ai_latency_mode", settings)
        self.assertNotIn("custom_ai_ocr_image_format", settings)
        self.assertNotIn("custom_ai_ocr_image_mode", settings)
        self.assertNotIn("custom_ai_ocr_image_quality", settings)
        self.assertNotIn("custom_ai_ocr_image_detail", settings)
        self.assertNotIn("capture_backend", settings)

    def test_legacy_speed_settings_migrate_to_speed(self):
        settings = {
            "custom_ai_ocr_image_format": "webp",
            "custom_ai_ocr_image_mode": "small_grayscale_webp",
            "custom_ai_ocr_image_quality": "72",
            "custom_ai_ocr_image_detail": "low",
        }

        self.assertEqual(
            migrate_legacy_ai_optimization_settings(settings),
            AI_OPTIMIZATION_SPEED,
        )

    def test_balanced_legacy_settings_migrate_to_auto(self):
        settings = {
            "custom_ai_ocr_image_format": "webp",
            "custom_ai_ocr_image_mode": "balanced_webp",
            "custom_ai_ocr_image_quality": "85",
            "custom_ai_ocr_image_detail": "auto",
        }

        self.assertEqual(
            migrate_legacy_ai_optimization_settings(settings),
            AI_OPTIMIZATION_AUTO,
        )

    def test_existing_unified_mode_wins_over_legacy_values(self):
        settings = {
            "ai_optimization_mode": "speed",
            "custom_ai_ocr_image_detail": "high",
        }

        self.assertEqual(
            migrate_legacy_ai_optimization_settings(settings),
            AI_OPTIMIZATION_SPEED,
        )

    def test_old_ocr_defaults_migrate_without_overwriting_custom_values(self):
        legacy = {"stability_threshold": "2", "paddleocr_min_score": "0.35"}
        self.assertTrue(migrate_legacy_ocr_defaults(legacy))
        self.assertEqual(legacy["stability_threshold"], "0")
        self.assertEqual(legacy["paddleocr_min_score"], "0.45")

        custom = {"stability_threshold": "4", "paddleocr_min_score": "0.60"}
        self.assertFalse(migrate_legacy_ocr_defaults(custom))
        self.assertEqual(custom["stability_threshold"], "4")
        self.assertEqual(custom["paddleocr_min_score"], "0.60")


class AiResponsePolicyTests(unittest.TestCase):
    def test_response_mode_mapping(self):
        self.assertEqual(resolve_ai_response_mode("auto"), "adaptive")
        self.assertEqual(resolve_ai_response_mode("speed"), "safe")
        self.assertEqual(resolve_ai_response_mode("quality"), "safe")


class AiOcrImagePolicyTests(unittest.TestCase):
    def test_speed_policy_uses_small_low_detail_webp(self):
        decision = resolve_ai_ocr_image_policy(
            "speed",
            profile={"base_url": "https://api.openai.com/v1"},
            image_size=(1280, 180),
        )

        self.assertEqual(
            decision.contract_key,
            "webp|small_grayscale_webp|72|low",
        )

    def test_direct_xai_auto_uses_jpeg(self):
        decision = resolve_ai_ocr_image_policy(
            "auto",
            profile={"base_url": "https://api.x.ai/v1"},
            image_size=(1280, 180),
        )

        self.assertEqual(decision.image_format, "jpeg")

    def test_auto_slow_route_reduces_payload(self):
        decision = resolve_ai_ocr_image_policy(
            "auto",
            profile={"base_url": "https://relay.example/v1"},
            image_size=(1280, 180),
            route_p90_seconds=4.5,
            route_sample_count=8,
        )

        self.assertEqual(decision.image_mode, "small_grayscale_webp")
        self.assertEqual(decision.image_quality, 75)
        self.assertEqual(decision.image_detail, "low")

    def test_auto_thin_subtitle_region_uses_high_detail(self):
        decision = resolve_ai_ocr_image_policy(
            "auto",
            profile={"base_url": "https://relay.example/v1"},
            image_size=(1475, 120),
        )

        self.assertEqual(decision.image_detail, "high")
        self.assertEqual(decision.image_quality, 90)

    def test_quality_policy_uses_lossless_png_high_detail(self):
        decision = resolve_ai_ocr_image_policy(
            "quality",
            profile={"base_url": "https://relay.example/v1"},
            image_size=(1280, 180),
        )

        self.assertEqual(
            decision.contract_key,
            "png|lossless_webp|100|high",
        )

    def test_route_capability_memory_remembers_rejected_webp(self):
        memory = AiOcrImageCapabilityMemory()
        profile = {
            "base_url": "https://relay.example/v1",
            "model": "vision",
            "wire_api": "responses",
        }
        memory.mark_format_unsupported(profile, "webp")

        decision = resolve_ai_ocr_image_policy(
            "auto",
            profile=profile,
            image_size=(1280, 180),
            capability_memory=memory,
        )

        self.assertEqual(decision.image_format, "jpeg")

    def test_capability_memory_is_route_scoped(self):
        memory = AiOcrImageCapabilityMemory()
        rejected = {"base_url": "https://a.example/v1", "model": "vision"}
        healthy = {"base_url": "https://b.example/v1", "model": "vision"}
        memory.mark_format_unsupported(rejected, "webp")

        self.assertTrue(memory.is_format_unsupported(rejected, "webp"))
        self.assertFalse(memory.is_format_unsupported(healthy, "webp"))


if __name__ == "__main__":
    unittest.main()
