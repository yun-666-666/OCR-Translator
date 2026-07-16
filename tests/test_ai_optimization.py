import unittest

from ai_optimization import (
    AI_OPTIMIZATION_AUTO,
    AI_OPTIMIZATION_QUALITY,
    AI_OPTIMIZATION_SPEED,
    AiOcrImageCapabilityMemory,
    ai_ocr_route_metric_name,
    migrate_legacy_ai_optimization_settings,
    migrate_legacy_ocr_defaults,
    normalize_ai_optimization_mode,
    looks_like_unsupported_image_format_error,
    resolve_ai_ocr_image_policy,
    resolve_ai_response_mode,
)
from custom_ai_policy import CustomAILatencyModeAdvisor


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

    def test_adaptive_high_p90_without_second_profile_stays_safe(self):
        advisor = CustomAILatencyModeAdvisor(
            min_samples=3,
            stream_latency_threshold_seconds=1.0,
            race_latency_threshold_seconds=3.0,
        )
        for duration in (3.0, 3.5, 4.0):
            advisor.observe_request(duration, success=True)

        decision = advisor.resolve(
            "adaptive",
            stream_supported=True,
            healthy_race_profile_count=1,
        )

        self.assertEqual(decision.mode, "safe")

    def test_adaptive_very_high_p90_with_two_profiles_uses_bounded_race(self):
        advisor = CustomAILatencyModeAdvisor(
            min_samples=3,
            race_latency_threshold_seconds=3.0,
        )
        for duration in (4.0, 4.5, 5.0):
            advisor.observe_request(duration, success=True)

        decision = advisor.resolve(
            "adaptive",
            stream_supported=True,
            healthy_race_profile_count=2,
        )

        self.assertEqual(decision.mode, "race")

    def test_adaptive_primary_cooldown_with_one_alternative_stays_safe(self):
        advisor = CustomAILatencyModeAdvisor(min_samples=3)

        decision = advisor.resolve(
            "adaptive",
            stream_supported=True,
            healthy_race_profile_count=1,
            primary_cooldown_seconds=10.0,
        )

        self.assertEqual(decision.mode, "safe")


class AiOcrImagePolicyTests(unittest.TestCase):
    def test_route_metric_name_is_profile_scoped_and_sanitized(self):
        first = ai_ocr_route_metric_name(
            {"base_url": "https://a.example/v1", "model": "vision-a"}
        )
        second = ai_ocr_route_metric_name(
            {"base_url": "https://b.example/v1", "model": "vision-b"}
        )

        self.assertTrue(first.startswith("api_ocr_duration:"))
        self.assertNotEqual(first, second)
        self.assertNotIn("example", first)

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
            image_size=(640, 240),
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

    def test_auto_thin_or_large_image_stays_high_detail_on_slow_route(self):
        thin = resolve_ai_ocr_image_policy(
            "auto",
            profile={"base_url": "https://relay.example/v1"},
            image_size=(1280, 100),
            route_p90_seconds=4.5,
            route_sample_count=8,
        )
        large = resolve_ai_ocr_image_policy(
            "auto",
            profile={"base_url": "https://relay.example/v1"},
            image_size=(1920, 1080),
            route_p90_seconds=4.5,
            route_sample_count=8,
        )
        wide = resolve_ai_ocr_image_policy(
            "auto",
            profile={"base_url": "https://relay.example/v1"},
            image_size=(1920, 200),
            route_p90_seconds=4.5,
            route_sample_count=8,
        )

        self.assertEqual(thin.image_detail, "high")
        self.assertEqual(large.image_detail, "high")
        self.assertEqual(wide.image_detail, "high")

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

    def test_only_supported_format_error_rejects_current_format(self):
        self.assertTrue(
            looks_like_unsupported_image_format_error(
                "Only JPEG and PNG images are supported.",
                "webp",
            )
        )
        self.assertFalse(
            looks_like_unsupported_image_format_error(
                "Only JPEG images are supported.",
                "jpeg",
            )
        )


if __name__ == "__main__":
    unittest.main()
