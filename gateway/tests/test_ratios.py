"""Tests for the documented ratio → pixel-size mapping."""

import pytest

from app import ratios


def test_documented_ratios_match_the_model_card():
    assert ratios.size_for("1:1") == (2048, 2048)
    assert ratios.size_for("4:3") == (2400, 1792)
    assert ratios.size_for("3:4") == (1792, 2400)
    assert ratios.size_for("3:2") == (2528, 1696)
    assert ratios.size_for("2:3") == (1696, 2528)
    assert ratios.size_for("16:9") == (2752, 1536)
    assert ratios.size_for("9:16") == (1536, 2752)


def test_documented_sizes_are_untouched_and_divisible_by_32():
    for ratio, expected in ratios.DOCUMENTED.items():
        assert ratios.size_for(ratio) == expected
        for side in expected:
            assert side % 32 == 0


def test_documented_sizes_keep_the_native_area():
    for ratio in ratios.DOCUMENTED:
        width, height = ratios.size_for(ratio)
        assert abs(width * height - ratios.NATIVE_PIXELS) < 0.05 * ratios.NATIVE_PIXELS


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3:2", "3:2"),
        (" 3 : 2 ", "3:2"),
        ("3x2", "3:2"),
        ("3X2", "3:2"),
        ("3/2", "3:2"),
        ("6:4", "3:2"),
        ("16:9", "16:9"),
        ("1024x768", "4:3"),
    ],
)
def test_normalize_accepts_common_spellings(raw, expected):
    assert ratios.normalize(raw) == expected


def test_normalize_rejects_garbage():
    assert ratios.normalize("") == ""
    assert ratios.normalize("breit") == ""
    assert ratios.normalize("16:0") == ""
    assert ratios.normalize("0:9") == ""
    assert ratios.normalize(None) == ""
    assert ratios.normalize("-3:2") == ""


def test_undocumented_ratio_is_derived_from_the_native_area():
    width, height = ratios.size_for("21:9")
    assert width % ratios.MULTIPLE == 0 and height % ratios.MULTIPLE == 0
    assert width > height
    assert abs(width / height - 21 / 9) < 0.05


def test_derived_ratio_keeps_roughly_the_native_area():
    width, height = ratios.size_for("5:4")
    assert abs(width * height - ratios.NATIVE_PIXELS) < 0.05 * ratios.NATIVE_PIXELS


def test_unknown_ratio_falls_back_to_square():
    assert ratios.size_for("breit") == ratios.DEFAULT_SIZE
    assert ratios.size_for(None) == ratios.DEFAULT_SIZE
    assert ratios.DEFAULT_SIZE == ratios.DOCUMENTED[ratios.DEFAULT_RATIO]


def test_derived_sizes_respect_the_pixel_budget():
    budget = 1024 * 1024
    width, height = ratios.size_for("32:9", max_pixels=budget)
    assert width * height <= budget
    assert abs(width / height - 32 / 9) < 0.1


def test_documented_sizes_are_clamped_to_the_budget():
    width, height = ratios.size_for("16:9", max_pixels=1024 * 1024)
    assert width * height <= 1024 * 1024
    assert width > height


def test_sides_are_never_below_the_minimum():
    width, height = ratios.size_for("1:64")
    assert width >= ratios.MIN_SIDE and height >= ratios.MIN_SIDE


def test_parse_size_reads_the_supported_spellings():
    assert ratios.parse_size("1024x768") == (1024, 768)
    assert ratios.parse_size("1024X768") == (1024, 768)
    assert ratios.parse_size("1024×768") == (1024, 768)
    assert ratios.parse_size("1024*768") == (1024, 768)
    assert ratios.parse_size("1024 768") == (1024, 768)
    assert ratios.parse_size(" 2048x2048 ") == (2048, 2048)


def test_parse_size_is_not_confused_by_a_ratio():
    assert ratios.parse_size("3:2") is None
    assert ratios.parse_size("3/2") is None
    assert ratios.parse_size("breit") is None
    assert ratios.parse_size("") is None
    assert ratios.parse_size(None) is None
    assert ratios.parse_size("1024x") is None


def test_parse_size_snaps_to_the_multiple_and_clamps():
    assert ratios.parse_size("1000x1000") == (1024, 1024)
    assert ratios.parse_size("10x10") == (ratios.MIN_SIDE, ratios.MIN_SIDE)


def test_parse_size_is_bounded_by_max_pixels():
    width, height = ratios.parse_size("8192x8192", max_pixels=1024 * 1024)
    assert width * height <= 1024 * 1024
