"""Stub module for tensorflow_datasets."""

from typing import List


class c4:
    """Stub for tfds.text.c4."""
    MC4_LANGUAGES = [
        "af", "am", "ar", "az", "be", "bg", "bn", "ca", "ceb", "co", "cs", "cy",
        "da", "de", "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fil", "fr",
        "fy", "ga", "gd", "gl", "gu", "ha", "haw", "he", "hi", "hmn", "hr", "ht",
        "hu", "hy", "id", "ig", "is", "it", "iw", "ja", "jv", "ka", "kk", "km",
        "kn", "ko", "ku", "ky", "la", "lb", "lo", "lt", "lv", "mg", "mi", "mk",
        "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "no", "ny", "pa", "pl",
        "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "sm", "sn", "so", "sq",
        "sr", "st", "su", "sv", "sw", "ta", "te", "tg", "th", "tr", "uk", "ur",
        "uz", "vi", "xh", "yi", "yo", "zh", "zh-Hans", "zh-Hant", "zu"
    ]


class text:
    """Namespace for tfds.text."""
    c4 = c4


def load(name: str, split: str = "train", **kwargs):
    """Stub for tfds.load."""
    print(f"[STUB] tfds.load('{name}', split='{split}') - tensorflow_datasets not fully available")
    return []


print("[INFO] Using tensorflow_datasets stub module")
