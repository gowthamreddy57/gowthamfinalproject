"""Stub module for seqio (unavailable on Windows).

This provides minimal implementations to allow the code to load.
"""

import os
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


class SentencePieceVocabulary:
    """Stub for seqio.SentencePieceVocabulary."""
    
    def __init__(self, sentencepiece_model_file: str, extra_ids: int = 0):
        self.sentencepiece_model_file = sentencepiece_model_file
        self.extra_ids = extra_ids
        self._vocab_size = 250000 + extra_ids  # Default mT5 vocab size
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def encode(self, text: str) -> List[int]:
        """Stub encode - returns empty list."""
        print(f"[STUB] SentencePieceVocabulary.encode called - seqio not available on Windows")
        return []
    
    def decode(self, ids: List[int]) -> str:
        """Stub decode - returns empty string."""
        print(f"[STUB] SentencePieceVocabulary.decode called - seqio not available on Windows")
        return ""


class Vocabulary:
    """Base vocabulary stub."""
    pass


class TaskRegistry:
    """Stub for seqio.TaskRegistry."""
    _tasks: Dict[str, Any] = {}
    
    @classmethod
    def add(cls, name: str, **kwargs):
        """Register a task (stub - just stores the name)."""
        cls._tasks[name] = kwargs
        
    @classmethod
    def get(cls, name: str) -> Any:
        return cls._tasks.get(name)
    
    @classmethod
    def names(cls) -> List[str]:
        return list(cls._tasks.keys())


class MixtureRegistry:
    """Stub for seqio.MixtureRegistry."""
    _mixtures: Dict[str, Any] = {}
    
    @classmethod
    def add(cls, name: str, tasks: List[str], **kwargs):
        cls._mixtures[name] = {"tasks": tasks, **kwargs}
    
    @classmethod
    def get(cls, name: str) -> Any:
        return cls._mixtures.get(name)
    
    @classmethod
    def names(cls) -> List[str]:
        return list(cls._mixtures.keys())


class TfdsDataSource:
    """Stub for seqio.TfdsDataSource."""
    
    def __init__(self, tfds_name: str, splits: Optional[Dict[str, str]] = None, **kwargs):
        self.tfds_name = tfds_name
        self.splits = splits or {}


class FunctionDataSource:
    """Stub for seqio.FunctionDataSource."""
    
    def __init__(self, dataset_fn: Callable, splits: List[str], **kwargs):
        self.dataset_fn = dataset_fn
        self.splits = splits


class TextLineDataSource:
    """Stub for seqio.TextLineDataSource."""
    
    def __init__(self, source: str, **kwargs):
        self.source = source


@dataclass
class Feature:
    """Stub for seqio.Feature."""
    vocabulary: Any = None
    add_eos: bool = True
    required: bool = True
    dtype: str = "int32"


class Task:
    """Stub for seqio.Task."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs


class Mixture:
    """Stub for seqio.Mixture."""
    
    def __init__(self, name: str, tasks: List[str], **kwargs):
        self.name = name
        self.tasks = tasks


class preprocessors:
    """Stub for seqio.preprocessors."""
    
    @staticmethod
    def tokenize(dataset, output_features=None, **kwargs):
        """Stub tokenize."""
        return dataset
    
    @staticmethod
    def append_eos_after_trim(dataset, output_features=None, **kwargs):
        """Stub append_eos_after_trim."""
        return dataset


class CacheDatasetPlaceholder:
    """Stub for seqio.CacheDatasetPlaceholder."""
    
    def __init__(self, required: bool = False):
        self.required = required
    
    def __call__(self, dataset, **kwargs):
        return dataset


# Module-level convenience
def get_mixture_or_task(name: str):
    """Get a registered mixture or task."""
    if name in MixtureRegistry._mixtures:
        return MixtureRegistry.get(name)
    return TaskRegistry.get(name)


print("[INFO] Using seqio stub module - full functionality requires Linux with tensorflow-text")
