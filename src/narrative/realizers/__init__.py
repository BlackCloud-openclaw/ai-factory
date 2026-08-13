# src/narrative/realizers/__init__.py

from src.narrative.realizers.interfaces import TextGenerator
from src.narrative.realizers.reference import ReferenceNarrativeRealizer

__all__ = [
    "TextGenerator",
    "ReferenceNarrativeRealizer",
]