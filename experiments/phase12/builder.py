import logging
from typing import List, Optional
from src.writing.planning_contract import PlanningContract
from .matching.converters import StateChangeConverterRegistry
from .matching.state_expectation import StateExpectation, ExpectedStateChange

logger = logging.getLogger(__name__)


class StateExpectationBuilder:
    def __init__(self, converter_registry: Optional[StateChangeConverterRegistry] = None):
        self._converter_registry = converter_registry or StateChangeConverterRegistry.with_defaults()

    def build(self, contract: PlanningContract) -> StateExpectation:
        changes: List[ExpectedStateChange] = []
        for sc in contract.observables.state_changes:
            converter = self._converter_registry.get(sc.type)
            if converter is None:
                logger.warning("Unsupported StateChange type: %s, skipping", sc.type)
                continue
            expected = converter.convert(sc)
            if expected is not None:
                changes.append(expected)
        return StateExpectation(changes=tuple(changes))