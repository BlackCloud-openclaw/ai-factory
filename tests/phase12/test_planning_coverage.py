import pytest
from experiments.phase12.model import EvaluationContext
from experiments.phase12.matching import RuleExecutionUnitMatcher
from experiments.phase12.metrics import PlanningCoverageMetric
from src.writing.planning_contract import PlanningContract, Intent, Execution, ExecutionUnit, Observables, ContractMetadata
from src.writing.events import ItemAcquireEvent, PlotFlagSetEvent


def create_contract(units):
    return PlanningContract(
        version="1.0", scene_id="test",
        intent=Intent(goal="test", conflict="test", expected_outcome="test"),
        execution=Execution(units=units),
        observables=Observables(),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0)
    )


class TestPlanningCoverage:
    @pytest.mark.asyncio
    async def test_all_covered(self):
        u1 = ExecutionUnit(id="U1", label="action", description="获得神秘玉佩")
        u2 = ExecutionUnit(id="U2", label="action", description="触发丹劫")
        contract = create_contract([u1, u2])
        ctx = EvaluationContext(
            planning_contract=contract,
            scene_text="",
            events=[ItemAcquireEvent(actor="林逸", item="神秘玉佩"), PlotFlagSetEvent(flag="丹劫", value=True)],
            snapshot_before={},
            snapshot_after={}
        )
        result = await PlanningCoverageMetric().evaluate(ctx)
        assert result.score == 1.0
        assert result.details["missing_unit_ids"] == []

    @pytest.mark.asyncio
    async def test_partial_covered(self):
        u1 = ExecutionUnit(id="U1", label="action", description="获得神秘玉佩")
        u2 = ExecutionUnit(id="U2", label="action", description="触发丹劫")
        contract = create_contract([u1, u2])
        ctx = EvaluationContext(
            planning_contract=contract,
            scene_text="",
            events=[ItemAcquireEvent(actor="林逸", item="神秘玉佩")],
            snapshot_before={},
            snapshot_after={}
        )
        result = await PlanningCoverageMetric().evaluate(ctx)
        assert result.score == 0.5
        assert result.details["missing_unit_ids"] == ["U2"]

    @pytest.mark.asyncio
    async def test_no_units(self):
        contract = create_contract([])
        ctx = EvaluationContext(
            planning_contract=contract,
            scene_text="",
            events=[],
            snapshot_before={},
            snapshot_after={}
        )
        result = await PlanningCoverageMetric().evaluate(ctx)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_events(self):
        u1 = ExecutionUnit(id="U1", label="action", description="获得神秘玉佩")
        contract = create_contract([u1])
        ctx = EvaluationContext(
            planning_contract=contract,
            scene_text="",
            events=[],
            snapshot_before={},
            snapshot_after={}
        )
        result = await PlanningCoverageMetric().evaluate(ctx)
        assert result.score == 0.0
        assert result.details["missing_unit_ids"] == ["U1"]