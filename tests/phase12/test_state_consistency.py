import pytest
from experiments.phase12.model import EvaluationContext
from experiments.phase12.metrics import StateConsistencyMetric
from src.writing.planning_contract import PlanningContract, Intent, Execution, Observables, ContractMetadata, StateChange


def create_contract(changes):
    return PlanningContract(
        version="1.0", scene_id="test",
        intent=Intent(goal="test", conflict="test", expected_outcome="test"),
        execution=Execution(units=[]),
        observables=Observables(state_changes=changes),
        constraints=[],
        metadata=ContractMetadata(chapter=1, scene_index=0)
    )


def mock_snap(characters=None, flags=None, rels=None):
    class M:
        pass
    m = M()
    m.characters = characters or {}
    m.global_flags = flags or {}
    m.relationships = rels or {}
    return m


class TestStateConsistency:
    @pytest.mark.asyncio
    async def test_realm_match(self):
        c = create_contract([StateChange(type="realm", actor="林逸", to_major_realm="筑基", to_minor_stage=1)])
        ctx = EvaluationContext(
            planning_contract=c,
            scene_text="",
            events=[],
            snapshot_before=mock_snap(characters={"林逸": {"realm": "炼气"}}),
            snapshot_after=mock_snap(characters={"林逸": {"realm": "筑基"}})
        )
        result = await StateConsistencyMetric().evaluate(ctx)
        assert result.score == 1.0
        field_results = result.details.get("field_results", [])
        if field_results:
            assert "expectation_id" in field_results[0]

    @pytest.mark.asyncio
    async def test_realm_mismatch(self):
        c = create_contract([StateChange(type="realm", actor="林逸", to_major_realm="金丹", to_minor_stage=1)])
        ctx = EvaluationContext(
            planning_contract=c,
            scene_text="",
            events=[],
            snapshot_before=mock_snap(characters={"林逸": {"realm": "炼气"}}),
            snapshot_after=mock_snap(characters={"林逸": {"realm": "筑基"}})
        )
        result = await StateConsistencyMetric().evaluate(ctx)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_hp_match(self):
        c = create_contract([StateChange(type="hp", actor="林逸", new_hp=80)])
        ctx = EvaluationContext(
            planning_contract=c,
            scene_text="",
            events=[],
            snapshot_before=mock_snap(characters={"林逸": {"hp": 100}}),
            snapshot_after=mock_snap(characters={"林逸": {"hp": 80}})
        )
        result = await StateConsistencyMetric().evaluate(ctx)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_hp_mismatch(self):
        c = create_contract([StateChange(type="hp", actor="林逸", new_hp=80)])
        ctx = EvaluationContext(
            planning_contract=c,
            scene_text="",
            events=[],
            snapshot_before=mock_snap(characters={"林逸": {"hp": 100}}),
            snapshot_after=mock_snap(characters={"林逸": {"hp": 90}})
        )
        result = await StateConsistencyMetric().evaluate(ctx)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_plot_flag_match(self):
        c = create_contract([StateChange(type="plot_flag", name="丹劫触发", value=True)])
        ctx = EvaluationContext(
            planning_contract=c,
            scene_text="",
            events=[],
            snapshot_before=mock_snap(flags={}),
            snapshot_after=mock_snap(flags={"丹劫触发": True})
        )
        result = await StateConsistencyMetric().evaluate(ctx)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_multiple_changes(self):
        changes = [
            StateChange(type="realm", actor="林逸", to_major_realm="筑基", to_minor_stage=1),
            StateChange(type="hp", actor="林逸", new_hp=80),
            StateChange(type="plot_flag", name="丹劫触发", value=True),
        ]
        c = create_contract(changes)
        ctx = EvaluationContext(
            planning_contract=c,
            scene_text="",
            events=[],
            snapshot_before=mock_snap(
                characters={"林逸": {"realm": "炼气", "hp": 100}},
                flags={}
            ),
            snapshot_after=mock_snap(
                characters={"林逸": {"realm": "筑基", "hp": 80}},
                flags={"丹劫触发": True}
            )
        )
        result = await StateConsistencyMetric().evaluate(ctx)
        assert result.score == 1.0
        assert result.details["matched"] == 3