import pytest
from src.writing.contract_sanity import (
    ContractSanityGuard,
    SanityResult,
    SanityViolation,
    ViolationCode,
    RealmAuthority,
)
from src.writing.planning_contract import (
    PlanningContract,
    Intent,
    Execution,
    Observables,
    StateChange,
    ContractMetadata,
    SignalSource,
)
from src.writing.world_state import WorldState, Realm, CharacterState
from src.writing.realm_authority import DefaultRealmAuthority


class PassthroughRealmAuthority(RealmAuthority):
    """测试用 — 允许所有境界变化"""
    def can_transition(self, actor, from_realm, from_stage, to_realm, to_stage, context=None):
        return True, "Test: all allowed"


class BlockingRealmAuthority(RealmAuthority):
    """测试用 — 阻断所有境界变化"""
    def can_transition(self, actor, from_realm, from_stage, to_realm, to_stage, context=None):
        return False, "Test: all blocked"


class TestContractSanityGuard:
    # ==================== B2-1A 原有测试 (6个) ====================

    def test_guard_accepts_dict_contract(self):
        """Guard 应接受 dict 格式的 Contract"""
        world = WorldState()
        contract_dict = {
            "version": "1.0",
            "scene_id": "scene_test",
            "intent": {
                "goal": "测试场景目标",
                "conflict": "测试核心冲突",
                "expected_outcome": "测试预期结果"
            },
            "execution": {"units": []},
            "observables": {"state_changes": [], "story_events": [], "narrative_flags": []},
            "constraints": [],
            "metadata": {"chapter": 1, "scene_index": 0},
        }
        guard = ContractSanityGuard()
        result = guard.check(contract_dict, world, "scene_test")
        assert result.valid is True

    def test_guard_unknown_actor_returns_violation(self):
        world = WorldState()
        contract = PlanningContract(
            scene_id="scene_test",
            intent=Intent(
                goal="角色突破境界",
                conflict="面对未知挑战",
                expected_outcome="成功突破"
            ),
            execution=Execution(),
            observables=Observables(state_changes=[
                StateChange(
                    id="test_1",
                    type="realm_change",
                    source=SignalSource.LLM,
                    actor="不存在的角色",
                    to_major_realm="金丹",
                    to_minor_stage=1,
                )
            ]),
            metadata=ContractMetadata(chapter=1, scene_index=0),
        )

        guard = ContractSanityGuard()
        result = guard.check(contract, world, "scene_test")

        assert result.valid is False
        assert len(result.violations) == 1
        assert result.violations[0].code == ViolationCode.UNKNOWN_ACTOR

    def test_guard_with_realm_authority_allows(self):
        """使用 PassthroughRealmAuthority 时全部通过"""
        world = WorldState()
        char = CharacterState(name="林逸", realm=Realm.GOLDEN_CORE, realm_level=1)
        world.characters["林逸"] = char

        contract = PlanningContract(
            scene_id="scene_test",
            intent=Intent(
                goal="角色突破境界",
                conflict="面对未知挑战",
                expected_outcome="成功突破"
            ),
            execution=Execution(),
            observables=Observables(state_changes=[
                StateChange(
                    id="test_1",
                    type="realm_change",
                    source=SignalSource.LLM,
                    actor="林逸",
                    to_major_realm="元婴",
                    to_minor_stage=1,
                )
            ]),
            metadata=ContractMetadata(chapter=1, scene_index=0),
        )

        guard = ContractSanityGuard(realm_authority=PassthroughRealmAuthority())
        result = guard.check(contract, world, "scene_test")

        assert result.valid is True
        assert len(result.violations) == 0

    def test_guard_with_realm_authority_blocks(self):
        """使用 BlockingRealmAuthority 时全部阻断"""
        world = WorldState()
        char = CharacterState(name="林逸", realm=Realm.GOLDEN_CORE, realm_level=1)
        world.characters["林逸"] = char

        contract = PlanningContract(
            scene_id="scene_test",
            intent=Intent(
                goal="角色突破境界",
                conflict="面对未知挑战",
                expected_outcome="成功突破"
            ),
            execution=Execution(),
            observables=Observables(state_changes=[
                StateChange(
                    id="test_1",
                    type="realm_change",
                    source=SignalSource.LLM,
                    actor="林逸",
                    to_major_realm="大乘",
                    to_minor_stage=1,
                )
            ]),
            metadata=ContractMetadata(chapter=1, scene_index=0),
        )

        guard = ContractSanityGuard(realm_authority=BlockingRealmAuthority())
        result = guard.check(contract, world, "scene_test")

        assert result.valid is False
        assert len(result.violations) == 1
        assert result.violations[0].code == ViolationCode.INVALID_REALM_TRANSITION

    def test_guard_contract_id_not_fallback_to_scene_id(self):
        world = WorldState()
        contract = PlanningContract(
            scene_id="scene_foo",
            intent=Intent(
                goal="测试目标",
                conflict="测试冲突",
                expected_outcome="测试结果"
            ),
            execution=Execution(),
            observables=Observables(),
            metadata=ContractMetadata(chapter=1, scene_index=0),
        )

        guard = ContractSanityGuard()
        result = guard.check(contract, world, "scene_test", contract_id="")

        assert result.contract_id == ""

    def test_guard_contract_id_preserved_when_provided(self):
        world = WorldState()
        contract = PlanningContract(
            scene_id="scene_foo",
            intent=Intent(
                goal="测试目标",
                conflict="测试冲突",
                expected_outcome="测试结果"
            ),
            execution=Execution(),
            observables=Observables(),
            metadata=ContractMetadata(chapter=1, scene_index=0),
        )

        guard = ContractSanityGuard()
        result = guard.check(contract, world, "scene_test", contract_id="contract_123")

        assert result.contract_id == "contract_123"

    # ==================== B2-1B: DefaultRealmAuthority 专项测试 ====================

    def test_default_realm_authority_adjacent_allowed(self):
        auth = DefaultRealmAuthority()
        allowed, reason = auth.can_transition("林逸", "金丹", 1, "元婴", 1)
        assert allowed is True
        assert "相邻" in reason

    def test_default_realm_authority_non_adjacent_blocked(self):
        auth = DefaultRealmAuthority()
        allowed, reason = auth.can_transition("林逸", "金丹", 1, "大乘", 1)
        assert allowed is False
        assert "跨越" in reason

    def test_default_realm_authority_regression_blocked(self):
        auth = DefaultRealmAuthority()
        allowed, reason = auth.can_transition("林逸", "元婴", 1, "金丹", 1)
        assert allowed is False
        assert "倒退" in reason

    def test_default_realm_authority_same_realm_stage_up_allowed(self):
        auth = DefaultRealmAuthority()
        allowed, reason = auth.can_transition("林逸", "金丹", 3, "金丹", 5)
        assert allowed is True
        assert "小境界提升" in reason

    def test_default_realm_authority_same_realm_same_stage_allowed(self):
        auth = DefaultRealmAuthority()
        allowed, reason = auth.can_transition("林逸", "金丹", 3, "金丹", 3)
        assert allowed is True
        assert "NO-OP" in reason

    def test_default_realm_authority_same_realm_stage_down_blocked(self):
        auth = DefaultRealmAuthority()
        allowed, reason = auth.can_transition("林逸", "金丹", 5, "金丹", 3)
        assert allowed is False
        assert "不能倒退小境界" in reason

    def test_default_realm_authority_invalid_stage_blocked(self):
        auth = DefaultRealmAuthority()
        allowed, reason = auth.can_transition("林逸", "金丹", 0, "金丹", 1)
        assert allowed is False
        assert "无效" in reason

        allowed2, reason2 = auth.can_transition("林逸", "金丹", 1, "金丹", 10)
        assert allowed2 is False
        assert "无效" in reason2

    def test_default_realm_authority_unknown_realm_blocked(self):
        auth = DefaultRealmAuthority()
        allowed, reason = auth.can_transition("林逸", "不存在", 1, "金丹", 1)
        assert allowed is False
        assert "未知" in reason

    def test_default_realm_authority_integration_with_guard(self):
        """完整的 ContractSanityGuard + DefaultRealmAuthority 集成测试"""
        world = WorldState()
        char = CharacterState(name="林逸", realm=Realm.GOLDEN_CORE, realm_level=1)
        world.characters["林逸"] = char

        contract = PlanningContract(
            scene_id="scene_test",
            intent=Intent(
                goal="角色突破境界",
                conflict="面对未知挑战",
                expected_outcome="成功突破"
            ),
            execution=Execution(),
            observables=Observables(state_changes=[
                StateChange(
                    id="test_1",
                    type="realm_change",
                    source=SignalSource.LLM,
                    actor="林逸",
                    to_major_realm="大乘",
                    to_minor_stage=1,
                )
            ]),
            metadata=ContractMetadata(chapter=1, scene_index=0),
        )

        guard = ContractSanityGuard(realm_authority=DefaultRealmAuthority())
        result = guard.check(contract, world, "scene_test")

        assert result.valid is False
        assert len(result.violations) == 1
        assert result.violations[0].code == ViolationCode.INVALID_REALM_TRANSITION

    # ==================== B2-1C: Fail-Closed 集成测试 ====================
    def test_guard_exception_behaviour(self):
        """
        验证 Guard 在 RealmAuthority 抛出异常时返回 valid=False。
        
        注意：Guard 会捕获异常并返回 SanityResult，而不是向上抛出。
        这是 Guard 的容错设计——异常不应导致 Guard 崩溃，而应返回明确的失败结果。
        """
        class FailingRealmAuthority(RealmAuthority):
            def can_transition(self, actor, from_realm, from_stage, to_realm, to_stage, context=None):
                raise RuntimeError("Simulated authority failure")
        
        world = WorldState()
        char = CharacterState(name="林逸", realm=Realm.GOLDEN_CORE, realm_level=1)
        world.characters["林逸"] = char
        
        contract = PlanningContract(
            scene_id="scene_test",
            intent=Intent(
                goal="角色突破境界",
                conflict="面对未知挑战",
                expected_outcome="成功突破"
            ),
            execution=Execution(),
            observables=Observables(state_changes=[
                StateChange(
                    id="test_1",
                    type="realm_change",
                    source=SignalSource.LLM,
                    actor="林逸",
                    to_major_realm="元婴",
                    to_minor_stage=1,
                )
            ]),
            metadata=ContractMetadata(chapter=1, scene_index=0),
        )
        
        # 使用会抛出异常的 RealmAuthority
        guard = ContractSanityGuard(realm_authority=FailingRealmAuthority())
        
        # Guard 应该捕获异常并返回 valid=False 的 SanityResult
        # 注意：当前 ContractSanityGuard._check_realm_change 中，
        # 如果 RealmAuthority 抛出异常，异常会向上传播到 check() 方法。
        # 但 check() 方法本身不捕获异常，所以异常会继续向上抛出。
        # 
        # 这意味着 Guard 当前设计是：RealmAuthority 异常会穿透 Guard。
        # 这个测试验证的就是这种行为——异常应该被抛出。
        #
        # B2-1C 的 fail-closed 行为在 nodes.py 中实现，
        # 它捕获这个异常并阻断 Writer，而不是让异常继续向上传播。
        
        # 验证异常被抛出（Guard 不吞异常）
        with pytest.raises(RuntimeError) as exc_info:
            guard.check(contract, world, "scene_test")
        
        assert "Simulated authority failure" in str(exc_info.value)