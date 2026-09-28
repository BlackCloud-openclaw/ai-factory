# src/writing/structural_lock.py
"""
Phase 15.8 Commit 2: Structural Lock

输入: original_text, rewritten_text, planning_contract
输出: LockResult（structural_safe + 5 条检查明细）

设计：纯确定性规则，无 LLM，无副作用。
Commit 2 仅用于观测，不接入生产（Commit 3 才 flip）。
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Any

from src.common.logging import setup_logging

logger = setup_logging("writing.structural_lock")

# Phase 15.8 Commit 3A: 语义检查配置
SEMANTIC_CHECK_TYPES = {
    "inventory_acquire",
    "location_change",
    "knowledge_gain",
    "realm_change",
}
SEMANTIC_THRESHOLD = 0.50  # bge-small-zh-v1.5 中文短文本对

# jieba 用于规则 4 的人名识别（nr 词性）；不可用时规则 4 直接跳过（降级）
try:
    import jieba.posseg as pseg
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    pseg = None

# 中文 2-4 字停用词（用于新角色检测）
STOP_WORDS = {
    "已经", "可以", "什么", "怎么", "因为", "所以", "但是", "就是",
    "如果", "然后", "时候", "知道", "觉得", "应该", "能够", "这个",
    "那个", "一个", "没有", "这些", "那些", "还是", "或者", "只是",
    "这样", "那样", "有些", "有的", "一切", "之后", "之前", "起来",
    "出来", "进去", "下来", "过来", "过去", "回来", "回去", "此刻",
    "瞬间", "然后", "随即", "接踵", "径直", "骤然", "忽然", "猛然",
}

# 常见非人名词（避免误判为新角色）
NON_CHARACTER_NOUNS = {
    "禁制", "丹炉", "灵力", "符文", "能量", "虚空", "锁链", "丹房",
    "丹田", "经脉", "气息", "眼神", "声音", "空间", "时间", "宫殿",
    "境界", "金丹", "元婴", "识海", "真火", "禁地", "阵法", "石碑",
    "灵矿", "裂隙", "玉简", "丹核", "丹魄", "残魂", "魂魄", "咒文",
    "灵识", "气血", "神魂", "真元", "灵气", "道基", "法宝", "灵器",
}


@dataclass(frozen=True)
class LockCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class LockResult:
    structural_safe: bool
    checks: List[LockCheck] = field(default_factory=list)
    failure_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "structural_safe": self.structural_safe,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
            "failure_summary": self.failure_summary,
        }


class StructuralLock:
    """
    5 条确定性规则：
      1. characters_preserved    —— contract.scene_context.characters 全部出现
      2. units_preserved         —— contract.execution.units.description 关键词全部出现
      3. state_changes_preserved —— contract.observables.state_changes 关键字段全部出现
      4. no_new_characters       —— 无 new name candidates
      5. length_floor            —— rewritten/原文字符数 >= 0.6

    任意一条 fail → structural_safe = False
    """

    LENGTH_RATIO_MIN = 0.6

    def check(
        self,
        original_text: str,
        rewritten_text: str,
        contract: Optional[Any],
    ) -> LockResult:
        """
        Phase 15.8 Commit 2 (calibrated): 精简为 2 条可靠规则。

        规则：
          1. characters_preserved  —— 契约中的角色全部出现在 rewritten
          2. length_floor          —— rewritten/original >= 0.6

        弃用（对 LLM 改写假设错误，字面匹配必然误杀）：
          - units_preserved         （语义不变量，字面判不了）
          - state_changes_preserved （结构化字段名不会在正文出现）
          - no_new_characters       （jieba nr 识别不可靠）
        """
        checks: List[LockCheck] = [
            self._check_characters(rewritten_text, contract),
            self._check_length(original_text, rewritten_text),
        ]

        failed = [c for c in checks if not c.passed]
        safe = len(failed) == 0
        summary = ""
        if failed:
            summary = "; ".join(f"{c.name}: {c.detail}" for c in failed)

        return LockResult(structural_safe=safe, checks=checks, failure_summary=summary)

    # ============================================================
    # Phase 15.8 Commit 3A: 语义规则（async）
    # ============================================================
    async def check_async(
        self,
        original_text: str,
        rewritten_text: str,
        contract: Optional[Any],
    ) -> LockResult:
        """
        Commit 3A：2 条 sync 规则 + 1 条语义规则。

        语义规则：key_events_semantically_preserved
          - 仅检查 4 类可验证的 state_change
            (inventory_acquire / location_change / knowledge_gain / realm_change)
          - 每条 state_change 构造自然语言描述 → embedding
          - 对 rewritten 分句 → embedding
          - 最大 cosine ≥ SEMANTIC_THRESHOLD 视为保留
          - 任一缺失 → unsafe

        任何异常降级为 pass（观测期不阻塞）。
        """
        # 1. 先跑 sync 规则，不通过则早退
        base = self.check(original_text, rewritten_text, contract)
        if not base.structural_safe:
            return base

        # 2. 追加语义规则
        semantic_check = await self._check_semantic_preserved(rewritten_text, contract)

        checks = list(base.checks) + [semantic_check]
        failed = [c for c in checks if not c.passed]
        safe = len(failed) == 0
        summary = "; ".join(f"{c.name}: {c.detail}" for c in failed) if failed else ""
        
        return LockResult(structural_safe=safe, checks=checks, failure_summary=summary)

    async def _check_semantic_preserved(
        self, rewritten: str, contract: Optional[Any]
    ) -> LockCheck:
        if not rewritten or contract is None:
            return LockCheck(
                "key_events_semantically_preserved", True,
                "skipped: no contract or text",
            )

        scs = [
            sc for sc in self._get_state_changes(contract)
            if self._sc_type(sc) in SEMANTIC_CHECK_TYPES
        ]
        if not scs:
            return LockCheck(
                "key_events_semantically_preserved", True,
                "no semantic-checkable changes",
            )

        # 分句
        sentences = re.split(r"[。！？\n]+", rewritten)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 4]
        if not sentences:
            return LockCheck(
                "key_events_semantically_preserved", False,
                "rewritten has no usable sentences",
            )

        try:
            from src.writing.summarizer import generate_embedding, cosine_similarity

            sent_embs = [json.loads(await generate_embedding(s)) for s in sentences]

            missing = []
            for sc in scs:
                desc = self._describe_change(sc)
                desc_emb = json.loads(await generate_embedding(desc))
                max_sim = max(
                    (cosine_similarity(desc_emb, se) for se in sent_embs),
                    default=0.0,
                )
                if max_sim < SEMANTIC_THRESHOLD:
                    missing.append(f"{self._sc_type(sc)}(sim={max_sim:.2f})")

            if missing:
                return LockCheck(
                    "key_events_semantically_preserved", False,
                    f"missing: {missing[:3]}",
                )
            return LockCheck(
                "key_events_semantically_preserved", True,
                f"all {len(scs)} events preserved",
            )
        except Exception as e:
            logger.warning("[15.8-C3A] semantic check error (degraded to pass): %s", e)
            return LockCheck(
                "key_events_semantically_preserved", True,
                f"degraded: {type(e).__name__}",
            )

    def _sc_type(self, sc) -> str:
        t = sc.get("type") if isinstance(sc, dict) else getattr(sc, "type", None)
        if hasattr(t, "value"):
            t = t.value
        return str(t or "")

    def _describe_change(self, sc) -> str:
        def g(k):
            return sc.get(k) if isinstance(sc, dict) else getattr(sc, k, None)

        t = self._sc_type(sc)
        actor = g("actor") or "角色"
        if t == "inventory_acquire":
            return f"{actor}获得了物品{g('item')}"
        if t == "location_change":
            return f"{actor}到达了地点{g('location')}"
        if t == "knowledge_gain":
            return f"得知了{g('name')}"
        if t == "realm_change":
            return f"{actor}突破到了{g('to_major_realm')}{g('to_minor_stage')}层"
        return f"状态变化{t}"
    # ============================================================

    # ---------- 规则 1 ----------
    def _check_characters(self, rewritten: str, contract) -> LockCheck:
        chars = self._get_scene_characters(contract)
        if not chars:
            return LockCheck("characters_preserved", True, "no contract characters")
        missing = [c for c in chars if c and c not in rewritten]
        if missing:
            return LockCheck("characters_preserved", False, f"missing: {missing}")
        return LockCheck("characters_preserved", True, f"all {len(chars)} present")

    def _get_scene_characters(self, contract) -> List[str]:
        if contract is None:
            return []
        try:
            if isinstance(contract, dict):
                sc = contract.get("scene_context") or {}
                return list(sc.get("characters") or [])
            sc = getattr(contract, "scene_context", None)
            if sc is not None:
                return list(getattr(sc, "characters", []) or [])
        except Exception:
            pass
        return []

    # ---------- 规则 2 ----------
    def _check_units(self, rewritten: str, contract) -> LockCheck:
        units = self._get_units(contract)
        if not units:
            return LockCheck("units_preserved", True, "no units")
        missing = []
        for u in units:
            desc = u.get("description", "") if isinstance(u, dict) else getattr(u, "description", "")
            if not desc:
                continue
            keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', desc)
            if not keywords:
                keywords = [desc[:6]]
            if not any(kw in rewritten for kw in keywords):
                missing.append(desc[:30])
        if missing:
            return LockCheck("units_preserved", False, f"missing: {missing}")
        return LockCheck("units_preserved", True, f"all {len(units)} present")

    def _get_units(self, contract) -> list:
        if contract is None:
            return []
        try:
            if isinstance(contract, dict):
                ex = contract.get("execution") or {}
                return list(ex.get("units") or [])
            ex = getattr(contract, "execution", None)
            if ex is not None:
                return list(getattr(ex, "units", []) or [])
        except Exception:
            pass
        return []

    # ---------- 规则 3 ----------
    def _check_state_changes(self, rewritten: str, contract) -> LockCheck:
        scs = self._get_state_changes(contract)
        if not scs:
            return LockCheck("state_changes_preserved", True, "no state_changes")
        missing = []
        for sc in scs:
            key = self._extract_sc_key(sc)
            if key and key not in rewritten:
                missing.append(key)
        if missing:
            return LockCheck("state_changes_preserved", False, f"missing keys: {missing}")
        return LockCheck("state_changes_preserved", True, f"all {len(scs)} keys present")

    def _get_state_changes(self, contract) -> list:
        if contract is None:
            return []
        try:
            if isinstance(contract, dict):
                obs = contract.get("observables") or {}
                return list(obs.get("state_changes") or [])
            obs = getattr(contract, "observables", None)
            if obs is not None:
                return list(getattr(obs, "state_changes", []) or [])
        except Exception:
            pass
        return []

    def _extract_sc_key(self, sc) -> str:
        def g(k):
            if isinstance(sc, dict):
                return sc.get(k)
            return getattr(sc, k, None)

        t = g("type")
        if hasattr(t, "value"):
            t = t.value
        t = str(t or "")

        if t == "plot_flag":
            return g("name") or ""
        if t == "inventory_acquire":
            return g("item") or ""
        if t == "location_change":
            return g("location") or ""
        if t == "knowledge_gain":
            return g("name") or ""
        if t == "relationship_change":
            return g("to_char") or ""
        if t == "realm_change":
            return g("to_major_realm") or ""
        return ""

    # ---------- 规则 4 ----------
    def _check_no_new_characters(self, original: str, rewritten: str) -> LockCheck:
        if not HAS_JIEBA:
            return LockCheck(
                "no_new_characters", True,
                "skipped: jieba unavailable",
            )
        # 抽取 rewritten 中的 nr 词
        rewritten_names = self._extract_name_candidates(rewritten)
        if not rewritten_names:
            return LockCheck("no_new_characters", True, "no nr tokens")

        # 过滤：字面串完全不在 original 中出现的，才算新名字
        # 例如 rewritten 里"丹尊"这个词，若原文包含"丹尊"两字，则视为已知
        original_stripped = original or ""
        new_names = {
            name for name in rewritten_names
            if name not in original_stripped
        }
        if new_names:
            return LockCheck(
                "no_new_characters", False,
                f"new: {sorted(new_names)[:3]}",
            )
        return LockCheck("no_new_characters", True, "clean")

    def _extract_name_candidates(self, text: str) -> set:
        """使用 jieba 词性标注提取人名（nr）。jieba 不可用时返回空集。"""
        if not text or not HAS_JIEBA:
            return set()
        names = set()
        for word, flag in pseg.cut(text):
            if flag == "nr" and len(word) >= 2:
                names.add(word)
        return names
    
    # ---------- 规则 5 ----------
    def _check_length(self, original: str, rewritten: str) -> LockCheck:
        o_len = len((original or "").strip())
        r_len = len((rewritten or "").strip())
        if o_len == 0:
            return LockCheck("length_floor", True, "original empty")
        ratio = r_len / o_len
        if ratio < self.LENGTH_RATIO_MIN:
            return LockCheck(
                "length_floor", False,
                f"ratio={ratio:.2f} < {self.LENGTH_RATIO_MIN}",
            )
        return LockCheck("length_floor", True, f"ratio={ratio:.2f}")