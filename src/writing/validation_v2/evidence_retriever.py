import re
from typing import List, Dict, Any, Optional, Set, Tuple
from .models import Evidence, EvidenceSource, VerificationClaim


class EvidenceRetriever:
    """
    证据检索器 — 提取候选证据供 SemanticJudge 使用。

    策略：
    1. actor 作为硬 identity
    2. target 作为软提示，不作为硬过滤
    3. 窗口化检索（关键词句 + 前后各 N 句）
    4. 无命中时按分块取样
    """

    TYPE_KEYWORDS = {
        "realm_change": ["突破", "晋升", "踏入", "晋级", "进阶", "境界"],
        "knowledge_gain": ["领悟", "理解", "得知", "感悟", "明悟", "参透"],
        "inventory_acquire": ["获得", "得到", "拿到", "拾取", "收取"],
        "location_change": ["进入", "抵达", "来到", "到达"],
        "plot_flag": ["触发", "激活", "标记", "发生"],
        "relationship_change": ["关系", "交恶", "结盟", "和解"],
    }

    def __init__(
        self,
        max_context_chars: int = 800,
        max_fragments: int = 5,
        window_size: int = 1,
    ):
        self.max_context_chars = max_context_chars
        self.max_fragments = max_fragments
        self.window_size = window_size

    def retrieve(
        self,
        claim: VerificationClaim,
        writer_events: Optional[List[Dict[str, Any]]] = None,
        scene_text: str = "",
    ) -> Tuple[List[Evidence], bool]:
        """检索候选证据，返回 (evidence_list, has_candidates)"""
        evidences: List[Evidence] = []
        found_any = False

        # 1. 从 writer_events 提取
        if writer_events:
            for evt in writer_events:
                evt_type = evt.get("type", "")
                evt_desc = evt.get("description", "")
                evt_text = f"[{evt_type}] {evt_desc}" if evt_desc else evt_type

                if self._has_identity_match(claim, evt_type, evt_desc):
                    evidences.append(
                        Evidence(
                            evidence_id=f"evt_{len(evidences)}",
                            source=EvidenceSource.WRITER_EVENT,
                            text=evt_text[:self.max_context_chars],
                            metadata={"event_type": evt_type},
                        )
                    )
                    found_any = True

        # 2. 从 scene_text 提取
        if scene_text:
            fragments = self._extract_fragments(claim, scene_text)
            for frag in fragments:
                evidences.append(
                    Evidence(
                        evidence_id=f"txt_{len(evidences)}",
                        source=EvidenceSource.SCENE_TEXT,
                        text=frag[:self.max_context_chars],
                    )
                )
                found_any = True

        return evidences[:self.max_fragments], found_any

    def _has_identity_match(
        self,
        claim: VerificationClaim,
        event_type: str,
        event_desc: str,
    ) -> bool:
        """actor 是硬 identity，target 作为软提示不硬过滤"""
        if claim.actor:
            if claim.actor not in event_type and claim.actor not in event_desc:
                return False
        # target 不硬过滤，只作为候选提示
        return True

    def _get_keywords(self, claim: VerificationClaim) -> Set[str]:
        """获取检索关键词"""
        keywords = set()

        if claim.actor:
            keywords.add(claim.actor)

        if claim.target:
            for value in claim.target.values():
                if isinstance(value, str) and value:
                    keywords.add(value)

        if not keywords:
            change_type = claim.state_change_type
            default_kws = self.TYPE_KEYWORDS.get(change_type, [])
            keywords.update(default_kws)

        return keywords

    def _extract_fragments(
        self,
        claim: VerificationClaim,
        scene_text: str,
    ) -> List[str]:
        """从正文中提取相关片段（窗口化 + 分块取样）"""
        keywords = self._get_keywords(claim)

        sentences = re.split(r'[。！？\n]+', scene_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 2]

        if not sentences:
            return []

        fragments = []
        used_indices = set()

        # 策略1：关键词窗口检索
        for idx, sent in enumerate(sentences):
            if any(kw in sent for kw in keywords):
                start = max(0, idx - self.window_size)
                end = min(len(sentences), idx + self.window_size + 1)
                window_text = " ".join(sentences[start:end])
                if window_text not in used_indices:
                    fragments.append(window_text)
                    used_indices.add(window_text)
                if len(fragments) >= self.max_fragments:
                    break

        # 策略2：无命中 → 分块取样（覆盖全文）
        if not fragments:
            block_size = max(5, len(sentences) // self.max_fragments)
            for i in range(0, len(sentences), block_size):
                block = " ".join(sentences[i:i+block_size])
                fragments.append(block)
                if len(fragments) >= self.max_fragments:
                    break

        return fragments[:self.max_fragments]