#!/usr/bin/env python3
"""
Phase 5.1: Manipulation Check
验证 MP2 和 MP3 的 Gap 操纵与 Match 操纵是否成功。

实验流程：
1. Option Calibration - 验证四个候选选项没有默认偏好
2. Gap Check - 验证 High/Low Gap 是否改变预测分布
3. Match Check - 验证 Correct/Wrong Goal 是否改变 Top-1 预测

输出：
- raw_data.json: 所有原始概率、置信度、理由
- report.md: 统计表格 + PASS/FAIL 诊断 + Overall Summary
"""

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import httpx

# ============================================================
# 配置
# ============================================================

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"
SAMPLES_PER_CONDITION = 5
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120.0  # 增加到 120 秒
REQUEST_DELAY = 2.0      # 请求间延迟 2 秒
CONNECT_TIMEOUT = 10.0

ENTROPY_THRESHOLD_HIGH = 1.2
ENTROPY_THRESHOLD_LOW = 0.8
EFFECTIVE_CHOICES_HIGH = 3
EFFECTIVE_CHOICES_LOW = 2
CONFIDENCE_HIGH_MAX = 65
CONFIDENCE_LOW_MIN = 75
TOP1_CHANGE_RATE_CORRECT_MIN = 0.70
TOP1_CHANGE_RATE_WRONG_MAX = 0.40
CALIBRATION_MAX_TOP1 = 0.55
OUTPUT_DIR = Path(__file__).parent / "reports" / "manipulation_check"


# ============================================================
# 实验材料
# ============================================================

@dataclass
class Option:
    id: str
    text: str

@dataclass
class MasterPair:
    id: str
    scene_a: str
    options: List[Option]
    correct_goal: str
    wrong_goal: str

# MP2: 古井拓印符文 → 研究
MP2_SCENE_A = """古井的青苔在月光下泛着幽绿，林逸用手电筒光束劈开浓雾时，井口溢出的寒气正凝结在他睫毛上。铁链锈迹斑驳处泛着铜绿，他拨开青苔，发现铁链上密布的环状凹痕——像是被无数细绳勒出的伤疤。

手电筒扫过井壁时，光斑在某处突然扭曲。潮湿的岩面上，用朱砂刻着三圈螺旋纹，最深处嵌着半枚指纹。他掏出匕首刮开更多苔藓，取出怀中的拓印纸，将符文拓下。符纸上的纹路在月光下泛起幽蓝微光，他收起符纸，站起身。远处传来守夜人的梆子声，三更已过。"""

MP2_OPTIONS = [
    Option("A", "回书房，对照《宗门禁制录》查典籍"),
    Option("B", "去丹房，比对符文与药材图谱的关联"),
    Option("C", "去找长老，试探性地询问符文来历"),
    Option("D", "前往禁地外围，现场比对封印结构"),
]

MP2_CORRECT_GOAL = "当前目标是：破译符文的含义。"
MP2_WRONG_GOAL = "当前目标是：确认古井附近是否还存在其他相同符文。"

# MP3: 幽谷青铜令牌 → 调查
MP3_SCENE_A = """幽谷深处弥漫着妖兽的腥臭，林逸的玄铁剑插在岩缝里，剑刃上还沾着暗红血迹。三日前那场遭遇让他左臂多出三道深可见骨的爪痕，此刻肿胀处泛着诡异的青黑。

他撕下衣摆包扎伤口，脚下踩碎了半截断裂的妖兽骨刺。月光从岩顶裂隙漏下，照见他手中那枚刚夺来的青铜令牌。令牌表面刻着模糊的狼首纹，边角有被火焰灼烧的焦痕。他将令牌翻面，背面隐约可见一串被刮去的刻字，只剩最后一笔还能辨认——像是个"禁"字。远处传来瀑布轰鸣声，他扶着岩壁站起身。"""

MP3_OPTIONS = [
    Option("A", "回石屋，处理伤口并仔细检查令牌"),
    Option("B", "返回宗门，向执事报告妖兽与令牌"),
    Option("C", "折返幽谷深处，寻找更多痕迹与线索"),
    Option("D", "前往附近坊市，打听令牌的来历"),
]

MP3_CORRECT_GOAL = "当前目标是：查明令牌的来历。"
MP3_WRONG_GOAL = "当前目标是：返回宗门报告妖兽活动。"


# ============================================================
# 核心类
# ============================================================

class ManipulationCheck:
    def __init__(self):
        self.raw_data = {
            "calibration": [],
            "gap": [],
            "match": [],
        }
        self.master_pairs = {
            "MP2": MasterPair(
                id="MP2",
                scene_a=MP2_SCENE_A,
                options=MP2_OPTIONS,
                correct_goal=MP2_CORRECT_GOAL,
                wrong_goal=MP2_WRONG_GOAL,
            ),
            "MP3": MasterPair(
                id="MP3",
                scene_a=MP3_SCENE_A,
                options=MP3_OPTIONS,
                correct_goal=MP3_CORRECT_GOAL,
                wrong_goal=MP3_WRONG_GOAL,
            ),
        }
        self._call_count = 0
    
    # ============================================================
    # 1. Option Calibration
    # ============================================================
    
    async def calibrate(self, mp_id: str, samples: int = SAMPLES_PER_CONDITION) -> List[Dict]:
        """运行 Option Calibration，验证四个选项是否平衡"""
        mp = self.master_pairs[mp_id]
        results = []
        
        prompt = f"""你是一位小说情节分析师。请阅读以下场景结尾。

【场景结尾】
{mp.scene_a}

【任务】
判断以下四个后续场景的合理性分配。
四个方案都同样合理，没有标准答案。

A. {mp.options[0].text}
B. {mp.options[1].text}
C. {mp.options[2].text}
D. {mp.options[3].text}

请为每个选项分配概率（0-100%），总和=100%。

输出格式：
A: XX%
B: XX%
C: XX%
D: XX%

【附加问题】
是否存在明显默认选项？
（回答：是 / 否）

如果“是”，请说明是哪一个。"""

        for s in range(samples):
            self._call_count += 1
            print(f"  [Calibration] {mp_id} sample {s+1}/{samples} (call #{self._call_count})")
            response = await self._call_llm(prompt)
            result = self._parse_calibration_response(response)
            result["sample"] = s
            result["mp_id"] = mp_id
            result["raw_response"] = response
            results.append(result)
            await asyncio.sleep(REQUEST_DELAY)
        
        self.raw_data["calibration"].extend(results)
        return results
    
    def _parse_calibration_response(self, response: str) -> Dict:
        """解析 Calibration 响应，返回概率和默认选项判断"""
        probs = self._extract_probs(response)
        default = self._extract_default(response)
        return {"probabilities": probs, "default": default}
    
    # ============================================================
    # 2. Gap Check
    # ============================================================
    
    async def run_gap(self, mp_id: str, gap_type: str, samples: int = SAMPLES_PER_CONDITION) -> List[Dict]:
        """运行 Gap Check"""
        mp = self.master_pairs[mp_id]
        results = []
        
        prompt = f"""你是一位小说情节分析师。请阅读以下场景结尾。

【场景结尾】
{mp.scene_a}

【任务】
判断以下四个后续场景哪一个最合理。
四个方案都同样合理，没有标准答案。

A. {mp.options[0].text}
B. {mp.options[1].text}
C. {mp.options[2].text}
D. {mp.options[3].text}

请为每个选项分配概率（0-100%），总和=100%。

输出格式：
A: XX%
B: XX%
C: XX%
D: XX%

【置信度】
你对自己的判断有多确定？（0-100）
输出：Confidence: XX%"""

        for s in range(samples):
            self._call_count += 1
            print(f"  [Gap] {mp_id} {gap_type} sample {s+1}/{samples} (call #{self._call_count})")
            response = await self._call_llm(prompt)
            probs = self._extract_probs(response)
            confidence = self._extract_confidence(response)
            results.append({
                "mp_id": mp_id,
                "gap_type": gap_type,
                "sample": s,
                "probabilities": probs,
                "confidence": confidence,
                "raw_response": response,
            })
            await asyncio.sleep(REQUEST_DELAY)
        
        self.raw_data["gap"].extend(results)
        return results
    
    # ============================================================
    # 3. Match Check
    # ============================================================
    
    async def run_match(
        self,
        mp_id: str,
        gap_type: str,
        state_type: str,
        samples: int = SAMPLES_PER_CONDITION,
    ) -> List[Dict]:
        """运行 Match Check，测试 State 是否改变 Top-1"""
        mp = self.master_pairs[mp_id]
        results = []
        
        # 选择 State
        if state_type == "correct":
            state = mp.correct_goal
        elif state_type == "wrong":
            state = mp.wrong_goal
        else:
            raise ValueError(f"Unknown state_type: {state_type}")
        
        prompt = f"""你是一位小说情节分析师。请阅读以下场景结尾，以及附加的状态信息。

【场景结尾】
{mp.scene_a}

【附加状态信息】
{state}

【任务】
在知道上述状态信息后，判断以下四个后续场景哪一个最合理。

A. {mp.options[0].text}
B. {mp.options[1].text}
C. {mp.options[2].text}
D. {mp.options[3].text}

请为每个选项分配概率（0-100%），总和=100%。

输出格式：
A: XX%
B: XX%
C: XX%
D: XX%

【置信度】
你对自己的判断有多确定？（0-100）
输出：Confidence: XX%

【理由（关键问题）】
请用一句话说明你做出这个选择的理由。
输出：Reason: ..."""

        # 获取无 State 的 Baseline（从 Gap Check 中提取）
        gap_data = [d for d in self.raw_data["gap"] if d["mp_id"] == mp_id and d["gap_type"] == gap_type]
        if not gap_data:
            print(f"  Warning: No baseline data for {mp_id} {gap_type}, running Gap Check first...")
            gap_data = await self.run_gap(mp_id, gap_type, samples=3)
        
        for s in range(samples):
            self._call_count += 1
            print(f"  [Match] {mp_id} {gap_type} {state_type} sample {s+1}/{samples} (call #{self._call_count})")
            response = await self._call_llm(prompt)
            probs = self._extract_probs(response)
            confidence = self._extract_confidence(response)
            reason = self._extract_reason(response)
            
            # 获取对应的 Baseline Top-1
            baseline_probs = gap_data[s % len(gap_data)]["probabilities"]
            baseline_top1 = max(baseline_probs, key=baseline_probs.get)
            new_top1 = max(probs, key=probs.get)
            top1_changed = baseline_top1 != new_top1
            
            # Reason Consistency
            consistency = self._reason_consistency(reason, state, new_top1)
            
            results.append({
                "mp_id": mp_id,
                "gap_type": gap_type,
                "state_type": state_type,
                "sample": s,
                "probabilities": probs,
                "confidence": confidence,
                "baseline_top1": baseline_top1,
                "new_top1": new_top1,
                "top1_changed": top1_changed,
                "reason": reason,
                "reason_consistency": consistency,
                "raw_response": response,
            })
            await asyncio.sleep(REQUEST_DELAY)
        
        self.raw_data["match"].extend(results)
        return results
    
    # ============================================================
    # 4. LLM 调用（带重试）
    # ============================================================
    
    async def _call_llm(self, prompt: str, retries: int = MAX_RETRIES) -> str:
        """调用 LLM API，带重试和指数退避"""
        last_error = None
        
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(
                    trust_env=False,
                    timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)
                ) as client:
                    payload = {
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 512,
                    }
                    response = await client.post(
                        f"{LLM_API_BASE}/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"].get("content", "")
                    if not content:
                        content = data["choices"][0]["message"].get("reasoning_content", "")
                    
                    # 检查内容是否有效
                    if content and content.strip():
                        return content
                    else:
                        print(f"  [WARNING] Empty response, attempt {attempt+1}/{retries}")
                        last_error = "Empty response"
                        
            except httpx.TimeoutException as e:
                print(f"  [ERROR] Timeout (attempt {attempt+1}/{retries}): {e}")
                last_error = str(e)
            except Exception as e:
                print(f"  [ERROR] LLM call failed (attempt {attempt+1}/{retries}): {e}")
                last_error = str(e)
            
            # 重试前等待
            if attempt < retries - 1:
                wait = (attempt + 1) * 2
                print(f"  [RETRY] Waiting {wait}s before retry...")
                await asyncio.sleep(wait)
        
        print(f"  [FATAL] All {retries} attempts failed. Last error: {last_error}")
        return ""
    
    # ============================================================
    # 5. 解析辅助
    # ============================================================
    
    def _extract_probs(self, response: str) -> Dict[str, float]:
        """从 LLM 响应中提取四个选项的概率"""
        if not response or not response.strip():
            print("  [WARNING] Empty response in _extract_probs, returning default")
            return {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        
        # 调试：打印前 200 字符
        print(f"  [DEBUG] Response preview: {response[:200].replace(chr(10), ' ')}...")
        
        probs = {}
        for opt_id in ["A", "B", "C", "D"]:
            patterns = [
                rf'{opt_id}[:：]\s*(\d+)%',
                rf'{opt_id}\s*[:：]\s*(\d+\.?\d*)%',
                rf'{opt_id}\s*[:：]\s*(\d+)',
                rf'{opt_id}\s*=\s*(\d+)%',
                rf'{opt_id}\s*=\s*(\d+)',
            ]
            found = False
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    probs[opt_id] = float(match.group(1)) / 100.0
                    found = True
                    break
            if not found:
                probs[opt_id] = 0.25
        
        # 归一化
        total = sum(probs.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            for k in probs:
                probs[k] = probs[k] / total
        elif total == 0:
            probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        
        return probs
    
    def _extract_confidence(self, response: str) -> float:
        match = re.search(r'Confidence[:：]\s*(\d+)%', response, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
        return 0.5
    
    def _extract_reason(self, response: str) -> str:
        match = re.search(r'Reason[:：]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_default(self, response: str) -> Optional[str]:
        match = re.search(r'是否存在明显默认选项[:：]\s*(是|否)', response)
        if match:
            is_default = match.group(1) == "是"
            if is_default:
                default_match = re.search(r'说明[:：]\s*([A-D])', response)
                if default_match:
                    return default_match.group(1)
        return None
    
    # ============================================================
    # 6. 指标计算
    # ============================================================
    
    def _entropy(self, probs: Dict[str, float]) -> float:
        values = [p for p in probs.values() if p > 0]
        if not values:
            return 0.0
        return -sum(p * math.log2(p) for p in values)
    
    def _effective_choices(self, probs: Dict[str, float], threshold: float = 0.15) -> int:
        return sum(1 for p in probs.values() if p > threshold)
    
    def _mean(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    
    def _std(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = self._mean(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    
    def _reason_consistency(self, reason: str, state: str, top1: str) -> str:
        """
        三级 Reason Consistency：
        - Strong: 有因果词 + 引用了 State 的核心概念
        - Weak: 有因果词但未引用 State
        - Unsupported: 无因果词或理由过短/回避
        """
        if not reason or len(reason.strip()) < 8:
            return "Unsupported"
        
        causal_words = ["因为", "所以", "由于", "为了", "因此", "于是", "从而", "导致", "促使", "更适合", "有利于", "可以"]
        has_causal = any(w in reason for w in causal_words)
        
        avoidance = ["默认", "随便", "不知道", "无法判断", "都一样", "随机", "猜"]
        has_avoidance = any(w in reason for w in avoidance)
        
        if has_avoidance:
            return "Unsupported"
        
        # 检查是否引用 State 的核心概念
        state_cores = []
        for word in ["符文", "令牌", "破译", "来历", "现场", "验证", "调查", "研究"]:
            if word in state:
                state_cores.append(word)
        
        if not state_cores:
            return "Weak" if has_causal else "Unsupported"
        
        has_state_ref = any(core in reason for core in state_cores)
        
        if has_causal and has_state_ref:
            return "Strong"
        elif has_causal:
            return "Weak"
        else:
            return "Unsupported"
    
    # ============================================================
    # 7. 报告生成
    # ============================================================
    
    def generate_report(self) -> str:
        """生成完整的 Markdown 报告"""
        lines = []
        lines.append("# Phase 5.1: Manipulation Check Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().isoformat()}")
        lines.append(f"**Model**: {LLM_MODEL}")
        lines.append(f"**Samples per condition**: {SAMPLES_PER_CONDITION}")
        lines.append("")
        
        results = {
            "MP2": {"calibration": None, "gap": {}, "match": {}},
            "MP3": {"calibration": None, "gap": {}, "match": {}},
        }
        
        # 整理数据
        for d in self.raw_data["calibration"]:
            mp = d["mp_id"]
            if results[mp]["calibration"] is None:
                results[mp]["calibration"] = []
            results[mp]["calibration"].append(d)
        
        for d in self.raw_data["gap"]:
            mp = d["mp_id"]
            gt = d["gap_type"]
            if gt not in results[mp]["gap"]:
                results[mp]["gap"][gt] = []
            results[mp]["gap"][gt].append(d)
        
        for d in self.raw_data["match"]:
            mp = d["mp_id"]
            gt = d["gap_type"]
            st = d["state_type"]
            if gt not in results[mp]["match"]:
                results[mp]["match"][gt] = {}
            if st not in results[mp]["match"][gt]:
                results[mp]["match"][gt][st] = []
            results[mp]["match"][gt][st].append(d)
        
        # 生成每个 MP 的报告
        for mp_id in ["MP2", "MP3"]:
            lines.append(f"## {mp_id}")
            lines.append("")
            
            # --- Option Calibration ---
            lines.append("### Option Calibration")
            cal_data = results[mp_id]["calibration"]
            if cal_data:
                all_probs = [d["probabilities"] for d in cal_data if d["probabilities"]]
                if all_probs:
                    avg_probs = {}
                    for opt in ["A", "B", "C", "D"]:
                        avg_probs[opt] = self._mean([p[opt] for p in all_probs])
                    lines.append("| Option | Mean Probability |")
                    lines.append("|--------|------------------|")
                    for opt, val in avg_probs.items():
                        lines.append(f"| {opt} | {val:.1%} |")
                    lines.append("")
                    
                    max_prob = max(avg_probs.values())
                    default_count = sum(1 for d in cal_data if d.get("default"))
                    if max_prob < CALIBRATION_MAX_TOP1:
                        lines.append("**Result**: ✅ PASS")
                        lines.append(f"**Reason**: No option exceeded {CALIBRATION_MAX_TOP1:.0%} average probability. Max = {max_prob:.1%}.")
                    else:
                        lines.append("**Result**: ❌ FAIL")
                        lines.append(f"**Reason**: Option {max(avg_probs, key=avg_probs.get)} exceeded {CALIBRATION_MAX_TOP1:.0%} average probability.")
                    lines.append("")
            
            # --- Gap Check ---
            lines.append("### Gap Check")
            for gt in ["high", "low"]:
                data = results[mp_id]["gap"].get(gt, [])
                if not data:
                    lines.append(f"  {gt}: No data")
                    continue
                valid_data = [d for d in data if d["probabilities"]]
                if not valid_data:
                    lines.append(f"  {gt}: No valid data")
                    continue
                entropies = [self._entropy(d["probabilities"]) for d in valid_data]
                eff_choices = [self._effective_choices(d["probabilities"]) for d in valid_data]
                confidences = [d["confidence"] for d in valid_data if d.get("confidence")]
                
                lines.append(f"#### Gap: {gt.upper()}")
                lines.append(f"- **Mean Entropy**: {self._mean(entropies):.2f} ± {self._std(entropies):.2f}")
                lines.append(f"- **Effective Choices**: {self._mean(eff_choices):.1f}")
                if confidences:
                    lines.append(f"- **Confidence**: {self._mean(confidences):.1%} ± {self._std(confidences):.1%}")
                lines.append("")
            
            # Gap 判定
            high_data = [d for d in results[mp_id]["gap"].get("high", []) if d["probabilities"]]
            low_data = [d for d in results[mp_id]["gap"].get("low", []) if d["probabilities"]]
            if high_data and low_data:
                h_entropy = self._mean([self._entropy(d["probabilities"]) for d in high_data])
                l_entropy = self._mean([self._entropy(d["probabilities"]) for d in low_data])
                h_eff = self._mean([self._effective_choices(d["probabilities"]) for d in high_data])
                l_eff = self._mean([self._effective_choices(d["probabilities"]) for d in low_data])
                h_conf = self._mean([d["confidence"] for d in high_data if d.get("confidence")])
                l_conf = self._mean([d["confidence"] for d in low_data if d.get("confidence")])
                
                gap_pass = (
                    h_entropy > l_entropy and
                    h_eff >= EFFECTIVE_CHOICES_HIGH and
                    l_eff <= EFFECTIVE_CHOICES_LOW
                )
                if gap_pass:
                    lines.append("**Gap Check Result**: ✅ PASS")
                    lines.append(f"Reason: High Gap has higher entropy ({h_entropy:.2f} vs {l_entropy:.2f}) and more effective choices ({h_eff:.1f} vs {l_eff:.1f}).")
                else:
                    lines.append("**Gap Check Result**: ❌ FAIL")
                    if not (h_eff >= EFFECTIVE_CHOICES_HIGH):
                        lines.append(f"  - High Gap effective choices ({h_eff:.1f}) < {EFFECTIVE_CHOICES_HIGH}")
                    if not (l_eff <= EFFECTIVE_CHOICES_LOW):
                        lines.append(f"  - Low Gap effective choices ({l_eff:.1f}) > {EFFECTIVE_CHOICES_LOW}")
                    if not (h_entropy > l_entropy):
                        lines.append(f"  - High entropy ({h_entropy:.2f}) <= Low entropy ({l_entropy:.2f})")
                lines.append("")
            
            # --- Match Check ---
            lines.append("### Match Check")
            for gt in ["high", "low"]:
                match_data = results[mp_id]["match"].get(gt, {})
                if not match_data:
                    continue
                lines.append(f"#### Gap: {gt.upper()}")
                for st in ["correct", "wrong"]:
                    data = match_data.get(st, [])
                    valid_data = [d for d in data if d["probabilities"]]
                    if not valid_data:
                        continue
                    changed = sum(1 for d in valid_data if d["top1_changed"])
                    strong = sum(1 for d in valid_data if d["reason_consistency"] == "Strong")
                    weak = sum(1 for d in valid_data if d["reason_consistency"] == "Weak")
                    unsupported = sum(1 for d in valid_data if d["reason_consistency"] == "Unsupported")
                    n = len(valid_data)
                    
                    label = "Correct Goal" if st == "correct" else "Wrong Goal"
                    lines.append(f"**{label}**")
                    lines.append(f"- Top-1 Change Rate: {changed}/{n} ({changed/n:.0%})")
                    lines.append(f"- Reason Consistency: Strong {strong}/{n} ({strong/n:.0%}), Weak {weak}/{n} ({weak/n:.0%}), Unsupported {unsupported}/{n} ({unsupported/n:.0%})")
                lines.append("")
            
            # Match 判定
            match_results = []
            for gt in ["high", "low"]:
                match_data = results[mp_id]["match"].get(gt, {})
                if "correct" in match_data and "wrong" in match_data:
                    c_data = [d for d in match_data["correct"] if d["probabilities"]]
                    w_data = [d for d in match_data["wrong"] if d["probabilities"]]
                    if c_data and w_data:
                        c_changed = sum(1 for d in c_data if d["top1_changed"]) / len(c_data)
                        w_changed = sum(1 for d in w_data if d["top1_changed"]) / len(w_data)
                        match_results.append((gt, c_changed, w_changed))
            
            if match_results:
                all_pass = True
                for gt, c_rate, w_rate in match_results:
                    if not (c_rate >= TOP1_CHANGE_RATE_CORRECT_MIN and w_rate <= TOP1_CHANGE_RATE_WRONG_MAX):
                        all_pass = False
                        break
                if all_pass:
                    lines.append("**Match Check Result**: ✅ PASS")
                    lines.append(f"Reason: Correct Goal change rate ≥ {TOP1_CHANGE_RATE_CORRECT_MIN:.0%}, Wrong Goal ≤ {TOP1_CHANGE_RATE_WRONG_MAX:.0%}.")
                else:
                    lines.append("**Match Check Result**: ❌ FAIL")
                    for gt, c_rate, w_rate in match_results:
                        if c_rate < TOP1_CHANGE_RATE_CORRECT_MIN:
                            lines.append(f"  - {gt} Correct Goal: {c_rate:.0%} < {TOP1_CHANGE_RATE_CORRECT_MIN:.0%}")
                        if w_rate > TOP1_CHANGE_RATE_WRONG_MAX:
                            lines.append(f"  - {gt} Wrong Goal: {w_rate:.0%} > {TOP1_CHANGE_RATE_WRONG_MAX:.0%}")
                lines.append("")
        
        # ============================================================
        # Overall Summary
        # ============================================================
        lines.append("---")
        lines.append("")
        lines.append("## Overall Summary")
        lines.append("")
        lines.append("| Check | MP2 | MP3 |")
        lines.append("|-------|-----|-----|")
        
        for mp_id in ["MP2", "MP3"]:
            pass_checks = []
            
            # Calibration
            cal_data = results[mp_id]["calibration"]
            if cal_data:
                all_probs = [d["probabilities"] for d in cal_data if d["probabilities"]]
                if all_probs:
                    avg_probs = {}
                    for opt in ["A", "B", "C", "D"]:
                        avg_probs[opt] = self._mean([p[opt] for p in all_probs])
                    max_prob = max(avg_probs.values())
                    cal_pass = max_prob < CALIBRATION_MAX_TOP1
                    pass_checks.append(cal_pass)
            
            # Gap
            high_data = [d for d in results[mp_id]["gap"].get("high", []) if d["probabilities"]]
            low_data = [d for d in results[mp_id]["gap"].get("low", []) if d["probabilities"]]
            if high_data and low_data:
                h_eff = self._mean([self._effective_choices(d["probabilities"]) for d in high_data])
                l_eff = self._mean([self._effective_choices(d["probabilities"]) for d in low_data])
                h_ent = self._mean([self._entropy(d["probabilities"]) for d in high_data])
                l_ent = self._mean([self._entropy(d["probabilities"]) for d in low_data])
                gap_pass = h_ent > l_ent and h_eff >= EFFECTIVE_CHOICES_HIGH and l_eff <= EFFECTIVE_CHOICES_LOW
                pass_checks.append(gap_pass)
            
            # Match
            match_check_passed = True
            for gt in ["high", "low"]:
                match_data = results[mp_id]["match"].get(gt, {})
                if "correct" in match_data and "wrong" in match_data:
                    c_data = [d for d in match_data["correct"] if d["probabilities"]]
                    w_data = [d for d in match_data["wrong"] if d["probabilities"]]
                    if c_data and w_data:
                        c_rate = sum(1 for d in c_data if d["top1_changed"]) / len(c_data)
                        w_rate = sum(1 for d in w_data if d["top1_changed"]) / len(w_data)
                        if not (c_rate >= TOP1_CHANGE_RATE_CORRECT_MIN and w_rate <= TOP1_CHANGE_RATE_WRONG_MAX):
                            match_check_passed = False
                            break
            pass_checks.append(match_check_passed)
            
            all_pass = all(pass_checks) if pass_checks else False
            status = "✅ PASS" if all_pass else "❌ FAIL"
            lines.append(f"| {mp_id} | {status} |")
        
        lines.append("")
        
        # Diagnosis
        lines.append("### Diagnosis")
        
        # 更精确的判断
        mp2_status = "PASS" if all(pass_checks) else "FAIL"
        mp3_status = "PASS" if all(pass_checks) else "FAIL"
        
        lines.append(f"- MP2: {mp2_status}")
        lines.append(f"- MP3: {mp3_status}")
        
        if mp2_status == "PASS" and mp3_status == "PASS":
            lines.append("")
            lines.append("**Overall Decision: GO**")
            lines.append("All manipulation checks passed. Proceed to Phase 5.2 (Writer Experiment).")
        else:
            lines.append("")
            lines.append("**Overall Decision: NO-GO**")
            failed = []
            if mp2_status == "FAIL":
                failed.append("MP2")
            if mp3_status == "FAIL":
                failed.append("MP3")
            lines.append(f"Manipulation failed for: {', '.join(failed)}")
            lines.append("Revise experiment materials before Phase 5.2.")
        
        return "\n".join(lines)
    
    # ============================================================
    # 9. 主入口
    # ============================================================
    
    async def run_all(self):
        """运行所有实验"""
        print("=" * 60)
        print("Phase 5.1: Manipulation Check")
        print("=" * 60)
        print("")
        print(f"Model: {LLM_MODEL}")
        print(f"Samples per condition: {SAMPLES_PER_CONDITION}")
        print(f"Max retries: {MAX_RETRIES}")
        print(f"Request timeout: {REQUEST_TIMEOUT}s")
        print(f"Request delay: {REQUEST_DELAY}s")
        print("")
        
        # Step 0: Option Calibration
        print("Step 0: Option Calibration")
        for mp_id in ["MP2", "MP3"]:
            print(f"  {mp_id}")
            await self.calibrate(mp_id)
        print("")
        
        # Step 1: Gap Check
        print("Step 1: Gap Check")
        for mp_id in ["MP2", "MP3"]:
            for gt in ["high", "low"]:
                print(f"  {mp_id} {gt}")
                await self.run_gap(mp_id, gt)
        print("")
        
        # Step 2: Match Check
        print("Step 2: Match Check")
        for mp_id in ["MP2", "MP3"]:
            for gt in ["high", "low"]:
                for st in ["correct", "wrong"]:
                    print(f"  {mp_id} {gt} {st}")
                    await self.run_match(mp_id, gt, st)
        print("")
        
        # Step 3: Generate Report
        print("Step 3: Generating Report")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 保存原始数据
        raw_path = OUTPUT_DIR / "raw_data.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(self.raw_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Raw data saved to: {raw_path}")
        
        # 生成报告
        report = self.generate_report()
        report_path = OUTPUT_DIR / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Report saved to: {report_path}")
        
        print("")
        print("=" * 60)
        print(f"Done! Total LLM calls: {self._call_count}")
        print("=" * 60)


# ============================================================
# 运行
# ============================================================

async def main():
    check = ManipulationCheck()
    await check.run_all()

if __name__ == "__main__":
    asyncio.run(main())