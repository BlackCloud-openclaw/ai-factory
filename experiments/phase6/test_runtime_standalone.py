#!/usr/bin/env python3
"""
独立 Runtime 测试脚本 - 不依赖生产环境
直接调用 LLM 生成文本，然后用 Runtime 完整链路分析并修订

使用方式:
    python test_runtime_standalone.py                    # 使用内置示例文本
    python test_runtime_standalone.py --llm              # 调用真实 LLM 生成
    python test_runtime_standalone.py --input story.txt  # 从文件读取
"""

import sys
import os

# 将项目根目录添加到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入 Runtime 核心组件
from src.runtime.observation_compiler import ObservationCompiler, ObservationIR
from src.runtime.validator import Validator, ComplianceReport
from src.runtime.edit_compiler import EditCompiler, EditPlan, EditOperation
from src.runtime.patch_renderer import PatchRenderer, RenderedPatch

import json
import re
import time
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

# ... 后续代码保持不变 ...

# ============================================================
# 1. 配置
# ============================================================

# 层目标配置
DEFAULT_LAYER_TARGETS = {
    "reasoning": "enhanced",
    "justification": "enhanced",
    "construction": "enhanced",
    "prediction": "enhanced"
}

# 示例文本（用于无 LLM 时的快速测试）
EXAMPLE_TEXT = """林逸的指节在袖中捏紧又松开三次。那封泛黄的密信在怀中发烫，墨迹晕染处仍能辨认出"天机阁地底第七重"的批注——正是师兄失踪前夜用特殊药水写下的。

"你留下的信里说'血色月华'是假象。"林逸听见自己嗓音发涩，"可那天值夜的弟子明明看见红月亮贯穿三更天。"

风卷起几片零落桃花，落在那人玄色衣摆上。他忽然想起师兄被除名那日，宗门大殿外也是这样漫天花雨。"""


# ============================================================
# 2. LLM 调用（可选，需要 API 服务运行）
# ============================================================

def call_llm(prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    调用本地 LLM API 生成文本
    需要先启动: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    """
    try:
        import requests
        
        # 检查服务是否可用
        try:
            resp = requests.get("http://localhost:8000/health", timeout=2)
            if resp.status_code != 200:
                print("⚠️  API 服务未就绪，跳过 LLM 生成")
                return None
        except:
            print("⚠️  无法连接到 API 服务 (http://localhost:8000)")
            print("   请先启动: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
            return None
        
        # 发送生成请求
        payload = {
            "user_input": prompt,
            "task_type": "scene_plan",
            "novel_id": "runtime_test",
            "resume": False
        }
        
        resp = requests.post("http://localhost:8000/api/v1/resume", json=payload, timeout=30)
        if resp.status_code != 200:
            print(f"⚠️  API 请求失败: {resp.status_code}")
            return None
        
        task_id = resp.json().get("task_id")
        if not task_id:
            print("⚠️  未获取到 task_id")
            return None
        
        # 轮询等待结果
        print(f"⏳ 等待 LLM 生成 (task_id={task_id})...")
        for attempt in range(60):
            time.sleep(2)
            task_resp = requests.get(f"http://localhost:8000/api/v1/task/{task_id}", timeout=10)
            if task_resp.status_code != 200:
                continue
            
            data = task_resp.json()
            status = data.get("status")
            
            if status == "success":
                content = data.get("result", {}).get("content", "")
                if content:
                    print(f"✅ LLM 生成成功 ({len(content)} 字符)")
                    return content
                else:
                    print("⚠️  生成内容为空")
                    return None
            
            elif status == "failed":
                print(f"❌ LLM 生成失败: {data.get('error')}")
                return None
            
            # 显示进度
            progress = data.get("progress", 0)
            if progress > 0 and attempt % 5 == 0:
                print(f"  进度: {progress}%")
        
        print("⚠️  LLM 生成超时")
        return None
        
    except ImportError:
        print("⚠️  requests 库未安装，无法调用 LLM")
        return None
    except Exception as e:
        print(f"⚠️  LLM 调用异常: {e}")
        return None


def generate_test_content_with_llm() -> Optional[str]:
    """使用 LLM 生成测试文本"""
    prompt = """续写一段玄幻小说场景（300-500字）：

林逸收到一封神秘密信，信中提到了"天机阁地底第七重"。他意识到这与他师兄的失踪有关，决定前往调查。

要求：
- 展现林逸的推理过程
- 包含心理活动
- 有环境描写
- 语言流畅自然"""

    print("\n" + "=" * 60)
    print("调用 LLM 生成测试文本")
    print("=" * 60)
    
    return call_llm(prompt)


# ============================================================
# 3. Runtime 分析核心
# ============================================================

class RuntimeTester:
    """Runtime 完整链路测试器"""
    
    def __init__(self, layer_targets: Dict[str, str] = None):
        self.layer_targets = layer_targets or DEFAULT_LAYER_TARGETS.copy()
        self.obs_compiler = ObservationCompiler()
        self.validator = Validator()
        self.edit_compiler = EditCompiler()
        self.patch_renderer = PatchRenderer()
    
    def analyze(self, text: str, label: str = "文本") -> Dict[str, Any]:
        """运行完整的 Runtime 分析"""
        
        # 1. ObservationCompiler → ObservationIR
        ir = self.obs_compiler.compile(text)
        
        # 2. Validator → ComplianceReport
        report = self.validator.validate(ir, self.layer_targets)
        
        # 3. 构建结果
        result = {
            "text": text,
            "char_count": len(text),
            "sentence_count": len(ir.sentences),
            "pattern_count": len(ir.patterns),
            "pattern_types": list(set(p.pattern_type for p in ir.patterns)),
            "pattern_samples": [f"{p.text}({p.pattern_type})" for p in ir.patterns[:10]],
            "compliance": report.overall_compliance,
            "layer_results": [],
            "non_compliant_layers": [],
            "ir_hash": ir.source_hash[:16],
            "sentences": [s.text for s in ir.sentences[:10]],  # 仅前10句
            "all_sentences": [s.text for s in ir.sentences],
        }
        
        for layer_result in report.layer_results:
            layer_info = {
                "layer": layer_result.layer,
                "target_level": layer_result.target_level,
                "compliant": layer_result.compliant,
                "evidence_count": len(layer_result.evidence_list),
                "evidence": []
            }
            
            if not layer_result.compliant:
                result["non_compliant_layers"].append(layer_result.layer)
                for ev in layer_result.evidence_list:
                    layer_info["evidence"].append({
                        "anchor_sentence_id": ev.anchor_sentence_id,
                        "present_patterns": ev.present_patterns,
                        "missing_pattern_types": ev.missing_pattern_types
                    })
            
            result["layer_results"].append(layer_info)
        
        result["ir"] = ir
        result["report"] = report
        result["label"] = label
        
        return result
    
    def plan_revision(self, analysis_result: Dict[str, Any]) -> Optional[EditPlan]:
        """生成修订计划"""
        ir = analysis_result.get("ir")
        report = analysis_result.get("report")
        
        if not ir or not report:
            return None
        
        # 如果已经全合规，无需修订
        if report.overall_compliance >= 1.0:
            return None
        
        # EditCompiler 生成 EditPlan
        plan = self.edit_compiler.compile(ir, report, diagnosis_id="D_test")
        
        return plan
    
    def render_patch(self, plan: EditPlan, ir: ObservationIR) -> RenderedPatch:
        """渲染修订 Prompt"""
        return self.patch_renderer.render(plan, ir)


# ============================================================
# 4. 报告生成
# ============================================================

def print_analysis_report(result: Dict[str, Any]):
    """打印分析报告"""
    print("\n" + "=" * 70)
    print(f"📊 Runtime 分析报告: {result.get('label', '文本')}")
    print("=" * 70)
    
    # 基本统计
    print(f"\n📈 基本统计:")
    print(f"  字符数: {result['char_count']}")
    print(f"  句子数: {result['sentence_count']}")
    print(f"  Patterns: {result['pattern_count']}")
    print(f"  类型: {', '.join(result['pattern_types']) if result['pattern_types'] else '(无)'}")
    
    # 合规率
    compliance = result['compliance']
    if compliance >= 1.0:
        print(f"\n✅ 合规率: {compliance:.2f} (全部合规)")
    elif compliance >= 0.75:
        print(f"\n🟡 合规率: {compliance:.2f} (部分合规)")
    else:
        print(f"\n🔴 合规率: {compliance:.2f} (需要修订)")
    
    # 各层详情
    print(f"\n📋 各层详情:")
    for lr in result['layer_results']:
        status = "✅" if lr['compliant'] else "❌"
        print(f"  {lr['layer']}: {status} (target={lr['target_level']})")
        if not lr['compliant'] and lr['evidence']:
            for ev in lr['evidence']:
                print(f"     缺失: {', '.join(ev['missing_pattern_types'])}")
    
    # 前 5 句
    print(f"\n📝 前 5 句:")
    for i, sent in enumerate(result.get('sentences', [])[:5]):
        print(f"  {i+1}. {sent[:60]}...")
    
    # 前 10 个 Pattern
    if result.get('pattern_samples'):
        print(f"\n🔍 前 10 个 Pattern:")
        for p in result['pattern_samples'][:10]:
            print(f"  {p}")
    
    # 非合规层
    if result['non_compliant_layers']:
        print(f"\n⚠️  非合规层: {', '.join(result['non_compliant_layers'])}")
    else:
        print(f"\n✅ 所有层合规")


def print_revision_report(plan: Optional[EditPlan], ir: Optional[ObservationIR]):
    """打印修订报告"""
    if not plan or not plan.actions:
        print("\n" + "=" * 70)
        print("📝 修订计划: 无需修订 (已合规)")
        print("=" * 70)
        return
    
    print("\n" + "=" * 70)
    print("📝 修订计划")
    print("=" * 70)
    
    print(f"\nActions: {len(plan.actions)}")
    for i, action in enumerate(plan.actions):
        print(f"\n  [{i+1}] {action.operation.value}")
        print(f"      Anchor: {action.anchor_sentence_id}")
        print(f"      Payload: {action.payload_type}")
        if action.reference_pattern_id:
            print(f"      Reference: {action.reference_pattern_id}")
        if action.preserve_sentence_ids:
            print(f"      Preserve Sentences: {action.preserve_sentence_ids}")
    
    # 显示渲染后的 Prompt 预览（如果有 IR）
    if ir and plan:
        renderer = PatchRenderer()
        rendered = renderer.render(plan, ir)
        print(f"\n📄 修订指令预览 (前 300 字符):")
        print(rendered.full_prompt[:300] + "...\n")


# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="独立 Runtime 测试")
    parser.add_argument("--llm", action="store_true", help="使用真实 LLM 生成测试文本")
    parser.add_argument("--input", "-i", type=str, help="从文件读取文本")
    parser.add_argument("--text", "-t", type=str, help="直接指定文本")
    parser.add_argument("--no-revision", action="store_true", help="不生成修订计划")
    parser.add_argument("--save", "-s", type=str, help="保存结果到文件")
    args = parser.parse_args()

    print("=" * 70)
    print("🧪 Runtime 独立测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # ---- 获取测试文本 ----
    text = None
    
    if args.input:
        # 从文件读取
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"\n📂 从文件读取: {args.input} ({len(text)} 字符)")
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return 1
    
    elif args.text:
        text = args.text
        print(f"\n📝 使用指定文本 ({len(text)} 字符)")
    
    elif args.llm:
        text = generate_test_content_with_llm()
        if not text:
            print("\n⚠️  LLM 生成失败，使用示例文本")
            text = EXAMPLE_TEXT
    
    else:
        print("\n📝 使用示例文本 (--llm 调用真实 LLM，--input 读取文件)")
        text = EXAMPLE_TEXT
    
    if not text or len(text) < 10:
        print("❌ 文本为空或太短")
        return 1
    
    # ---- 初始化 Runtime 测试器 ----
    tester = RuntimeTester()
    
    # ---- 运行分析 ----
    print(f"\n🔍 运行 Runtime 分析...")
    result = tester.analyze(text, label="测试文本")
    print_analysis_report(result)
    
    # ---- 生成修订计划 ----
    if not args.no_revision:
        print(f"\n🔧 生成修订计划...")
        plan = tester.plan_revision(result)
        print_revision_report(plan, result.get("ir"))
    
    # ---- 保存结果 ----
    if args.save:
        # 清理不可序列化对象
        save_result = {k: v for k, v in result.items() 
                      if k not in ["ir", "report", "text", "all_sentences"]}
        save_result["text_preview"] = result["text"][:500]
        save_result["sentence_count"] = result["sentence_count"]
        save_result["all_sentences_count"] = len(result["all_sentences"])
        
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(save_result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存: {args.save}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())