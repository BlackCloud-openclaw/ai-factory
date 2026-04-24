# src/orchestrator/nodes.py
import uuid
import time
import json
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.orchestrator.state import AgentState
from src.agents.research import ResearchAgent
from src.agents.executor import ExecutorAgent
from src.agents.memory import MemoryAgent
from src.agents.planner import PlannerAgent
from src.agents.validator import ValidatorAgent
from src.common.logging import setup_logging

logger = setup_logging("orchestrator.nodes")

# Global memory agent instance (shared across requests)
_memory_agent = MemoryAgent()


def get_memory_agent() -> MemoryAgent:
    """Get the shared MemoryAgent instance."""
    return _memory_agent


# ============================================================================
# Helper functions (used by analyze_node, research_node, etc.)
# ============================================================================

def _keyword_analyze(user_input: str) -> tuple[str, list[str]]:
    """Simple keyword-based intent analysis (placeholder for LLM)."""
    lower = user_input.lower()

    if any(kw in lower for kw in ["write", "code", "implement", "function", "class", "create"]):
        intent = "code_generation"
    elif any(kw in lower for kw in ["explain", "what is", "how does", "tell me", "research", "knowledge"]):
        intent = "research"
    else:
        intent = "general_chat"

    subtasks = [user_input]
    return intent, subtasks


COMPLEXITY_KEYWORDS = [
    "并且", "然后", "先", "再", "接着", "最后", "同时", "还要", "另外",
    "and", "then", "also", "finally", "next", "after", "before", "while",
    "multiple", "sequence", "pipeline", "workflow"
]

def _is_complex_task(user_input: str) -> bool:
    """Determine if a task is complex based on keywords and length."""
    if len(user_input) > 200:
        return True
    lower = user_input.lower()
    return any(kw in lower for kw in COMPLEXITY_KEYWORDS)


def _build_research_summary(results: list[dict[str, Any]]) -> str:
    """Build a summary string from research results."""
    if not results:
        return "No research results available."

    summaries = []
    for r in results:
        summary = r.get("summary", r.get("content", "No content"))
        source = r.get("source", "unknown")
        summaries.append(f"[{source}]: {summary}")

    return "\n\n".join(summaries)


def _heuristic_validate(state: AgentState) -> bool:
    """Simple heuristic validation (placeholder for LLM)."""
    if state.code_generated and len(state.code_generated.strip()) > 10:
        if state.execution_result:
            return state.execution_result.get("success", False)
        return True
    if state.research_results:
        return True
    return False


# ============================================================================
# Node functions
# ============================================================================

async def load_memory_node(state: AgentState) -> dict[str, Any]:
    """Load project memory context before processing a request."""
    return await _memory_agent.run(state)   # MemoryAgent.run returns {"memory_context": ...}


async def save_memory_node(state: AgentState) -> dict[str, Any]:
    """Save important context after workflow completion."""
    project_id = state.project_id or state.metadata.get("session_id", "default")
    memory_agent = get_memory_agent()

    logger.info(f"Saving memory for project={project_id}")

    try:
        # Store the user input intent
        await memory_agent.store(
            project_id=project_id,
            key="last_intent",
            value=state.intent,
            metadata={"timestamp": time.time()},
        )

        # Store subtasks
        if state.subtasks:
            await memory_agent.store(
                project_id=project_id,
                key="last_subtasks",
                value=state.subtasks,
                metadata={"timestamp": time.time()},
            )

        # Store generated code if available
        if state.code_generated:
            await memory_agent.store(
                project_id=project_id,
                key="last_code",
                value=state.code_generated,
                metadata={
                    "timestamp": time.time(),
                    "file_path": state.code_file_path,
                },
            )

        # Store execution result summary
        if state.execution_result:
            exec_summary = {
                "success": state.execution_result.get("success", False),
                "stdout": str(state.execution_result.get("stdout", ""))[:200],
                "stderr": str(state.execution_result.get("stderr", ""))[:200],
            }
            await memory_agent.store(
                project_id=project_id,
                key="last_execution",
                value=exec_summary,
                metadata={"timestamp": time.time()},
            )

        # Append to conversation history
        conversation_entry = {
            "user_input": state.user_input,
            "intent": state.intent,
            "code_generated": bool(state.code_generated),
            "execution_success": (
                state.execution_result.get("success", False)
                if state.execution_result
                else None
            ),
            "timestamp": time.time(),
        }

        await memory_agent.append_to_memory(
            project_id=project_id,
            key="conversation_history",
            value=conversation_entry,
            max_items=50,
        )

        # Store final answer if available
        if state.final_answer:
            await memory_agent.store(
                project_id=project_id,
                key="last_answer",
                value=state.final_answer[:1000],
                metadata={"timestamp": time.time()},
            )

        logger.info(
            f"Saved memory for project={project_id}: intent={state.intent}, code={'yes' if state.code_generated else 'no'}"
        )

    except Exception as e:
        logger.error(f"Failed to save memory for project={project_id}: {e}", exc_info=True)

    # No state updates needed (memory is external)
    return {}


async def analyze_node(state: AgentState) -> dict[str, Any]:
    """Analyze user input to determine intent, subtasks, and complexity."""
    logger.info(f"Analyzing user input: {state.user_input[:200]}...")

    # Simple keyword-based analysis (can be replaced with LLM later)
    intent, subtasks = _keyword_analyze(state.user_input)
    is_complex = _is_complex_task(state.user_input)

    logger.info(f"Intent: {intent}, Subtasks: {subtasks}, is_complex: {is_complex}")

    return {
        "intent": intent,
        "subtasks": subtasks,
        "is_complex": is_complex,
        "current_node": "analyze",
    }


async def plan_node(state: AgentState) -> dict[str, Any]:
    """Generate a task plan using PlannerAgent."""
    planner = PlannerAgent()
    updates = await planner.run(state)   # returns {"task_plan": ..., "plan_id": ..., ...}
    # Ensure task_plan is present
    if "task_plan" not in updates:
        logger.warning("PlannerAgent.run did not return task_plan")
    return updates


async def scheduler_node(state: AgentState) -> dict[str, Any]:
    """Execute all subtasks from the plan using TaskScheduler."""
    # Priority: use task_plan (new) or fallback to plan (legacy)
    plan_data = getattr(state, 'task_plan', None)
    if not plan_data:
        plan_data = getattr(state, 'plan', None)
    if not plan_data:
        logger.warning("No plan data found in state for scheduler node")
        return {
            "plan_status": "no_plan",
            "subtask_results": {},
            "current_node": "scheduler",
        }

    from src.scheduler.task_scheduler import TaskScheduler
    from src.agents.planner import TaskPlan, Subtask

    # Reconstruct TaskPlan from plan_data
    subtasks_data = plan_data.get("subtasks", [])
    if not subtasks_data:
        logger.warning("No subtasks in plan, skipping scheduler")
        return {
            "plan_status": "no_subtasks",
            "subtask_results": {},
            "current_node": "scheduler",
        }

    subtasks = []
    for sd in subtasks_data:
        subtasks.append(
            Subtask(
                id=sd.get("id", f"st_{len(subtasks):03d}"),
                name=sd.get("name", sd.get("description", "")),
                description=sd.get("description", ""),
                type=sd.get("type", "code"),
                dependencies=sd.get("dependencies", []),
                required_tools=sd.get("required_tools", []),
            )
        )

    task_plan = TaskPlan(
        plan_id=plan_data.get("plan_id", f"plan_{uuid.uuid4().hex[:8]}"),
        original_request=state.user_input,
        subtasks=subtasks,
    )

    scheduler = TaskScheduler(max_concurrent=3, max_retries=2)
    task_id = await scheduler.submit_plan(task_plan)
    summary = await scheduler.run(task_id)

    subtask_results = summary.get("results", {})
    success_count = summary.get("success", 0)
    fail_count = summary.get("failed", 0)

    # Collect code outputs
    code_outputs = []
    research_outputs = []
    for st_id, result in subtask_results.items():
        if result.get("status") == "success":
            raw = result.get("result", {})
            if raw.get("type") == "code":
                code_outputs.append(raw.get("output", ""))
            elif raw.get("type") == "research":
                research_outputs.append(raw.get("output", ""))

    merged_code = "\n\n".join(code_outputs) if code_outputs else ""
    merged_research = "\n\n".join(research_outputs) if research_outputs else ""

    plan_status = "success" if fail_count == 0 else "partial"
    if success_count == 0:
        plan_status = "failed"

    logger.info(f"Scheduler completed: {success_count} success, {fail_count} failed, status={plan_status}")

    return {
        "task_id": task_id,
        "subtask_results": subtask_results,
        "plan_status": plan_status,
        "code_generated": merged_code,
        "research_results": ( [{"summary": r, "source": "scheduler"} for r in research_outputs] if research_outputs else state.research_results ),
        "execution_result": ( {"success": True, "stdout": merged_code[:500]} if merged_code and success_count > 0 else None ),
        "current_node": "scheduler",
    }


async def research_node(state: AgentState) -> dict[str, Any]:
    """Run research agent to retrieve knowledge and summarize."""
    logger.info(f"Running research for: {state.user_input[:200]}...")
    research_agent = ResearchAgent()
    result = await research_agent.run(state)   # result contains research_results, sources

    summary = _build_research_summary(result.get("research_results", []))
    return {
        "research_results": result.get("research_results", []),
        "sources": result.get("sources", []),
        "current_node": "research",
    }


async def code_node(state: AgentState) -> dict[str, Any]:
    """Run coder agent to generate and execute code."""
    logger.info(f"Running code generation for subtasks: {state.subtasks}")
    executor = ExecutorAgent()
    updates = await executor.run(state)   # returns code_generated, code_file_path, execution_result

    exec_summary = ""
    if updates.get("execution_result"):
        exec_res = updates["execution_result"]
        if exec_res.get("success"):
            exec_summary = f"Code executed successfully. Output: {exec_res.get('stdout', '')[:200]}"
        else:
            exec_summary = f"Code execution failed: {exec_res.get('stderr', '')[:200]}"

    # Ensure research_results are preserved if not overwritten
    research_results = state.research_results
    return {
        "code_generated": updates.get("code_generated", ""),
        "code_file_path": updates.get("code_file_path", ""),
        "execution_result": updates.get("execution_result"),
        "research_results": research_results,   # carry forward
        "current_node": "code",
    }


async def validate_node(state: AgentState) -> dict[str, Any]:
    """Validate the final output against the user's request using ValidatorAgent."""
    logger.info("Running validation")
    validator = ValidatorAgent()
    updates = await validator.run(state)   # returns validation_result, final_answer

    validation_result = updates.get("validation_result", {})
    passed = validation_result.get("passed", False)
    feedback = validation_result.get("feedback", "")

    logger.info(f"Validation result: passed={passed}, feedback={feedback}")

    retry_count = getattr(state, 'retry_count', 0)
    if not passed:
        retry_count += 1

    max_retries = getattr(state, 'max_retries_per_subtask', 2)
    needs_retry = retry_count < max_retries

    final_answer = updates.get("final_answer", "")
    if not final_answer and passed:
        final_answer = state.code_generated or _build_research_summary(state.research_results)

    if passed:
        return {
            "validation_result": validation_result,
            "final_answer": final_answer,
            "current_node": "validate",
            "error": None,
            "needs_retry": False,
            "retry_count": retry_count,
        }
    else:
        return {
            "validation_result": validation_result,
            "error": feedback,
            "retry_count": retry_count,
            "current_node": "validate",
            "needs_retry": needs_retry,
        }


def advance_subtask_node(state: AgentState) -> dict[str, Any]:
    """Move to the next subtask in the remaining list."""
    remaining = getattr(state, 'remaining_subtasks', []) or []
    if remaining:
        next_task = remaining[0]
        new_remaining = remaining[1:]
        current_index = getattr(state, 'current_subtask_index', 0) or 0
        return {
            "subtasks": [next_task["description"]],
            "current_subtask_index": current_index + 1,
            "current_subtask_id": next_task["id"],
            "remaining_subtasks": new_remaining,
            "validation_result": None,
            "execution_result": None,
            "needs_retry": False,
            "retry_count": 0,
        }
    return {"subtasks": []}