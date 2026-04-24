import uuid
import time
import json
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.orchestrator.state import AgentState
from src.agents.research import ResearchAgent
from src.agents.executor import ExecutorAgent
from src.agents.memory import MemoryAgent
from src.common.logging import setup_logging

logger = setup_logging("orchestrator.nodes")

# Global memory agent instance (shared across requests)
_memory_agent = MemoryAgent()


def get_memory_agent() -> MemoryAgent:
    """Get the shared MemoryAgent instance."""
    return _memory_agent


# System prompt for intent analysis
ANALYZE_SYSTEM_PROMPT = """You are an intent analyzer. Given a user's request:
1. Identify the primary intent (research, code_generation, both, or general_chat)
2. Break down the request into subtasks
3. Return JSON with "intent" and "subtasks" keys.

Example output:
{{
    "intent": "code_generation",
    "subtasks": ["Write a Python function", "Test the function"]
}}"""

# System prompt for memory loading guidance
LOAD_MEMORY_PROMPT = """You are loading context memory for a project.
Review the existing memories and provide a concise summary of what has been done before.
This will help inform the next steps in the workflow.

Existing memories:
{memories}

Summarize the key context from these memories that should inform the current task."""

# System prompt for memory saving guidance
SAVE_MEMORY_PROMPT = """You are saving important context after completing a task.
Extract the most important information that should be remembered for future reference.

User request: {user_input}
Generated code: {code_generated}
Execution result: {execution_result}
Research results: {research_results}

  Identify key facts to store as memory for this project."""

# System prompt for code validation
VALIDATE_SYSTEM_PROMPT = """You are a code validator. Given:
1. The original user request
2. The generated code
3. The execution result

Determine if the code satisfies the user's request.
Return JSON with "passed" (boolean), "feedback" (string), and "suggestions" (list of strings)."""


async def load_memory_node(state: AgentState) -> dict[str, Any]:
    """Load project memory context before processing a request.

    Retrieves stored memories for the project and merges them into
    the state's memory_context for use by downstream nodes.
    """
    project_id = state.project_id or state.metadata.get("session_id", "default")
    memory_agent = get_memory_agent()

    logger.info(f"Loading memory for project={project_id}")

    try:
        memories = await memory_agent.list_all(project_id)
        logger.info(f"Loaded {len(memories)} memory entries for project={project_id}")

        if memories:
            memory_summary_parts = []
            for key, value in memories.items():
                value_str = str(value)[:500]
                memory_summary_parts.append(f"- {key}: {value_str}")
            memory_summary = "\n".join(memory_summary_parts)
            logger.info(f"Memory summary: {memory_summary[:300]}...")
        else:
            memory_summary = "No previous memories found for this project."
            logger.info(f"No memories found for project={project_id}")

        return {
            "memory_context": memories,
            "project_id": project_id,
            "current_node": "load_memory",
        }

    except Exception as e:
        logger.error(
            f"Failed to load memory for project={project_id}: {e}", exc_info=True
        )
        return {
            "memory_context": {},
            "project_id": project_id,
            "current_node": "load_memory",
        }


async def save_memory_node(state: AgentState) -> dict[str, Any]:
    """Save important context after workflow completion.

    Stores key information from the workflow execution as memory
    for the project, enabling continuity across requests.
    """
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

        return {
            "current_node": "save_memory",
        }

    except Exception as e:
        logger.error(
            f"Failed to save memory for project={project_id}: {e}", exc_info=True
        )
        return {
            "current_node": "save_memory",
        }


async def analyze_node(state: AgentState) -> dict[str, Any]:
    """Analyze user input to determine intent, subtasks, and complexity."""
    logger.info(f"Analyzing user input: {state.user_input[:200]}...")

    messages = [
        SystemMessage(content=ANALYZE_SYSTEM_PROMPT),
        HumanMessage(content=f"User input: {state.user_input}"),
    ]
    state = state.model_copy(update={"current_node": "analyze", "messages": messages})

    # TODO: Replace with actual LLM call once LLM integration is complete
    # For now, do simple keyword-based analysis
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
    """Plan complex tasks using PlannerAgent to generate structured subtask plans."""
    logger.info(f"Planning complex task: {state.user_input[:200]}...")

    from src.agents.planner import PlannerAgent

    planner = PlannerAgent()
    plan = await planner.plan(state.user_input, {"memory_context": state.memory_context})

    if plan and plan.subtasks:
        first_subtask = plan.subtasks[0]
        logger.info(
            f"Plan generated: {plan.plan_id} with {len(plan.subtasks)} subtasks, "
            f"first: {first_subtask.id}"
        )
    else:
        first_subtask = None
        logger.warning("Empty plan generated, using fallback")

    first_subtask_desc = first_subtask.description if first_subtask else state.user_input
    remaining = plan.subtasks[1:] if plan and len(plan.subtasks) > 1 else []

    return {
        "plan": plan.model_dump() if plan else [],
        "subtasks": [first_subtask_desc] if first_subtask else [],
        "current_subtask_index": 0,
        "remaining_subtasks": [s.model_dump() for s in remaining],
        "current_subtask_id": first_subtask.id if first_subtask else "",
        "max_retries_per_subtask": 2,
        "retry_count": 0,
        "current_node": "plan",
    }


async def scheduler_node(state: AgentState) -> dict[str, Any]:
    """Execute all subtasks from the plan using TaskScheduler.

    Runs the TaskScheduler to process all subtasks respecting dependencies
    and concurrency limits. Merges results back into the state.
    """
    logger.info(f"Running scheduler for plan: {len(state.plan) if state.plan else 0} subtasks")

    from src.agents.planner import TaskPlan, Subtask
    from src.scheduler.task_scheduler import TaskScheduler

    # Reconstruct TaskPlan from state
    plan_data = state.plan if isinstance(state.plan, dict) else {}
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
                dependencies=sd.get("dependencies", sd.get("depends_on", [])),
                required_tools=sd.get("required_tools", []),
            )
        )

    task_plan = TaskPlan(
        plan_id=plan_data.get("plan_id", f"plan_{uuid.uuid4().hex[:8]}"),
        description=plan_data.get("description", ""),
        subtasks=subtasks,
        execution_order=plan_data.get("execution_order", [s.id for s in subtasks]),
    )

    scheduler = TaskScheduler(max_concurrent=3, max_retries=2)
    task_id = await scheduler.submit_plan(task_plan)
    summary = await scheduler.run(task_id)

    # Merge results into state
    subtask_results = summary.get("results", {})
    success_count = summary.get("success", 0)
    fail_count = summary.get("failed", 0)

    # Collect all code outputs for downstream nodes
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

    logger.info(
        f"Scheduler completed: {success_count} success, {fail_count} failed, "
        f"status={plan_status}"
    )

    return {
        "task_id": task_id,
        "subtask_results": subtask_results,
        "plan_status": plan_status,
        "code_generated": merged_code,
        "research_results": (
            [{"summary": r, "source": "scheduler"} for r in research_outputs]
            if research_outputs
            else state.research_results
        ),
        "execution_result": (
            {"success": True, "stdout": merged_code[:500]}
            if merged_code and success_count > 0
            else None
        ),
        "current_node": "scheduler",
    }


async def research_node(state: AgentState) -> dict[str, Any]:
    """Run research agent to retrieve knowledge and summarize."""
    logger.info(f"Running research for subtasks: {state.subtasks}")

    research_agent = ResearchAgent()
    all_results = []

    for subtask in state.subtasks:
        result = await research_agent.run(subtask, state)
        all_results.append(result)
        state = state.model_copy(
            update={
                "messages": state.messages
                + [
                    {
                        "role": "assistant",
                        "content": f"Research on '{subtask}': {result.get('summary', 'No results')}",
                    }
                ]
            }
        )

    state = state.model_copy(
        update={
            "research_results": all_results,
            "current_node": "research",
        }
    )

    # Build summary from research results
    summary = _build_research_summary(all_results)
    return {
        "research_results": all_results,
        "current_node": "research",
        "messages": state.messages
        + [{"role": "assistant", "content": f"Research complete. Summary: {summary}"}],
    }


async def code_node(state: AgentState) -> dict[str, Any]:
    """Run coder agent to generate and execute code."""
    logger.info(f"Running code generation for subtasks: {state.subtasks}")

    executor_agent = ExecutorAgent()
    code_context = {
        "research_results": state.research_results,
        "user_input": state.user_input,
        "subtasks": state.subtasks,
    }

    result = await executor_agent.run(code_context, state)

    state = state.model_copy(
        update={
            "code_generated": result.get("code", ""),
            "code_file_path": result.get("file_path", ""),
            "execution_result": result.get("execution_result"),
            "current_node": "code",
        }
    )

    exec_summary = ""
    if result.get("execution_result"):
        exec_result = result["execution_result"]
        if exec_result.get("success"):
            exec_summary = f"Code executed successfully. Output: {exec_result.get('stdout', '')[:200]}"
        else:
            exec_summary = (
                f"Code execution failed: {exec_result.get('stderr', '')[:200]}"
            )
            state = state.model_copy(
                update={
                    "error": exec_result.get("stderr", "Execution failed"),
                    "retry_count": state.retry_count + 1,
                }
            )

    return {
        "code_generated": result.get("code", ""),
        "code_file_path": result.get("file_path", ""),
        "execution_result": result.get("execution_result"),
        "current_node": "code",
        "messages": state.messages
        + [
            {
                "role": "assistant",
                "content": f"Code {'succeeded' if result.get('execution_result', {}).get('success') else 'failed'}. {exec_summary}",
            }
        ],
    }


async def validate_node(state: AgentState) -> dict:
    """Validate the final output against the user's request using ValidatorAgent.

    Performs:
    1. Syntax checking via py_compile
    2. LLM-based semantic validation of requirement fulfillment
    """
    logger.info("Running validation")

    from src.agents.validator import ValidatorAgent
    agent = ValidatorAgent()

    code = getattr(state, 'code_generated', '')
    user_input = getattr(state, 'user_input', '')
    exec_result = getattr(state, 'execution_result', {}) or {}

    validation_result = await agent.validate(
        code=code,
        user_input=user_input,
        execution_result=exec_result,
    )

    passed = validation_result.get("passed", False)
    feedback = validation_result.get("feedback", "Validation completed.")
    suggestions = validation_result.get("suggestions", [])

    logger.info(f"Validation result: passed={passed}, feedback={feedback}")

    retry_count = getattr(state, 'retry_count', 0)
    if not passed:
        retry_count += 1

    max_retries = getattr(state, 'max_retries_per_subtask', 2)
    needs_retry = retry_count < max_retries

    if passed:
        final_answer = code or state.final_answer
        if not final_answer:
            final_answer = f"Based on the research, here's what I found:\n\n{_build_research_summary(state.research_results)}"
        return {
            "validation_result": validation_result,
            "final_answer": final_answer,
            "current_node": "validate",
            "error": None,
            "needs_retry": False,
            "retry_count": retry_count,
        }
    else:
        logger.info(f"Validation failed, retry_count={retry_count}, max_retries={max_retries}, needs_retry={needs_retry}")
        return {
            "validation_result": validation_result,
            "error": feedback,
            "retry_count": retry_count,
            "current_node": "validate",
            "needs_retry": needs_retry,
        }


def _keyword_analyze(user_input: str) -> tuple[str, list[str]]:
    """Simple keyword-based intent analysis (placeholder for LLM)."""
    lower = user_input.lower()

    if any(
        kw in lower
        for kw in ["write", "code", "implement", "function", "class", "create"]
    ):
        intent = "code_generation"
    elif any(
        kw in lower
        for kw in ["explain", "what is", "how does", "tell me", "research", "knowledge"]
    ):
        intent = "research"
    elif any(kw in lower for kw in ["write", "code", "implement"]):
        intent = "code_generation"
    else:
        intent = "general_chat"

    subtasks = [user_input]
    return intent, subtasks


COMPLEXITY_KEYWORDS = [
    "并且",
    "然后",
    "先",
    "再",
    "接着",
    "最后",
    "同时",
    "还要",
    "另外",
    "and",
    "then",
    "also",
    "finally",
    "next",
    "after",
    "before",
    "while",
    "multiple",
    "sequence",
    "pipeline",
    "workflow",
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


def advance_subtask_node(state: AgentState) -> dict:
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


def _heuristic_validate(state: AgentState) -> bool:
    """Simple heuristic validation (placeholder for LLM)."""
    if state.code_generated and len(state.code_generated.strip()) > 10:
        if state.execution_result:
            return state.execution_result.get("success", False)
        return True
    if state.research_results:
        return True
    return False
