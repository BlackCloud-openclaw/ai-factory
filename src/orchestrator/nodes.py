import uuid
import time
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.orchestrator.state import AgentState
from src.agents.research import ResearchAgent
from src.agents.coder import CoderAgent
from src.common.logging import setup_logging

logger = setup_logging("orchestrator.nodes")

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

# System prompt for validation
VALIDATE_SYSTEM_PROMPT = """You are a code validator. Given:
1. The original user request
2. The generated code
3. The execution result

Determine if the code satisfies the user's request.
Return JSON with "passed" (boolean), "feedback" (string), and "suggestions" (list of strings)."""


async def analyze_node(state: AgentState) -> dict[str, Any]:
    """Analyze user input to determine intent and subtasks."""
    logger.info(f"Analyzing user input: {state.user_input[:200]}...")

    messages = [
        SystemMessage(content=ANALYZE_SYSTEM_PROMPT),
        HumanMessage(content=f"User input: {state.user_input}"),
    ]
    state = state.model_copy(
        update={"current_node": "analyze", "messages": messages}
    )

    # TODO: Replace with actual LLM call once LLM integration is complete
    # For now, do simple keyword-based analysis
    intent, subtasks = _keyword_analyze(state.user_input)

    logger.info(f"Intent: {intent}, Subtasks: {subtasks}")

    return {
        "intent": intent,
        "subtasks": subtasks,
        "current_node": "analyze",
    }


async def research_node(state: AgentState) -> dict[str, Any]:
    """Run research agent to retrieve knowledge and summarize."""
    logger.info(f"Running research for subtasks: {state.subtasks}")

    research_agent = ResearchAgent()
    all_results = []

    for subtask in state.subtasks:
        result = await research_agent.run(subtask, state)
        all_results.append(result)
        state = state.model_copy(update={"messages": state.messages + [
            {"role": "assistant", "content": f"Research on '{subtask}': {result.get('summary', 'No results')}"}
        ]})

    state = state.model_copy(update={
        "research_results": all_results,
        "current_node": "research",
    })

    # Build summary from research results
    summary = _build_research_summary(all_results)
    return {
        "research_results": all_results,
        "current_node": "research",
        "messages": state.messages + [
            {"role": "assistant", "content": f"Research complete. Summary: {summary}"}
        ],
    }


async def code_node(state: AgentState) -> dict[str, Any]:
    """Run coder agent to generate and execute code."""
    logger.info(f"Running code generation for subtasks: {state.subtasks}")

    coder_agent = CoderAgent()
    code_context = {
        "research_results": state.research_results,
        "user_input": state.user_input,
        "subtasks": state.subtasks,
    }

    result = await coder_agent.run(code_context, state)

    state = state.model_copy(update={
        "code_generated": result.get("code", ""),
        "code_file_path": result.get("file_path", ""),
        "execution_result": result.get("execution_result"),
        "current_node": "code",
    })

    exec_summary = ""
    if result.get("execution_result"):
        exec_result = result["execution_result"]
        if exec_result.get("success"):
            exec_summary = f"Code executed successfully. Output: {exec_result.get('stdout', '')[:200]}"
        else:
            exec_summary = f"Code execution failed: {exec_result.get('stderr', '')[:200]}"
            state = state.model_copy(update={
                "error": exec_result.get("stderr", "Execution failed"),
                "retry_count": state.retry_count + 1,
            })

    return {
        "code_generated": result.get("code", ""),
        "code_file_path": result.get("file_path", ""),
        "execution_result": result.get("execution_result"),
        "current_node": "code",
        "messages": state.messages + [
            {"role": "assistant", "content": f"Code {'succeeded' if result.get('execution_result', {}).get('success') else 'failed'}. {exec_summary}"}
        ],
    }


async def validate_node(state: AgentState) -> dict[str, Any]:
    """Validate the final output against the user's request."""
    logger.info("Running validation")

    validation_prompt = f"""
Original request: {state.user_input}
Generated code: {state.code_generated}
Execution result: {state.execution_result}

Does the output satisfy the user's request?
"""

    messages = [
        SystemMessage(content=VALIDATE_SYSTEM_PROMPT),
        HumanMessage(content=validation_prompt),
    ]

    # TODO: Replace with actual LLM call
    # For now, use heuristic validation
    passed = _heuristic_validate(state)
    feedback = "Validation passed." if passed else "Validation needs improvement."

    state = state.model_copy(update={
        "validation_result": {"passed": passed, "feedback": feedback},
        "current_node": "validate",
    })

    if passed:
        final_answer = state.code_generated or state.final_answer
        if not final_answer:
            final_answer = f"Based on the research, here's what I found:\n\n{_build_research_summary(state.research_results)}"
        return {
            "validation_result": {"passed": True, "feedback": feedback},
            "final_answer": final_answer,
            "current_node": "validate",
            "error": None,
        }
    else:
        return {
            "validation_result": {"passed": False, "feedback": feedback},
            "error": feedback,
            "retry_count": state.retry_count + 1,
            "current_node": "validate",
        }


def _keyword_analyze(user_input: str) -> tuple[str, list[str]]:
    """Simple keyword-based intent analysis (placeholder for LLM)."""
    lower = user_input.lower()

    if any(kw in lower for kw in ["write", "code", "implement", "function", "class", "create"]):
        intent = "code_generation"
    elif any(kw in lower for kw in ["explain", "what is", "how does", "tell me", "research", "knowledge"]):
        intent = "research"
    elif any(kw in lower for kw in ["write", "code", "implement"]):
        intent = "code_generation"
    else:
        intent = "general_chat"

    subtasks = [user_input]
    return intent, subtasks


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
