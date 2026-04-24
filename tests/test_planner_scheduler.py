#!/usr/bin/env python
# test_planner_scheduler.py
import asyncio
import sys
sys.path.insert(0, '/home/data/projects/ai_factory')

from src.agents.planner import PlannerAgent
from src.scheduler.task_scheduler import TaskScheduler
from src.agents.executor import ExecutorAgent
from src.common.logging import setup_logging

logger = setup_logging("test")

async def test_planner():
    print("\n=== Test 1: Planner Agent ===\n")
    planner = PlannerAgent()
    
    request = "写一个Python函数计算斐波那契数列第n项，并执行测试n=10"
    plan = await planner.plan(request)
    
    print(f"Task ID: {plan.task_id}")
    print(f"Original request: {plan.original_request}")
    print(f"Subtask count: {len(plan.subtasks)}")
    print(f"Execution order: {plan.execution_order}")
    for st in plan.subtasks:
        print(f"  - {st.id}: {st.name} (deps: {st.dependencies}, tools: {st.required_tools})")
    return plan

async def test_scheduler():
    print("\n=== Test 2: Task Scheduler ===\n")
    scheduler = TaskScheduler(max_concurrent=2)
    
    # Create a simple plan manually
    from src.agents.planner import SubTask, TaskPlan
    from datetime import datetime
    
    subtasks = [
        SubTask(id="task_1", name="Generate code", description="写斐波那契函数", 
                required_tools=["code_generate"], dependencies=[]),
        SubTask(id="task_2", name="Execute code", description="执行n=10", 
                required_tools=["code_execute"], dependencies=["task_1"]),
    ]
    plan = TaskPlan(task_id="test_001", original_request="test", subtasks=subtasks, execution_order=["task_1", "task_2"])
    
    # Submit plan
    await scheduler.submit_plan(plan)
    
    # Run scheduler
    results = await scheduler.run()
    print(f"Results: {results}")
    return results

async def test_full_integration():
    print("\n=== Test 3: Full Integration (Planner + Scheduler) ===\n")
    planner = PlannerAgent()
    scheduler = TaskScheduler(max_concurrent=2)
    
    request = "写一个Python脚本，计算1到100的和，并打印结果"
    
    # Plan
    plan = await planner.plan(request)
    print(f"Plan created with {len(plan.subtasks)} subtasks")
    
    # Execute
    await scheduler.submit_plan(plan)
    results = await scheduler.run()
    
    print(f"\nFinal results: {results}")
    return results

async def main():
    try:
        # Test 1: Planner only
        await test_planner()
        
        # Test 2: Scheduler only (with mock executor)
        await test_scheduler()
        
        # Test 3: Full integration
        await test_full_integration()
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())