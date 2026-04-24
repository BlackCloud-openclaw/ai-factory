# src/scheduler/task_scheduler.py
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import asyncpg
from src.common.logging import setup_logging
from src.config import config
from src.agents.executor import ExecutorAgent
from src.agents.planner import TaskPlan, Subtask

logger = setup_logging("scheduler.task_scheduler")

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"

@dataclass
class TaskJob:
    id: str
    task_id: str
    subtask_id: str
    status: str
    result: Optional[str]
    error: Optional[str]
    retry_count: int
    max_retries: int
    dependencies: List[str]
    subtask_type: str
    description: str
    created_at: float
    updated_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

class TaskScheduler:
    def __init__(self, dsn: Optional[str] = None, max_concurrent: int = 3, max_retries: int = 2):
        self.dsn = dsn or config.postgres_dsn
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self._pool: Optional[asyncpg.Pool] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.current_task_id: str = ""
        self.executor = ExecutorAgent()

    async def _init_pool(self):
        if not self._pool:
            self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=10)
            await self._create_table()
            logger.info("Task scheduler PostgreSQL pool initialized")

    async def _create_table(self):
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS task_jobs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    subtask_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    retry_count INT DEFAULT 0,
                    max_retries INT DEFAULT 2,
                    dependencies TEXT,
                    subtask_type TEXT,
                    description TEXT,
                    created_at DOUBLE PRECISION,
                    updated_at DOUBLE PRECISION,
                    started_at DOUBLE PRECISION,
                    completed_at DOUBLE PRECISION
                )
            """)

    async def submit_plan(self, plan: TaskPlan, task_id: Optional[str] = None) -> str:
        await self._init_pool()
        if not task_id:
            task_id = f"task_{uuid.uuid4().hex[:12]}"
        self.current_task_id = task_id
        now = time.time()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for st in plan.subtasks:
                    job_id = f"{task_id}_{st.id}"
                    await conn.execute("""
                        INSERT INTO task_jobs (id, task_id, subtask_id, status, result, error, retry_count, max_retries,
                                               dependencies, subtask_type, description, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """, job_id, task_id, st.id, TaskStatus.PENDING.value, None, None, 0, self.max_retries,
                        json.dumps(st.dependencies), st.type, st.description, now, now)
        logger.info(f"Submitted plan '{plan.plan_id}' with {len(plan.subtasks)} jobs under task_id={task_id}")
        return task_id

    async def _refresh_jobs(self, task_id: str) -> Dict[str, TaskJob]:
        await self._init_pool()
        jobs = {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM task_jobs WHERE task_id = $1", task_id)
            for row in rows:
                job = TaskJob(
                    id=row["id"],
                    task_id=row["task_id"],
                    subtask_id=row["subtask_id"],
                    status=row["status"],
                    result=row["result"],
                    error=row["error"],
                    retry_count=row["retry_count"],
                    max_retries=row["max_retries"],
                    dependencies=json.loads(row["dependencies"]) if row["dependencies"] else [],
                    subtask_type=row["subtask_type"],
                    description=row["description"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                )
                jobs[job.id] = job
        return jobs

    async def _update_job(self, job: TaskJob):
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE task_jobs
                SET status=$1, result=$2, error=$3, retry_count=$4,
                    updated_at=$5, started_at=$6, completed_at=$7
                WHERE id=$8
            """, job.status, job.result, job.error, job.retry_count,
                job.updated_at, job.started_at, job.completed_at, job.id)

    async def _execute_subtask(self, job: TaskJob) -> Dict[str, Any]:
        """执行子任务，支持 code、validate、research 等类型。"""
        logger.info(f"Executing subtask {job.subtask_id} (type={job.subtask_type}): {job.description[:100]}")

        # 1. code / write 类型：生成并执行代码
        if job.subtask_type in ("code", "write"):
            code_context = {
                "user_input": job.description,
                "research_results": [],
                "subtasks": [job.description]
            }
            exec_result = await self.executor.run(code_context, None)
            success = exec_result.get("execution_result", {}).get("success", False)
            return {
                "success": success,
                "output": exec_result.get("code", ""),
                "execution_result": exec_result.get("execution_result")
            }

        # 2. validate 类型：从依赖的 code 任务中提取代码进行验证
        elif job.subtask_type == "validate":
            if not job.dependencies:
                return {"success": False, "error": "Validate task has no dependency"}
            dep_id = job.dependencies[0]
            async with self._pool.acquire() as conn:
                dep_row = await conn.fetchrow(
                    "SELECT result, status FROM task_jobs WHERE task_id = $1 AND subtask_id = $2",
                    job.task_id, dep_id
                )
            if not dep_row or dep_row["status"] != TaskStatus.SUCCESS.value:
                return {"success": False, "error": f"Dependency {dep_id} not successful or missing"}

            dep_result = json.loads(dep_row["result"]) if dep_row["result"] else {}
            # 兼容多种字段名提取代码
            code_to_validate = (
                dep_result.get("output") or
                dep_result.get("code") or
                dep_result.get("execution_result", {}).get("code") or
                ""
            )
            logger.info(f"Extracted code for validation (length={len(code_to_validate)}): {code_to_validate[:200]}...")
            if not code_to_validate:
                logger.warning(f"Dependency result content (first 500 chars): {dep_row['result'][:500] if dep_row['result'] else 'None'}")

            from src.agents.validator import ValidatorAgent
            validator = ValidatorAgent()
            validation = await validator.validate(
                code=code_to_validate,
                user_input=job.description,
                execution_result=dep_result.get("execution_result", {})
            )
            return {
                "success": validation.get("passed", False),
                "output": validation
            }

        # 3. research 或其他类型（可扩展）
        else:
            # 这里可以添加对 research 等类型的处理
            logger.info(f"Unhandled subtask type '{job.subtask_type}', treating as success")
            return {"success": True, "output": "Not implemented"}

    async def _run_job(self, job: TaskJob) -> None:
        async with self._semaphore:
            if job.status != TaskStatus.PENDING.value:
                return
            job.status = TaskStatus.RUNNING.value
            job.started_at = time.time()
            job.updated_at = job.started_at
            await self._update_job(job)

            try:
                result = await self._execute_subtask(job)
                if result.get("success"):
                    job.status = TaskStatus.SUCCESS.value
                    job.result = json.dumps(result)
                    job.completed_at = time.time()
                    job.updated_at = job.completed_at
                    await self._update_job(job)
                    logger.info(f"Job {job.id} completed successfully")
                else:
                    raise Exception(result.get("error", "Execution failed"))
            except Exception as e:
                job.retry_count += 1
                job.error = str(e)
                job.updated_at = time.time()
                if job.retry_count <= job.max_retries:
                    job.status = TaskStatus.PENDING.value
                    job.started_at = None
                    job.completed_at = None
                    await self._update_job(job)
                    logger.warning(f"Job {job.id} failed (retry {job.retry_count}/{job.max_retries}), will retry later")
                    await asyncio.sleep(1.0 * job.retry_count)
                else:
                    job.status = TaskStatus.DEAD_LETTER.value
                    job.completed_at = time.time()
                    await self._update_job(job)
                    logger.error(f"Job {job.id} moved to DEAD_LETTER after {job.retry_count} attempts")

    async def run(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        tid = task_id or self.current_task_id
        if not tid:
            raise ValueError("No task_id provided")

        logger.info(f"Starting execution of jobs for task_id={tid}")

        while True:
            jobs = await self._refresh_jobs(tid)
            completed = {j.subtask_id for j in jobs.values() if j.status == TaskStatus.SUCCESS.value}
            failed = {j.subtask_id for j in jobs.values() if j.status == TaskStatus.DEAD_LETTER.value}
            running = {j.subtask_id for j in jobs.values() if j.status == TaskStatus.RUNNING.value}
            pending = {j.subtask_id for j in jobs.values() if j.status == TaskStatus.PENDING.value}

            if not pending and not running:
                break

            ready = []
            for job in jobs.values():
                if job.subtask_id not in pending:
                    continue
                deps_met = all(dep in completed for dep in job.dependencies)
                dep_failed = any(dep in failed for dep in job.dependencies)
                if deps_met and not dep_failed:
                    ready.append(job)

            if ready:
                tasks = [asyncio.create_task(self._run_job(job)) for job in ready]
                await asyncio.gather(*tasks, return_exceptions=True)
                continue

            if running:
                await asyncio.sleep(0.5)
                continue

            if pending:
                for job in jobs.values():
                    if job.subtask_id in pending:
                        job.status = TaskStatus.DEAD_LETTER.value
                        job.error = "Deadlock: dependencies never satisfied"
                        job.completed_at = time.time()
                        job.updated_at = job.completed_at
                        await self._update_job(job)
                continue

        final_jobs = await self._refresh_jobs(tid)
        success_count = sum(1 for j in final_jobs.values() if j.status == TaskStatus.SUCCESS.value)
        fail_count = sum(1 for j in final_jobs.values() if j.status == TaskStatus.DEAD_LETTER.value)
        logger.info(f"Task {tid} completed: {success_count} success, {fail_count} failed")
        return {
            "task_id": tid,
            "total": len(final_jobs),
            "success": success_count,
            "failed": fail_count,
        }