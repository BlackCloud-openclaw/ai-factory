"""Task scheduler with PostgreSQL persistence and dependency-aware execution.

Manages subtask lifecycle with a state machine (PENDING → RUNNING → SUCCESS/FAILED/RETRY),
concurrent execution control via asyncio.Semaphore, and dead-letter queue for permanently
failed tasks. Persists all state transitions to PostgreSQL.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import psycopg2
import psycopg2.pool

from src.agents.executor import ExecutorAgent
from src.agents.planner import TaskPlan, Subtask
from src.common.logging import setup_logging
from src.config import config

logger = setup_logging("scheduler.task_scheduler")


class TaskStatus(str, Enum):
    """Subtask execution status in the state machine."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"


@dataclass
class TaskJob:
    """Represents a single task job in the scheduler."""

    id: str = ""
    task_id: str = ""
    subtask_id: str = ""
    subtask_name: str = ""
    subtask_type: str = ""
    description: str = ""
    status: str = TaskStatus.PENDING.value
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    dependencies: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)


class TaskScheduler:
    """Scheduler that manages subtask execution with PostgreSQL persistence.

    Features:
    - Dependency-aware execution ordering
    - Concurrent execution with configurable limit (default 3)
    - Automatic retry with exponential backoff (max 2 retries)
    - Dead-letter queue for permanently failed tasks
    - Full state persistence to PostgreSQL task_jobs table
    """

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS task_jobs (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            subtask_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            result TEXT,
            error TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 2,
            subtask_name TEXT,
            subtask_type TEXT,
            description TEXT,
            dependencies TEXT DEFAULT '[]',
            required_tools TEXT DEFAULT '[]',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            started_at REAL,
            completed_at REAL
        )
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        max_concurrent: int = 3,
        max_retries: int = 2,
        executor_agent: Optional[ExecutorAgent] = None,
    ):
        self.dsn = dsn or config.postgres_dsn
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.executor_agent = executor_agent or ExecutorAgent()
        self._connection_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # In-memory state
        self.current_task_id: str = ""
        self.jobs: dict[str, TaskJob] = {}
        self.dead_letter: list[TaskJob] = []

    def _get_conn(self):
        """Get a database connection from the pool."""
        if self._connection_pool is None:
            self._connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=config.postgres_pool_size,
                dsn=self.dsn,
            )
            logger.info("Initialized PostgreSQL connection pool for task scheduler")
        return self._connection_pool.getconn()

    def _put_conn(self, conn):
        """Return a connection to the pool."""
        if self._connection_pool:
            self._connection_pool.putconn(conn)

    def _init_db(self):
        """Create the task_jobs table if it doesn't exist."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(self.CREATE_TABLE_SQL)
            conn.commit()
            logger.info("task_jobs table ready")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create task_jobs table: {e}", exc_info=True)
            raise
        finally:
            self._put_conn(conn)

    def _insert_job(self, job: TaskJob):
        """Insert a new task job into the database."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO task_jobs
                    (id, task_id, subtask_id, status, result, error, retry_count,
                     max_retries, subtask_name, subtask_type, description,
                     dependencies, required_tools, created_at, updated_at,
                     started_at, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job.id,
                    job.task_id,
                    job.subtask_id,
                    job.status,
                    job.result,
                    job.error,
                    job.retry_count,
                    job.max_retries,
                    job.subtask_name,
                    job.subtask_type,
                    job.description,
                    json.dumps(job.dependencies),
                    json.dumps(job.required_tools),
                    job.created_at,
                    job.updated_at,
                    job.started_at,
                    job.completed_at,
                ),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to insert job {job.id}: {e}", exc_info=True)
            raise
        finally:
            self._put_conn(conn)

    def _update_job(self, job: TaskJob):
        """Update an existing task job in the database."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE task_jobs
                SET status = %s, result = %s, error = %s, retry_count = %s,
                    updated_at = %s, started_at = %s, completed_at = %s
                WHERE id = %s
                """,
                (
                    job.status,
                    job.result,
                    job.error,
                    job.retry_count,
                    job.updated_at,
                    job.started_at,
                    job.completed_at,
                    job.id,
                ),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update job {job.id}: {e}", exc_info=True)
            raise
        finally:
            self._put_conn(conn)

    def _load_job(self, job_id: str) -> Optional[TaskJob]:
        """Load a task job from the database."""
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT * FROM task_jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_job(row)
        finally:
            self._put_conn(conn)

    def _row_to_job(self, row: dict) -> TaskJob:
        """Convert a database row to a TaskJob."""
        return TaskJob(
            id=row["id"],
            task_id=row["task_id"],
            subtask_id=row["subtask_id"],
            subtask_name=row.get("subtask_name", ""),
            subtask_type=row.get("subtask_type", ""),
            description=row.get("description", ""),
            status=row["status"],
            result=row.get("result"),
            error=row.get("error"),
            retry_count=row.get("retry_count", 0),
            max_retries=row.get("max_retries", 2),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
            dependencies=json.loads(row.get("dependencies", "[]")),
            required_tools=json.loads(row.get("required_tools", "[]")),
        )

    async def submit_plan(self, task_plan: TaskPlan, task_id: Optional[str] = None) -> str:
        """Submit a task plan for execution.

        Creates a TaskJob for each subtask in the plan, persists them to PostgreSQL,
        and initializes the scheduler's in-memory state.

        Args:
            task_plan: The TaskPlan containing subtasks to execute.
            task_id: Optional custom task id. Auto-generated if not provided.

        Returns:
            The task_id for this plan execution.
        """
        self._init_db()

        if not task_id:
            task_id = f"task_{uuid.uuid4().hex[:12]}"
        self.current_task_id = task_id

        now = time.time()
        for st in task_plan.subtasks:
            job_id = f"{task_id}_{st.id}"
            job = TaskJob(
                id=job_id,
                task_id=task_id,
                subtask_id=st.id,
                subtask_name=st.name,
                subtask_type=st.type,
                description=st.description,
                status=TaskStatus.PENDING.value,
                retry_count=0,
                max_retries=self.max_retries,
                created_at=now,
                updated_at=now,
                dependencies=st.dependencies,
                required_tools=st.required_tools,
            )
            self.jobs[job_id] = job
            self._insert_job(job)
            logger.debug(
                f"Submitted job {job_id} (type={st.type}, deps={st.dependencies})"
            )

        logger.info(
            f"Submitted plan '{task_plan.plan_id}' with {len(task_plan.subtasks)} jobs "
            f"under task_id={task_id}"
        )
        return task_id

    async def _execute_subtask(self, job: TaskJob) -> dict[str, Any]:
        """Execute a single subtask using the ExecutorAgent.

        Args:
            job: The TaskJob to execute.

        Returns:
            Result dict with success status, output, and metadata.
        """
        subtask_type = job.subtask_type
        description = job.description

        logger.info(
            f"Executing subtask {job.subtask_id} (type={subtask_type}): {description[:100]}"
        )

        try:
            if subtask_type == "research":
                from src.agents.research import ResearchAgent
                agent = ResearchAgent()
                result = await agent.run(description, {})
                return {
                    "success": True,
                    "type": "research",
                    "output": result.get("summary", ""),
                    "raw": result,
                }

            elif subtask_type in ("code", "write"):
                code_context = {
                    "user_input": description,
                    "research_results": [],
                    "subtasks": [description],
                }
                exec_result = await self.executor_agent.run(code_context, None)
                return {
                    "success": exec_result.get("execution_result", {}).get("success", False),
                    "type": "code",
                    "output": exec_result.get("code", ""),
                    "file_path": exec_result.get("file_path", ""),
                    "execution_result": exec_result.get("execution_result"),
                    "raw": exec_result,
                }

            elif subtask_type == "validate":
                from src.agents.validator import ValidatorAgent
                agent = ValidatorAgent()
                code = ""
                if isinstance(job.result, str):
                    code = job.result
                elif isinstance(job.result, dict):
                    code = job.result.get("output", job.result.get("code", ""))
                validation = await agent.validate(code=code, user_input=description, execution_result={})
                return {
                    "success": validation.get("passed", False),
                    "type": "validate",
                    "output": validation,
                    "raw": validation,
                }

            elif subtask_type == "plan":
                from src.agents.planner import PlannerAgent
                planner = PlannerAgent()
                plan = await planner.plan(description, {})
                return {
                    "success": True,
                    "type": "plan",
                    "output": {
                        "plan_id": plan.plan_id,
                        "subtasks": [
                            {"id": s.id, "name": s.name, "type": s.type}
                            for s in plan.subtasks
                        ],
                    },
                    "raw": plan.model_dump(),
                }

            else:
                logger.warning(f"Unknown subtask type '{subtask_type}', treating as code")
                code_context = {
                    "user_input": description,
                    "research_results": [],
                    "subtasks": [description],
                }
                exec_result = await self.executor_agent.run(code_context, None)
                return {
                    "success": exec_result.get("execution_result", {}).get("success", False),
                    "type": subtask_type,
                    "output": exec_result.get("code", ""),
                    "execution_result": exec_result.get("execution_result"),
                    "raw": exec_result,
                }

        except Exception as e:
            logger.error(f"Subtask {job.subtask_id} execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "type": subtask_type,
                "error": str(e),
            }

    async def _run_job(self, job_id: str) -> None:
        """Run a single job with semaphore-controlled concurrency and retry logic.

        State machine:
            PENDING → RUNNING → SUCCESS (on success)
            PENDING → RUNNING → PENDING (on failure, if retries remain — reset for main loop re-scan)
            PENDING → RUNNING → DEAD_LETTER (on failure, if no retries remain)

        Key change from original: retries no longer use recursion. Instead, the job
        is reset to PENDING and persisted, allowing the main `run()` loop to re-collect
        it in the next iteration with proper dependency re-evaluation.
        """
        async with self._semaphore:
            job = self.jobs.get(job_id)
            if not job or job.status != TaskStatus.PENDING.value:
                return

            # Mark as running
            job.status = TaskStatus.RUNNING.value
            job.started_at = time.time()
            job.updated_at = time.time()
            self._update_job(job)

            try:
                result = await self._execute_subtask(job)

                if result.get("success", False):
                    job.status = TaskStatus.SUCCESS.value
                    job.result = json.dumps(result)
                    job.completed_at = time.time()
                    job.updated_at = time.time()
                    self._update_job(job)
                    logger.info(
                        f"Job {job_id} completed successfully "
                        f"(took {job.completed_at - job.started_at:.1f}s)"
                    )
                else:
                    raise Exception(result.get("error", "Execution returned failure"))

            except Exception as e:
                job.retry_count += 1
                job.error = str(e)
                job.updated_at = time.time()

                if job.retry_count <= job.max_retries:
                    # Reset to PENDING so the main loop re-collects and re-scans deps
                    job.status = TaskStatus.PENDING.value
                    job.started_at = None
                    job.completed_at = None
                    self._update_job(job)
                    logger.warning(
                        f"Job {job_id} failed (attempt {job.retry_count}/{job.max_retries}), "
                        f"reset to PENDING for retry: {e}"
                    )
                    # Brief delay before next attempt
                    await asyncio.sleep(1.0 * job.retry_count)
                    # Return — main loop will re-collect this job in next iteration
                    return
                else:
                    job.status = TaskStatus.DEAD_LETTER.value
                    job.completed_at = time.time()
                    self._update_job(job)
                    self.dead_letter.append(job)
                    logger.error(
                        f"Job {job_id} permanently failed after {job.retry_count} attempts, "
                        f"moved to dead letter queue: {e}"
                    )

    async def run(self, task_id: Optional[str] = None) -> dict[str, Any]:
        tid = task_id or self.current_task_id
        if not tid:
            raise ValueError("No task_id provided and no current task")

        # 初始加载所有任务
        await self._refresh_jobs(tid)

        all_job_ids = {j.id for j in self.jobs.values() if j.task_id == tid}
        results: dict[str, dict[str, Any]] = {}

        logger.info(f"Starting execution of {len(all_job_ids)} jobs for task_id={tid}")

        while True:
            # 刷新所有任务状态（从数据库加载最新）
            await self._refresh_jobs(tid)

            # 构建成功的任务ID集合和失败的任务ID集合
            completed = {
                j.id for j in self.jobs.values()
                if j.task_id == tid and j.status == TaskStatus.SUCCESS.value
            }
            failed = {
                j.id for j in self.jobs.values()
                if j.task_id == tid and j.status in (TaskStatus.FAILED.value, TaskStatus.DEAD_LETTER.value)
            }
            running = {
                j.id for j in self.jobs.values()
                if j.task_id == tid and j.status == TaskStatus.RUNNING.value
            }

            # 检查是否有未完成的任务
            unfinished = {
                j.id for j in self.jobs.values()
                if j.task_id == tid and j.status not in (TaskStatus.SUCCESS.value, TaskStatus.FAILED.value, TaskStatus.DEAD_LETTER.value)
            }

            if not unfinished:
                # 所有任务都已完成
                break

            # 查找 ready 任务：PENDING 且依赖全部在 completed 中，且没有依赖在 failed 中
            ready = []
            for job in self.jobs.values():
                if job.task_id != tid or job.status != TaskStatus.PENDING.value:
                    continue
                # 检查依赖
                deps_met = all(dep_id in completed for dep_id in job.dependencies)
                if not deps_met:
                    # 检查是否存在依赖已经永久失败，如果是则直接标记为 DEAD_LETTER
                    dep_failed = [dep for dep in job.dependencies if dep in failed]
                    if dep_failed:
                        job.status = TaskStatus.DEAD_LETTER.value
                        job.error = f"Dependency failed: {', '.join(dep_failed)}"
                        job.completed_at = time.time()
                        job.updated_at = time.time()
                        self._update_job(job)
                        logger.warning(f"Job {job.id} marked DEAD_LETTER due to failed dependencies {dep_failed}")
                        continue
                    # 依赖未满足且未失败，跳过
                    continue
                # 依赖满足且没有失败依赖
                ready.append(job)

            if ready:
                logger.info(f"Launching {len(ready)} ready jobs for task_id={tid}")
                tasks = [asyncio.create_task(self._run_job(job.id)) for job in ready]
                await asyncio.gather(*tasks, return_exceptions=True)
                # 执行后继续循环，下一轮刷新状态
                continue

            # 没有 ready 任务但仍有未完成的任务
            if not running and unfinished:
                # 存在 pending 但无法满足依赖，且没有运行中的任务
                # 只有依赖任务已进入 DEAD_LETTER 或 FAILED 时，才标记为 DEAD_LETTER
                for job in self.jobs.values():
                    if job.task_id == tid and job.status == TaskStatus.PENDING.value and job.id in unfinished:
                        dep_failed = [dep for dep in job.dependencies if dep in failed]
                        if dep_failed:
                            job.status = TaskStatus.DEAD_LETTER.value
                            job.error = f"Dependency permanently failed: {', '.join(dep_failed)}"
                            job.completed_at = time.time()
                            job.updated_at = time.time()
                            self._update_job(job)
                            logger.warning(f"Job {job.id} marked DEAD_LETTER due to permanently failed dependencies {dep_failed}")
                continue

            # 没有 ready 任务，但有运行中的任务，等待
            logger.debug(f"No ready jobs, waiting for running tasks to complete (running: {running})")
            await asyncio.sleep(0.5)

        # 构建最终结果
        await self._refresh_jobs(tid)
        success_count = len([j for j in self.jobs.values() if j.task_id == tid and j.status == TaskStatus.SUCCESS.value])
        fail_count = len([j for j in self.jobs.values() if j.task_id == tid and j.status == TaskStatus.DEAD_LETTER.value])
        summary = {
            "task_id": tid,
            "total": len(all_job_ids),
            "success": success_count,
            "failed": fail_count,
            "results": results,
            "dead_letter_count": fail_count,
        }
        logger.info(f"Task {tid} completed: {success_count} success, {fail_count} failed")
        return summary

    def get_job_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get the current status of a job."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            return {
                "id": job.id,
                "subtask_id": job.subtask_id,
                "status": job.status,
                "retry_count": job.retry_count,
                "error": job.error,
            }
        job = self._load_job(job_id)
        if job:
            return {
                "id": job.id,
                "subtask_id": job.subtask_id,
                "status": job.status,
                "retry_count": job.retry_count,
                "error": job.error,
            }
        return None

    def reset(self):
        """Reset the scheduler state for a new plan."""
        self.jobs.clear()
        self.dead_letter.clear()
        self.current_task_id = ""
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def _refresh_jobs(self, task_id: str) -> None:
        """Reload all jobs for the given task_id from database."""
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT * FROM task_jobs WHERE task_id = %s", (task_id,))
            rows = cur.fetchall()
            for row in rows:
                job = self._row_to_job(row)
                self.jobs[job.id] = job
        finally:
            self._put_conn(conn)
