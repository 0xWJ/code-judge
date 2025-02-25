from multiprocessing import Process
import logging
import threading
from time import sleep, time
from pathlib import Path
import json
from dataclasses import asdict
import traceback

import psutil

from app.libs.executors.executor import ProcessExecuteResult
from app.model import Submission, SubmissionResult, WorkPayload
from app.libs.executors.python_executor import PythonExecutor, ScriptExecutor
from app.libs.executors.cpp_executor import CppExecutor
import app.config as app_config
from app.work_queue import connect_queue


logger = logging.getLogger(__name__)


def save_error_case(sub: Submission, result: ProcessExecuteResult | None = None, exception: Exception | None = None):
    if not app_config.ERROR_CASE_SAVE_PATH:
        return

    try:
        save_path = Path(app_config.ERROR_CASE_SAVE_PATH) / sub.sub_id
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / 'submission.json', 'w') as f:
            f.write(sub.model_dump_json(indent=True))
        with open(save_path / 'solution.txt', 'w') as f:
            f.write(sub.solution)
        if result:
            with open(save_path / 'result.json', 'w') as f:
                json.dump(asdict(result), f, indent=True)
        if exception:
            with open(save_path / 'exception.txt', 'w') as f:
                for line in traceback.format_exception(exception):
                    f.write(line)
    except Exception:
        logger.exception(f'Failed to save error case for submission {sub.sub_id}')


def executor_factory(type: str) -> ScriptExecutor:
    if type == 'python':
        return PythonExecutor(
            python_path=app_config.PYTHON_EXECUTOR_PATH,
            timeout=app_config.MAX_EXECUTION_TIME,
            memory_limit=app_config.MAX_MEMORY * 1024 * 1024,
        )
    elif type == 'cpp':
        return CppExecutor(
            compiler_path=app_config.CPP_COMPILER_PATH,
            timeout=app_config.MAX_EXECUTION_TIME,
            memory_limit=app_config.MAX_MEMORY * 1024 * 1024,
        )
    else:
        raise ValueError(f'Unsupported type: {type}')


def judge(sub: Submission):
    try:
        executor = executor_factory(sub.type)
        result = executor.execute_script(sub.solution, sub.input)
        # TODO: make this more robust
        success = result.success and result.stdout.strip() == sub.expected_output.strip()
        if not success:
            save_error_case(sub, result)
        sub_result = SubmissionResult(
            sub_id=sub.sub_id, success=success, cost=result.cost
        )
    except Exception as e:
        logger.exception(f'Worker failed to judge submission {sub.sub_id}')
        save_error_case(sub, None, e)
        sub_result = SubmissionResult(
            sub_id=sub.sub_id, success=False, cost=0
        )
    return sub_result


class Worker(Process):
    def _run_loop(self):
        redis_queue = connect_queue(False)
        while True:
            _, payload_json = redis_queue.block_pop(app_config.WORK_QUEUE_NAME)
            payload = WorkPayload.model_validate_json(payload_json)
            result = judge(payload.submission)
            result_queue_name = f'{app_config.REDIS_RESULT_PREFIX}{payload.work_id}'
            redis_queue.push(result_queue_name, result.model_dump_json())
            redis_queue.expire(result_queue_name, app_config.REDIS_RESULT_EXPIRE)

    def run(self):
        while True:
            try:
                self._run_loop()
            except Exception:
                logger.exception(f'Worker failed. Will retry in 60 seconds...')
                sleep(60)


class WorkerManager:
    def __init__(self):
        max_workers = app_config.MAX_WORKERS
        self.workers: list[Worker] = []
        logger.info(f'Starting {max_workers} workers...')
        for _ in range(max_workers):
            worker = Worker()
            worker.start()
            self.workers.append(worker)
        logger.info(f'Started {max_workers} workers')

    def run(self):
        while True:
            try:
                logger.info('Checking workers...')
                self._check_workers()
            except Exception as e:
                logger.exception(f'Check worker failed. Will retry in 60 seconds...')
            sleep(30)

    def run_background(self):
        self._check_thread = threading.Thread(target=self.run, name='worker-checker')
        self._check_thread.daemon = True
        self._check_thread.start()

    def _check_workers(self):
        failed_workers = 0
        busy_workers = 0
        hanged_workers = 0
        for i, worker in enumerate(self.workers):
            if not worker.is_alive():
                logger.error('Worker dead. Restarting...')
                worker = Worker()
                worker.start()
                self.workers[i] = worker
                failed_workers += 1
            else:
                try:
                    worker_p = psutil.Process(worker.pid)
                    is_busy = 0
                    is_hanged = 0
                    for subp in worker_p.children(recursive=True):
                        is_busy = 1
                        if subp.is_running() and time() - subp.create_time() > app_config.MAX_EXECUTION_TIME:
                            is_hanged = 1
                            logger.info(f'Worker {subp.pid} is running for {time() - subp.create_time()} seconds. Terminating...')
                            subp.kill()
                    busy_workers += is_busy
                    hanged_workers += is_hanged
                except Exception:
                    logger.exception(f'Failed to check worker {worker.pid}')

        logger.info(f'Total: {len(self.workers)}, free: {len(self.workers) - busy_workers} failed: {failed_workers}, busy: {busy_workers}, hanged: {hanged_workers}')
