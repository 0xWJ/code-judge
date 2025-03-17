import logging
from time import time
import asyncio
import uuid

import app.config as app_config
from app.libs.redis_queue import RedisQueue
from app.libs.utils import chunkify
from app.model import (
    Submission,
    SubmissionResult,
    WorkPayload,
    BatchSubmission,
    BatchSubmissionResult,
    ResultReason,
)


logger = logging.getLogger(__name__)


def _to_result(submission: Submission, start_time: float, result_json: tuple[str, bytes] | None):
    if result_json is None: # timeout
        return SubmissionResult(sub_id=submission.sub_id, success=False, cost=time() - start_time, reason=ResultReason.QUEUE_TIMEOUT)
    else:
        result = SubmissionResult.model_validate_json(result_json[1])
        if not result.success and result.cost >= app_config.MAX_EXECUTION_TIME:
            result.reason = ResultReason.WORKER_TIMEOUT
        return result


async def judge(redis_queue: RedisQueue, submission: Submission):
    start_time = time()
    try:
        payload = WorkPayload(submission=submission)
        payload_json = payload.model_dump_json()
        await redis_queue.push(app_config.REDIS_WORK_QUEUE_NAME, payload_json)
        result_queue_name = f'{app_config.REDIS_RESULT_PREFIX}{payload.work_id}'
        result_json = await redis_queue.block_pop(result_queue_name, app_config.MAX_QUEUE_WAIT_TIME)
        await redis_queue.delete(result_queue_name)
        return _to_result(submission, start_time, result_json)
    except Exception:
        logger.exception(f'Failed to judge submission {submission.sub_id}')
        return SubmissionResult(sub_id=submission.sub_id, success=False, cost=time() - start_time, reason=ResultReason.INTERNAL_ERROR)


async def _judge_batch_impl(redis_queue: RedisQueue, subs: list[Submission], long_batch=False):
    start_time = time()
    max_wait_time = app_config.LONG_BATCH_MAX_QUEUE_WAIT_TIME \
        if long_batch else app_config.MAX_QUEUE_WAIT_TIME
    batch_chunk_size = app_config.MAX_LONG_BATCH_CHUNK_SIZE \
        if long_batch else app_config.MAX_BATCH_CHUNK_SIZE
    # use a hash tag to make sure all payloads are in the same slot in redis cluster
    hash_tag = '{' + str(uuid.uuid4()) + '}'
    payloads = [WorkPayload(work_id=f'{hash_tag}:{idx}', submission=sub, long_running=long_batch) for idx, sub in enumerate(subs)]
    payload_chunks = list(chunkify(payloads, batch_chunk_size))

    async def _submit(payloads: list[WorkPayload]):
        payload_jsons = [payload.model_dump_json() for payload in payloads]
        await redis_queue.push(app_config.REDIS_WORK_QUEUE_NAME, *payload_jsons)

    async def _get_result(payloads: list[WorkPayload], max_chunk_wait_time):
        """max_chunk_wait_time <= 0 means no wait (which is different from block_pop)"""
        result_queue_names = {
            f'{app_config.REDIS_RESULT_PREFIX}{payload.work_id}': payload
            for payload in payloads
        }
        results = {}
        result_start_time = time()
        left_time = max_chunk_wait_time
        left_result_queue_names = list(result_queue_names.keys())

        while left_result_queue_names and left_time > 0:
            # try to pop all results
            # first try to pop all results in one go
            step_results = await redis_queue.pop_multi(*left_result_queue_names)
            name_results = [(k, v) for k, v in zip(left_result_queue_names, step_results) if v is not None]
            if not name_results:
                # if no results are popped, block pop the first result
                name_result = await redis_queue.block_pop(*left_result_queue_names, timeout=max_chunk_wait_time)
                if name_result is not None:
                    name_results.append((name_result[0].decode(), name_result[1]))

            if not name_results:
                # timeout, no results are ready. break the loop
                left_time = 0
                break

            for name_result in name_results:
                result_queue_name, _ = name_result
                payload = result_queue_names[result_queue_name]
                results[result_queue_name] = _to_result(payload.submission, start_time, name_result)
                left_result_queue_names.remove(result_queue_name)

            left_time = max_chunk_wait_time - int(time() - result_start_time)

        # fill non-ready work as timeout
        for result_queue_name in left_result_queue_names:
            results[result_queue_name] = _to_result(result_queue_names[result_queue_name].submission, start_time, None)

        await redis_queue.delete(*result_queue_names)
        return [results[result_queue_name] for result_queue_name in result_queue_names]

    # submit all submissions to the queue
    for chunk in payload_chunks:
        await _submit(chunk)

    results = []
    wait_start_time = time()
    for chunk in payload_chunks:
        # get all results from the queue
        left_time = max_wait_time - int(time() - wait_start_time)
        chunk_results = await _get_result(chunk, left_time)
        results.extend(chunk_results)
    return results


async def judge_batch(redis_queue: RedisQueue, batch_sub: BatchSubmission, long_batch=False):
    try:
        results = await _judge_batch_impl(redis_queue, batch_sub.submissions, long_batch)
    except Exception:
        logger.exception(f'Failed to judge batch submission {batch_sub.sub_id}')
        results=[
            SubmissionResult(
                sub_id=sub.sub_id,
                success=False,
                cost=0,
                reason=ResultReason.INTERNAL_ERROR
            ) for sub in batch_sub.submissions
        ]
    return BatchSubmissionResult(
        sub_id=batch_sub.sub_id,
        results=results
    )
