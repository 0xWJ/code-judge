import requests
import random
import logging
from time import sleep
from typing import Literal
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor


logger = logging.getLogger(__name__)


def chunkify(iterable, size):
    """Yield successive chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


@dataclass
class Submission:
    type: Literal['python', 'cpp']
    solution: str
    input: str | None = None
    expected_output: str | None = None


@dataclass
class SubmissionResult:
    sub_id: str
    success: bool
    cost: float
    stdout: str | None = None
    stderr: str | None = None
    reason: str = ''


@dataclass
class BatchSubmission:
    type: Literal['batch'] = 'batch'
    submissions: list[Submission]


@dataclass
class BatchSubmissionResult:
    sub_id: str
    results: list[SubmissionResult]

    @classmethod
    def from_response(cls, response: dict):
        return cls(
            sub_id=response['sub_id'],
            results=[SubmissionResult(**result) for result in response['results']]
        )


def _judge_batch(url: str, submissions: list[Submission], timeout: int) -> list[SubmissionResult]:
    if not submissions:
        return []

    batch_submission = BatchSubmission(submissions=submissions)
    try:
        response = requests.post(
            f'{url}/judge/batch',
            json=asdict(batch_submission),
            timeout=timeout,
        )
        response.raise_for_status()
        result = BatchSubmissionResult.from_response(response.json())
        n_queue_timeouts = sum(
            1 for r in result.results if r.reason == 'queue_timeout'
        )
        if n_queue_timeouts > 0:
            # TODO: Implement exponential backoff?
            sleep_time = min(60, random.random() * n_queue_timeouts * n_queue_timeouts / len(submissions) * 20)
            logger.warning(f'Got {n_queue_timeouts} of {len(submissions)} queue timeouts. '
                           f'Will sleep for {sleep_time} seconds to reduce concurrency.')
            sleep(sleep_time)
        return result.results
    except Exception as e:
        logger.exception(f'Failed to judge batch submission')
        return [
            SubmissionResult(sub_id='', success=False, cost=0, reason='internal_error')
            for _ in submissions
        ]


class JudgeClient:
    def __init__(self, url, *, batch_size=10, timeout=20, max_concurrent=10):
        self.url = url
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.executor = ProcessPoolExecutor(max_workers=max_concurrent)

    def judge(
            self,
            submissions: list[Submission],
    ) -> list[SubmissionResult]:
        if not submissions:
            return []

        sub_ids = list(range(len(submissions)))
        results = {}

        while submissions:
            pending_chunks = list(
                chunkify(
                    [(sub, id) for sub, id in zip(submissions, sub_ids)],
                    self.batch_size)
            )
            futures = [
                self.executor.submit(_judge_batch, self.url, [c[0] for c in chunk], self.timeout)
                for chunk in pending_chunks
            ]
            queue_timeouts = []
            for i, future in enumerate(futures):
                pending_chunk = pending_chunks[i]
                result = future.result()
                for (sub, sub_id), sub_result in zip(pending_chunk, result):
                    if sub_result.reason == 'queue_timeout':
                        # Retry the submission
                        queue_timeouts.append((sub_id, sub))
                    else:
                        results[sub_id] = sub_result

            submissions = [sub for _, sub in queue_timeouts]
            sub_ids = [sub_id for sub_id, _ in queue_timeouts]

        return [results[i] for i in range(len(submissions))]
