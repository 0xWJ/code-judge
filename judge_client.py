import requests
import aiohttp
import math
import asyncio
from typing import Literal
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
import logging


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
    run_success: bool
    cost: float
    stdout: str | None = None
    stderr: str | None = None
    reason: str = ''


@dataclass
class BatchSubmission:
    type: Literal['batch']
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


@dataclass
class ServerStatus:
    queue: int
    num_workers: int


def _judge_batch(url: str, submissions: list[Submission], timeout: int = 3600) -> list[SubmissionResult]:
    if not submissions:
        return []

    batch_submission = BatchSubmission(submissions=submissions, type='batch')
    response = requests.post(
        f'{url}/run/long-batch',
        json=asdict(batch_submission),
        timeout=timeout,
    )
    response.raise_for_status()
    result = BatchSubmissionResult.from_response(response.json())
    return result.results


async def _judge_batch_async(
    submissions: list[Submission],
    http: aiohttp.ClientSession,
) -> list[SubmissionResult]:
    if not submissions:
        return []

    batch_submission = BatchSubmission(submissions=submissions, type='batch')
    async with http.post(
        f'/run/long-batch',
        json=asdict(batch_submission),
    ) as response:
        response.raise_for_status()
        result = BatchSubmissionResult.from_response(await response.json())
        return result.results


class JudgeClient:
    """
    A client for the judge server.
    This client is used to send submissions to the judge server and receive results.
    """
    def __init__(self, url, *, max_batch_size=1000, max_workers=4):
        self.url = url
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers

    def get_status(self, timeout: int = 10) -> ServerStatus:
        response = requests.get(
            f'{self.url}/status',
            timeout=timeout,
        )
        response.raise_for_status()
        return ServerStatus(**response.json())

    def judge(self, submissions: list[Submission]) -> list[SubmissionResult]:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            return self._judge(executor, submissions)

    def _judge(
            self,
            executor: ProcessPoolExecutor,
            submissions: list[Submission],
    ) -> list[SubmissionResult]:
        if not submissions:
            return []

        n_sumissions = len(submissions)
        sub_ids = list(range(len(submissions)))
        results = {}

        while submissions:
            num_batches = max(math.ceil(len(submissions) / self.max_batch_size), self.max_workers)
            batch_size = math.ceil(len(submissions) / num_batches)

            logger.debug(f'Judging {len(submissions)} submissions in {num_batches} batches of {batch_size}.')

            pending_chunks = list(
                chunkify(
                    [(sub, id) for sub, id in zip(submissions, sub_ids)],
                    batch_size)
            )
            futures = [
                executor.submit(_judge_batch, self.url, [c[0] for c in chunk])
                for chunk in pending_chunks
            ]
            queue_timeouts = []
            for i, future in enumerate(futures):
                pending_chunk = pending_chunks[i]
                result = future.result()
                for (sub, sub_id), sub_result in zip(pending_chunk, result):
                    if sub_result.reason == 'queue_timeout':
                        # Retry the submission later
                        queue_timeouts.append((sub_id, sub))
                    else:
                        results[sub_id] = sub_result
                logger.debug(f'Processed {len(results)} submissions, Got {len(queue_timeouts)} timeouts in total.')

            submissions = [sub for _, sub in queue_timeouts]
            sub_ids = [sub_id for sub_id, _ in queue_timeouts]

        return [results[i] for i in range(n_sumissions)]


class BufferedJudgeClient:
    """
    A async client for the judge server that buffers submissions and sends them in batches.
    """
    def __init__(self, url, *, max_batch_size=1000, max_workers=4, timeout: int = 3600):
        self.url = url
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self._running = True
        self._submission_queue = asyncio.Queue()
        self._workers = [asyncio.create_task(self._sender_worker()) for _ in range(self.max_workers)]
        self._http = aiohttp.ClientSession(base_url=url, timeout=aiohttp.ClientTimeout(timeout))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._submission_queue.join()
        self._running = False
        await asyncio.wait(self._workers)
        await self._http.close()
        # Return False to propagate exceptions
        # in the context manager
        return False

    async def get_status(self, timeout: int = 10) -> ServerStatus:
        async with self._http.get(
            f'/status',
            timeout=aiohttp.ClientTimeout(timeout),
        ) as response:
            response.raise_for_status()
            return ServerStatus(**(await response.json()))

    async def judge(self, submissions: list[Submission]) -> list[SubmissionResult]:
        """
        Send a list of submissions to the judge server and return the results.
        """
        if not submissions:
            return []
        futures = []
        for sub in submissions:
            future = asyncio.Future()
            futures.append(future)
            await self._submission_queue.put((future, sub))

        result = await asyncio.gather(*futures)
        return result

    async def _sender_worker(self):
        while self._running:
            batch = self._get_next_batch()
            if not batch:
                await asyncio.sleep(1.0)
                continue
            submissions = [sub for _, sub in batch]
            try:
                result = await self._judge(submissions)
                for i, r in enumerate(result):
                    batch[i][0].set_result(r)
            except Exception as e:
                for i, _ in enumerate(batch):
                    batch[i][0].set_exception(e)

    def _get_next_batch(self):
        batch_submissions = []
        while len(batch_submissions) < self.max_batch_size:
            try:
                sub = self._submission_queue.get_nowait()
                batch_submissions.append(sub)
                self._submission_queue.task_done()
            except asyncio.QueueEmpty:
                break
        return batch_submissions

    async def _judge(self, submissions: list[Submission]) -> list[SubmissionResult]:
        if not submissions:
            return []

        n_submissions = len(submissions)
        sub_ids = list(range(len(submissions)))
        results = {}

        while submissions:
            logger.debug(f'Judging {len(submissions)} submissions.')
            results = await _judge_batch_async(submissions, self._http)

            queue_timeouts = []
            for sub_id, result in zip(sub_ids, results):
                if result.reason == 'queue_timeout':
                    # Retry the submission later
                    queue_timeouts.append((sub_id, submissions[sub_id]))
                else:
                    results[sub_id] = result

            logger.debug(f'Processed {len(results)} submissions, Got {len(queue_timeouts)} timeouts.')

            submissions = [sub for _, sub in queue_timeouts]
            sub_ids = [sub_id for sub_id, _ in queue_timeouts]

        return [results[i] for i in range(n_submissions)]
