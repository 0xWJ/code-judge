from enum import Enum
from typing import Literal
import uuid
from time import time

from pydantic import BaseModel, Field


class Submission(BaseModel):
    sub_id: str | None = None
    type: Literal['python', 'cpp', 'math']
    options: dict[str, str] | None = None
    solution: str
    input: str | None = None
    expected_output: str | None = None

    def model_post_init(self, __context):
        self.sub_id = self.sub_id or str(uuid.uuid4())


class ResultReason(Enum):
    UNSPECIFIED = ''
    INTERNAL_ERROR = 'internal_error'
    WORKER_TIMEOUT = 'worker_timeout'
    QUEUE_TIMEOUT = 'queue_timeout'


class SubmissionResult(BaseModel):
    sub_id: str
    success: bool
    cost: float
    stdout: str | None = None
    stderr: str | None = None
    reason: ResultReason = ResultReason.UNSPECIFIED


class BatchSubmission(BaseModel):
    sub_id: str | None = None
    type: Literal['batch'] = 'batch'
    submissions: list[Submission] = Field(..., min_length=1)

    def model_post_init(self, __context):
        self.sub_id = self.sub_id or str(uuid.uuid4())


class BatchSubmissionResult(BaseModel):
    sub_id: str
    results: list[SubmissionResult]


class JudgeResult(BaseModel):
    sub_id: str
    success: bool
    cost: float
    reason: ResultReason = ResultReason.UNSPECIFIED

    @classmethod
    def from_submission_result(cls, result: SubmissionResult):
        return cls(
            sub_id=result.sub_id,
            success=result.success,
            cost=result.cost,
            reason=result.reason
        )


class BatchJudgeResult(BaseModel):
    sub_id: str
    results: list[JudgeResult]

    @classmethod
    def from_submission_result(cls, result: BatchSubmissionResult):
        return cls(
            sub_id=result.sub_id,
            results=[JudgeResult.from_submission_result(r) for r in result.results]
        )


class WorkPayload(BaseModel):
    work_id: str | None = None
    timestamp: float | None = None
    long_running: bool = False
    submission: Submission | BatchSubmission = Field(..., discriminator='type')

    def model_post_init(self, __context):
        self.work_id = self.work_id or str(uuid.uuid4())
        self.timestamp = self.timestamp or time()
