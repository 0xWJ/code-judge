from enum import Enum
from typing import Literal
import uuid

from pydantic import BaseModel, Field


class Submission(BaseModel):
    sub_id: str | None = None
    type: Literal['python', 'cpp', 'math']
    options: dict[str, str] | None = None
    solution: str
    input: str | None = None
    expected_output: str

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


class WorkPayload(BaseModel):
    work_id: str | None = None
    submission: Submission | BatchSubmission = Field(..., discriminator='type')

    def model_post_init(self, __context):
        self.work_id = self.work_id or str(uuid.uuid4())
