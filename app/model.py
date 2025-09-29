from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse
import uuid
from time import time

from pydantic import BaseModel, Field


class Submission(BaseModel):
    sub_id: str | None = None
    type: Literal['python', 'cpp', 'math']
    # can be
    # `{session_type}:new`: start a session
    # `{session_type}:{machine_id}/{pid}`: resume a session
    # `{session_type}:{machine_id}/{pid}#close`: close a session
    # `{session_type}:` can be omitted
    # We have builtin support for jupyter
    # To use it, set session_type to 'jupyter'
    # 1. if session is 'new', the solution is ignored, and a new jupyter session is created
    # 2. if session is '{machine_id}/{pid}', the solution is executed in the existing jupyter session
    # 3. if session is '{machine_id}/{pid}#close', the solution is ignored, and the existing jupyter session is closed
    session: Literal['new'] | str | None = None
    options: dict[str, str] | None = None
    # the code to be executed
    solution: str
    input: str | None = None
    expected_output: str | None = None

    def model_post_init(self, __context):
        self.sub_id = self.sub_id or str(uuid.uuid4())

    def parse_session(self):
        session_parts = urlparse(self.session)
        session_type = session_parts.scheme
        session_id = session_parts.path
        session_action = session_parts.fragment
        return session_type, session_id, session_action


class ResultReason(Enum):
    UNSPECIFIED = ''
    INTERNAL_ERROR = 'internal_error'
    WORKER_TIMEOUT = 'worker_timeout'
    QUEUE_TIMEOUT = 'queue_timeout'
    INVALID_INPUT = 'invalid_input'


class SubmissionResult(BaseModel):
    sub_id: str
    success: bool         # Indicates if the submission was successful (run_success is True and output matches)
    run_success: bool     # Indicates if the submission ran successfully (no internal error and exit code 0)
    cost: float
    stdout: str | None = None
    stderr: str | None = None
    session: str | None = None
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
    run_success: bool
    cost: float
    session: str | None = None
    reason: ResultReason = ResultReason.UNSPECIFIED

    @classmethod
    def from_submission_result(cls, result: SubmissionResult):
        return cls(
            sub_id=result.sub_id,
            success=result.success,
            run_success=result.run_success,
            cost=result.cost,
            session=result.session,
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
    submission: Submission

    def model_post_init(self, __context):
        self.work_id = self.work_id or str(uuid.uuid4())
        self.timestamp = self.timestamp or time()
