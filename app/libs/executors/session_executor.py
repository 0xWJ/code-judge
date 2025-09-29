from typing import TYPE_CHECKING, TypedDict
import json
import shlex
from urllib.parse import urlparse

import psutil

import app.config as app_config
from ..redis_queue import RedisQueue
from .python_executor import PythonExecutor
from . import session_script


class SessionInfo(TypedDict):
    pid: int
    create_time: float
    stdout: str
    stderr: str
    # when the session is created
    # this flag is set to True
    # when the session is closed,
    # this flag is set to False
    active: bool


class SessionRunner:
    def gen_start(self) -> str:
        """
        Generate the script to start a new session.
        """
        ...

    def gen_run(self, session: SessionInfo, script: str) -> str:
        """
        Generate the script to run code in an existing session.
        """
        ...

    def gen_stop(self, session: SessionInfo) -> str:
        """
        Generate the script to stop an existing session.
        """
        ...


class JupyterSessionRunner(SessionRunner):
    start_script_template = """
import time
import os
from jupyter_client import KernelManager

os.chdir('/')  # change to a safe directory

km = KernelManager()
km.start_kernel()

kc = km.client()
kc.start_channels()

# wait for the kernel to be ready
def wait_for_kernel_ready(kc, timeout=10, poll_interval=0.5):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            msg_id = kc.execute("1 + 1", silent=True, user_expressions={{}}, allow_stdin=False)
            reply = kc.get_shell_msg(timeout=1)
            content = reply['content']
            if content['status'] == 'ok':
                return True
        except Exception as e:
            continue
        time.sleep(poll_interval)
    raise TimeoutError

try:
    wait_for_kernel_ready(kc, {timeout})
except TimeoutError:
    km.shutdown_kernel()
    raise

connection_info = km.get_connection_info()
print(connection_info, flush=True)
print('{ready_mark}', flush=True)

try:
    while True:
        time.sleep(1)
        if not km.is_alive():
            break
except KeyboardInterrupt:
    km.shutdown_kernel()
"""
    run_script_template = """
import json
import os
import sys
from jupyter_client import BlockingKernelClient


def execute_code_in_existing_kernel(code):
    kc = BlockingKernelClient()
    kc.load_connection_info({connection_info})

    if not kc.is_alive():
        print("Kernel is not alive", file=sys.stderr)
        exit(1)

    kc.start_channels()

    try:
        msg_id = kc.execute(code)

        while True:
            msg = kc.get_iopub_msg(timeout={timeout})
            if msg['parent_header'].get('msg_id') == msg_id:
                if msg['msg_type'] == 'status':
                    if msg['content']['execution_state'] == 'idle':
                        break
                elif msg['msg_type'] == 'stream':
                    if msg['content']['name'] == 'stdout':
                        print(msg['content']['text'])
                    else:
                        print(msg['content']['text'], file=sys.stderr)
                elif msg['msg_type'] == 'execute_result':
                    # ignore result.
                    pass
                elif msg['msg_type'] == 'error':
                    print(msg['content']['ename'], file=sys.stderr)
                    print(msg['content']['evalue'], file=sys.stderr)
                    print(msg['content']['traceback'], file=sys.stderr)
    finally:
        kc.stop_channels()
if '{script}':
    script = '{script}'
else:
    with open('{script_path}', 'r') as f:
        script = f.read()

execute_code_in_existing_kernel(script)
"""

    def gen_start(self, timeout) -> str:
        return self.start_script_template.format(
            ready_mark=session_script.READY_MARK,
            timeout=timeout - 1  # leave 1 second for the overhead
        )

    def gen_run(self, session: SessionInfo, script_path: str, timeout: int) -> str:
        return self.run_script_template.format(
            connection_info=session['stdout'],
            script='',
            script_path=script_path,
            timeout=timeout - 1  # leave 1 second for the overhead
        )

    def gen_stop(self, session: SessionInfo, timeout: int) -> str:
        return self.run_script_template.format(
            connection_info=session['stdout'],
            script='exit(0)',
            script_path='',
            timeout=timeout - 1  # leave 1 second for the overhead
        )


class SessionExecutor(PythonExecutor):
    def __init__(self, redis_queue: RedisQueue, session_initiator_cl: str, run_cl: str, session: str, timeout: int = None, memory_limit: int = None):
        self.redis = redis_queue
        self.session = session

        session_parts = urlparse(self.session)
        self.session_type = session_parts.scheme
        self.session_id = session_parts.path
        self.session_action = session_parts.fragment

        self.start_session_cl = shlex.split(
            session_initiator_cl.format(
                source=shlex.quote(session_script.__file__)
            )
        )
        self.runners ={
            'jupyter': JupyterSessionRunner(),
        }
        super().__init__(run_cl, timeout, memory_limit)

    def setup_command(self, tmp_path: str, script: str):
        runner = self.runners.get(self.session_type)
        source_path = f"{tmp_path}/source.py"
        cell_source_path = f"{tmp_path}/cell_source.py"

        with open(cell_source_path, mode='w') as f:
            f.write(script)
            f.flush()

        if self.session_id == 'new':
            with open(source_path, mode='w') as f:
                f.write(script if runner is None else runner.gen_start(timeout=self.timeout))
                f.flush()
            yield self.start_session_cl + shlex.split(self.run_cl.format(
                source=shlex.quote(source_path),
                workdir=shlex.quote(str(tmp_path))
            ))
        else:
            session_info_json = self.redis.dict.get(
                app_config.REDIS_SESSION_MACHINE_SESSIONS_NAME, self.session_id
            )
            print(session_info_json)
            if not session_info_json:
                raise ValueError("Session not found")

            session_info: SessionInfo = json.loads(session_info_json)
            if not session_info['active']:
                raise ValueError("Session is not active")

            pid = session_info['pid']
            session_process = psutil.Process(pid)  # check if the process is alive
            if not session_process.is_running() or session_process.create_time() != session_info['create_time']:
                raise ValueError("Session process is not running")
            print(session_process)
            if self.session_action == 'close':
                script_to_run = runner.gen_stop(session_info, timeout=self.timeout) if runner else script
            else:
                script_to_run = runner.gen_run(session_info, cell_source_path, timeout=self.timeout) if runner else script
            yield from super().setup_command(tmp_path, script_to_run)
            # print(script_to_run)
            # with open(source_path, mode='w') as f:
            #     f.write(script_to_run)
            #     f.flush()

            # yield shlex.split(self.run_cl.format(
            #     source=shlex.quote(source_path),
            #     workdir=shlex.quote(str(tmp_path))
            # ))

    def process_result(self, result):
        result = super().process_result(result)
        if self.session_id == 'new':
            if session_script.INFO_MARK in result.stdout:
                result.stdout, meta_info = result.stdout.split(session_script.INFO_MARK, 2)
                session_info: SessionInfo = json.loads(meta_info)
                session_info['stdout'] = result.stdout[:app_config.MAX_STDOUT_ERROR_LENGTH] if result.stdout else ''
                session_info['stderr'] = result.stderr[:app_config.MAX_STDOUT_ERROR_LENGTH] if result.stderr else ''
                session_info['active'] = True

                session_id = f"{app_config.MACHINE_ID}/{session_info['pid']}"
                self.redis.dict.set(
                    app_config.REDIS_SESSION_MACHINE_SESSIONS_NAME, session_id, json.dumps(session_info)
                )
                # self.redis.dict.expire(
                #     app_config.REDIS_SESSION_MACHINE_SESSIONS_NAME,
                #     app_config.REDIS_SESSION_EXPIRE,
                #     session_id
                # )

                result.session = session_id
                if self.session_type:
                    result.session = f'{self.session_type}:{result.session}'
            else:
                # failed to start the session
                if result.exit_code == 0:
                    result.exit_code = -1
                    result.stderr += "\nFailed to detect session ready from stdout.\n"

        elif self.session_action == 'close':
            session_info_json = self.redis.dict.get(
                app_config.REDIS_SESSION_MACHINE_SESSIONS_NAME, self.session_id
            )
            if not session_info_json:
                raise ValueError("Session not found")

            session_info: SessionInfo = json.loads(session_info_json)
            session_info['active'] = False

            pid = session_info['pid']
            session_process = psutil.Process(pid)  # check if the process is alive
            if not session_process.is_running() or session_process.create_time() != session_info['create_time']:
                # session process is killed.
                self.redis.dict.delete(
                    app_config.REDIS_SESSION_MACHINE_SESSIONS_NAME, self.session_id
                )
            else:
                # mark the session as inactive
                # worker manager will clean up inactive sessions
                self.redis.dict.set(
                    app_config.REDIS_SESSION_MACHINE_SESSIONS_NAME, self.session_id, json.dumps(session_info)
                )

        if self.session_id and result.session is None:
            result.session = self.session_id
        return result
