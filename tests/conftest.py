import pytest
import fakeredis
import os
import sys
import subprocess
import multiprocessing
from time import sleep

REDIS_PORT = 6388

if os.environ.get('REDIS_URI') is None:
    os.environ['REDIS_URI'] = f'redis://localhost:{REDIS_PORT}/7'

os.environ['RUN_WORKERS'] = '0'
os.environ['MAX_WORKERS'] = '4'


@pytest.fixture(scope='session')
def test_client():
    """
    Create a test client for the FastAPI app.
    """
    from app.main import app
    from fastapi.testclient import TestClient
    from app.libs.utils import nothrow_killpg

    def _start_fake_redis():
        fakeredis.TcpFakeServer.allow_reuse_address = True
        server = fakeredis.TcpFakeServer(('localhost', REDIS_PORT))
        server.serve_forever()

    # blpop will make the server blocks, and hard to kill
    # so here we run it in a seperate process instead of thread
    # Note allow_reuse_address is set to True to avoid address already in use error
    redis_process = multiprocessing.Process(target=_start_fake_redis)
    redis_process.start()

    sleep(1)  # wait for the server to start

    workers = subprocess.Popen(
        [sys.executable, 'run_workers.py'],
        start_new_session=True,
    )
    sleep(1)  # wait for the workers to start

    try:
        with TestClient(app) as client:
            yield client
    finally:
        nothrow_killpg(pgid=workers.pid)
        workers.wait()

        redis_process.kill()
        redis_process.join()
