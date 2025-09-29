import subprocess
import time
import sys
import selectors
import psutil
import os

os.environ['PYTHONUNBUFFERED'] = '1'  # ensure python output is unbuffered
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'  # for debugpy


READY_MARK = '##SESSION-READY##'
INFO_MARK = '##SESSION-INFO##'


if __name__ == '__main__':
    process = subprocess.Popen(
        sys.argv[1:],    # command will be passed as script arguments
        stdin=sys.stdin, # connect to parent process stdin
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)

    for key, _ in selector.select():
        line = key.fileobj.readline()
        if not line:  # EOF
            print("Session terminated unexpectedly", file=sys.stderr)
            exit(1)
        if key.fileobj is process.stdout:
            if line.startswith(READY_MARK):
                break
            print(line, end='', flush=True)  # end='' because line ends '\n'
        else:
            print(line, end='', flush=True, file=sys.stderr)

    print(INFO_MARK, flush=True)
    worker_p = psutil.Process(process.pid)
    print(f'{{"pid": {process.pid}, "create_time": {worker_p.create_time()}}}', flush=True)
