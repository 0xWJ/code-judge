import os
from app.version import __version__ as version


env = os.environ.get

ERROR_CASE_SAVE_PATH = env('ERROR_CASE_SAVE_PATH', '')  # default empty, which means not save error case

MAX_STDOUT_ERROR_LENGTH = int(env('MAX_STDOUT_ERROR_LENGTH', 1000))


def get_machine_id():
    with open('/etc/machine-id') as f:
        return f.read().strip()
MACHINE_ID = env('MACHINE_ID', None) or get_machine_id()

# timeline:
# |-----------------MAX_QUEUE_WAIT_TIME-------------------------------|
# |----MAX_QUEUE_WORK_LIFE_TIME----|-----MAX_EXECUTION_TIME-------|

MAX_EXECUTION_TIME = int(env('MAX_EXECUTION_TIME', 10))  # default 10 seconds (roughly)
# the max run time for a whole process
# add 5 seconds for additional code execution time (process creation, etc.) and communication time
MAX_PROCESS_TIME = MAX_EXECUTION_TIME + 5
# the work can only be in the queue for a limited time
# after that, it will to be considered as queue timeout in non-long-batch mode
MAX_QUEUE_WORK_LIFE_TIME = int(env('MAX_QUEUE_WORK_LIFE_TIME', 4))  # default 4s

# max wait time after submission is sent to the queue
MAX_QUEUE_WAIT_TIME = MAX_PROCESS_TIME + MAX_QUEUE_WORK_LIFE_TIME
LONG_BATCH_MAX_QUEUE_WAIT_TIME = int(env('LONG_BATCH_MAX_QUEUE_WAIT_TIME', 60*60 + MAX_PROCESS_TIME))  # default 1 hour
if LONG_BATCH_MAX_QUEUE_WAIT_TIME < 60 * 60 + MAX_PROCESS_TIME:
    raise ValueError('LONG_BATCH_MAX_QUEUE_WAIT_TIME must be bigger than 1 hour plus MAX_PROCESS_TIME')

# default 256 MB
# Please note 256MB is too small for some libraries like pytorch
# For example, loading pytorch can take over 4GB memory
MAX_MEMORY = int(env('MAX_MEMORY', 256))
MAX_WORKERS = int(env('MAX_WORKERS', os.cpu_count())) or os.cpu_count()  # default os.cpu_count()

RUN_WORKERS = int(env('RUN_WORKERS', 0))  # default 0, which means run workers in a separate process

MAX_BATCH_CHUNK_SIZE = int(env('MAX_BATCH_CHUNK_SIZE', 2))  # 0 means no limit
MAX_LONG_BATCH_CHUNK_SIZE = int(env('MAX_LONG_BATCH_CHUNK_SIZE', 100))

PYTHON_EXECUTOR_PATH = env('PYTHON_EXECUTOR_PATH', 'python3')
CPP_COMPILER_PATH = env('CPP_COMPILER_PATH', 'g++')

PYTHON_EXECUTE_COMMAND = env('PYTHON_EXECUTE_COMMAND', f'{PYTHON_EXECUTOR_PATH} {{source}}')
# In most cases, you don't need to change SESSION_INITIATOR_COMMAND
SESSION_INITIATOR_COMMAND = env('SESSION_INITIATOR_COMMAND', f'{PYTHON_EXECUTOR_PATH} {{source}}')
CPP_COMPILE_COMMAND = env('CPP_COMPILE_COMMAND', f'{CPP_COMPILER_PATH} -O2 -o {{exe}} {{source}}')
CPP_EXECUTE_COMMAND = env('CPP_EXECUTE_COMMAND', '{exe}')

# TODO: support fakeredis for testing.
REDIS_URI = env('REDIS_URI', '')
if not REDIS_URI:
    raise ValueError('REDIS_URI is not set')
REDIS_KEY_PREFIX = env('REDIS_KEY_PREFIX', 'js')
REDIS_RESULT_PREFIX = env('REDIS_RESULT_QUEUE_PREFIX', f'{REDIS_KEY_PREFIX}:{version}:result-queue:')
REDIS_RESULT_EXPIRE = int(env('REDIS_RESULT_EXPIRE', 60))  # default 1 minute
REDIS_RESULT_LONG_BATCH_EXPIRE = int(env('REDIS_RESULT_LONG_BATCH_EXPIRE', LONG_BATCH_MAX_QUEUE_WAIT_TIME))  # default 1 hour
REDIS_WORK_QUEUE_NAME = env('WORK_QUEUE_NAME', f'{REDIS_KEY_PREFIX}:{version}:work-queue')

REDIS_WORK_QUEUE_BLOCK_TIMEOUT = int(env('REDIS_WORK_QUEUE_BLOCK_TIMEOUT', 30))  # default 30 seconds
REDIS_WORKER_ID_PREFIX = env('REDIS_WORKER_ID_PREFIX', f'{REDIS_KEY_PREFIX}:{version}:work-ids:')
# default 2 minute + REDIS_WORK_QUEUE_BLOCK_TIMEOUT + MAX_PROCESS_TIME
REDIS_WORKER_REGISTER_EXPIRE = int(env('REDIS_WORKER_REGISTER_TIMEOUT', 120 + REDIS_WORK_QUEUE_BLOCK_TIMEOUT + MAX_PROCESS_TIME))

if REDIS_WORKER_REGISTER_EXPIRE < REDIS_WORK_QUEUE_BLOCK_TIMEOUT + MAX_PROCESS_TIME:
    raise ValueError('REDIS_WORKER_REGISTER_EXPIRE must be bigger than REDIS_WORK_QUEUE_BLOCK_TIMEOUT + MAX_PROCESS_TIME')

REDIS_MACHINE_IDS = env('REDIS_MACHINE_IDS', f'{REDIS_KEY_PREFIX}:{version}:machine-ids')
REDIS_MACHINE_REGISTER_EXPIRE = int(env('REDIS_MACHINE_REGISTER_EXPIRE', REDIS_WORKER_REGISTER_EXPIRE))

REDIS_SESSION_PREFIX = env('REDIS_SESSION_PREFIX', f'{REDIS_KEY_PREFIX}:{version}:session')
REDIS_SESSION_MACHINE_PREFIX_TEMPLATE = env('REDIS_SESSION_MACHINE_PREFIX_TEMPLATE', f'{REDIS_SESSION_PREFIX}:{{}}')
REDIS_SESSION_MACHINE_QUEUE_NAME_TEMPLATE = env('REDIS_SESSION_MACHINE_QUEUE_NAME_TEMPLATE', f'{REDIS_SESSION_MACHINE_PREFIX_TEMPLATE}:queue')
# Will store all sessions in a hash, key is pid, value is the session info json
REDIS_SESSION_MACHINE_SESSIONS_TEMPLATE = env('REDIS_SESSION_MACHINE_SESSIONS_TEMPLATE', f'{REDIS_SESSION_MACHINE_PREFIX_TEMPLATE}:set')

REDIS_SESSION_MACHINE_PREFIX = REDIS_SESSION_MACHINE_PREFIX_TEMPLATE.format(MACHINE_ID)
REDIS_SESSION_MACHINE_QUEUE_NAME = REDIS_SESSION_MACHINE_QUEUE_NAME_TEMPLATE.format(MACHINE_ID)
REDIS_SESSION_MACHINE_SESSIONS_NAME = REDIS_SESSION_MACHINE_SESSIONS_TEMPLATE.format(MACHINE_ID)

# We will set redis key expire time for session (last resort)
# This only works in latest versions of redis
REDIS_SESSION_EXPIRE = int(env('REDIS_SESSION_EXPIRE', 3600))  # default 1 hour
# The max session time we allow, must be smaller than REDIS_SESSION_EXPIRE
# So we have chance to clean up the session manually
MAX_SESSION_TIME = int(env('MAX_SESSION_TIME', REDIS_SESSION_EXPIRE / 2))  # default 30 minutes

# we use socket timeout to avoid blocking forever if a bad connection is used
REDIS_SOCKET_TIMEOUT = int(env('REDIS_SOCKET_TIMEOUT', 60)) # default 1 minute
