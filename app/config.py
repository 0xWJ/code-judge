import os


env = os.environ.get

ERROR_CASE_SAVE_PATH = env('ERROR_CASE_SAVE_PATH', '')  # default empty, which means not save error case

# TODO: support fakeredis for testing.
REDIS_URI = env('REDIS_URI', '')
if not REDIS_URI:
    raise ValueError('REDIS_URI is not set')
REDIS_KEY_PREFIX = env('REDIS_KEY_PREFIX', 'js:')
REDIS_RESULT_PREFIX = env('REDIS_RESULT_QUEUE_PREFIX', f'{REDIS_KEY_PREFIX}result-queue:')
REDIS_RESULT_EXPIRE = int(env('REDIS_RESULT_EXPIRE', 3600))  # default 1 hour
REDIS_WORK_QUEUE_NAME = env('WORK_QUEUE_NAME', f'{REDIS_KEY_PREFIX}work-queue')

MAX_EXECUTION_TIME = int(env('MAX_EXECUTION_TIME', 10))  # default 10 seconds
# default 15 seconds
# additional 5 seconds for communication between judge server and judge worker
MAX_QUEUE_WAIT_TIME = int(env('MAX_QUEUE_WAIT_TIME', MAX_EXECUTION_TIME + 5))
MAX_MEMORY = int(env('MAX_MEMORY', 256))  # default 256 MB
MAX_WORKERS = int(env('MAX_WORKERS', os.cpu_count())) or os.cpu_count()  # default os.cpu_count()

RUN_WORKERS = int(env('RUN_WORKERS', 0))  # default 0, which means run workers in a separate process

MAX_BATCH_CHUNK_SIZE = int(env('MAX_BATCH_CHUNK_SIZE', 20))

PYTHON_EXECUTOR_PATH = env('PYTHON_EXECUTOR_PATH', 'python3')
CPP_COMPILER_PATH = env('CPP_COMPILER_PATH', 'g++')
