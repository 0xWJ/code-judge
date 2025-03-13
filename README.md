# Pre-requisite

## Install docker

Please ref to [docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

Basically you can run the following commands to install docker:

1. Set up Docker's apt repository.
```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

2. Install the Docker packages.
```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

3. Verify
```bash
sudo docker run hello-world
```

# Run the project

## Build the image

```bash
sudo docker compose build
```

## Run

```bash
sudo docker compose up
```

## Stop

```bash
sudo docker compose down
```

# Run the project without docker

## Install redis server

```bash
sudo apt-get update
sudo apt-get install redis
```
and run the following command to start the redis server (if systemctl doesn't work).
```bash
redis-server --daemonize yes
```

## Start api service and worker

### Start them seperately
```bash
REDIS_URI=redis://localhost:6379 uvicorn app.main:app --workers 4 --limit-max-requests 1000
REDIS_URI=redis://localhost:6379 python run_workers.py
```

## Start them together (not recommended)
```bash
REDIS_URI=redis://localhost:6379 python debug_api.py
```

# Debug

## Run Redis
You need to have a redis server running to test the project. You can run the following command to start a redis server.

```bash
sudo docker run -p 6379:6379 --rm --name test-redis  redis:alpine
```
And set the `REDIS_URI` env variable to `redis://localhost:6379`.

## Run the project in debug mode

```bash
REDIS_URI=redis://localhost:6379 python debug_api.py
```

# Usage

The input and output of the script use the standard input and output of the script.

You can specify the input and expected output of the script in the request body like

```
{
  "type": "python",
  "solution": "print(input())",
  "input": "9",
  "expected_output": "9"
}
```

## use httpie
You need to firstly install httpie via
```
sudo apt install httpie
```
and then you can use the following command to test the api.
```bash
 http post http://0.0.0.0:8000/judge type=python solution="print(9)" expected_output=9
 ```
 ## use curl
  You can also use curl to test the api.
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"type":"python", "solution":"print(9)", "expected_output":"9"}' http://0.0.0.0:8000/judge
  ```

  ## Use requests
  You can also use python requests to test the api.
  ```python
  import requests
  response = requests.post(
    "http://0.0.0.0:8000/judge",
    json={
      "type":"python",
      "solution":"print(9)",
      "expected_output":"9"
    })
  print(response.json())
  ```

  # API
  ## POST /judge
  ### request (Submission)
  ```python
    # the submission id
    sub_id: str | None = None
    # the language type, currently only python and cpp are supported
    type: Literal['python', 'cpp']
    # the solution code
    solution: str
    # the standard input of the code (for example, input() function in python)
    input: str | None = None
    # the expected output of the code
    # we will compare the output of the code with this value if it is None
    expected_output: str | None = None
  ```
  ### Response
  ```python
    # the submission id
    sub_id: str
    # whether the checking is successful
    # true if the output of the code is equal to the expected output
    success: bool
    # the time cost of the code in seconds
    cost: float
    # the reason of failure
    # '': no reason, plain success or plain failure
    # 'worker_timeout': the code takes too long to run
    # 'queue_timeout': the code takes too long to be processed.
    #   This is usually caused by the workers being too busy.
    # 'internal_error': The failure is caused by the internal error of the system.
    #   This can be caused by the redis server being down or exceeding the max connection limit.
    reason: str
  ```

## /judge/batch
  ### Request
  ```python
      sub_id: str | None = None
      type: Literal['batch'] = 'batch'
      # list of submissions
      submissions: list[Submission]
  ```

  ### Response
  ```python
    sub_id: str
    # list of submission results
    results: list[SubmissionResult]
  ```

   ## POST /run
  ### request (Submission)
  ```python
    # the submission id
    sub_id: str | None = None
    # the language type, currently only python and cpp are supported
    type: Literal['python', 'cpp']
    # the solution code
    solution: str
    # the standard input of the code (for example, input() function in python)
    input: str | None = None
    # the expected output of the code
    # we will compare the output of the code with this value if it is None
    expected_output: str | None = None
  ```
  ### Response
  ```python
    # the submission id
    sub_id: str
    # whether the run is successful
    success: bool
    # the time cost of the code in seconds
    cost: float
    # the reason of failure
    # '': no reason, plain success or plain failure
    # 'worker_timeout': the code takes too long to run
    # 'queue_timeout': the code takes too long to be processed.
    #   This is usually caused by the workers being too busy.
    # 'internal_error': The failure is caused by the internal error of the system.
    #   This can be caused by the redis server being down or exceeding the max connection limit.
    reason: str
    stdout: str
    stderr: str
  ```

## /run/batch
  ### Request
  ```python
      sub_id: str | None = None
      type: Literal['batch'] = 'batch'
      # list of submissions
      submissions: list[Submission]
  ```

  ### Response
  ```python
    sub_id: str
    # list of submission results
    results: list[SubmissionResult]
  ```

# Mutiple node Deployment without orchestration tools

You can deploy the projects with k8s, docker swarm or other orchestration tools.

If you don't want to use them, you can also run workers/api in multiple machines with same redis.

1. Setup redis server in one machine or use a cloud redis server.
   Assume you are using azure redis server.
   1. Create a redis server in azure.
   2. Get the redis server uri from the azure portal. Please note that the uri should be in the format of `rediss+cluster://:<access key>@<host>:<port>`.
   Here `rediss` means tls. If tls is disabled, use `redis`.
   And `+cluster` means redis cluster. If high availability is disbled, remove `+cluster`.

   **Warning**: Cluster redis is not well tested. Please use it at your own risk. From our test, it can lead to hang or max connection error.
2. Run workers in all worker nodes with the same redis uri. You can reuse the training servers, as workers don't use GPU.
3. Run api in api nodes with the same redis uri. You can use one api node or multiple api nodes.


# Client Implementation

1. Batch API is preferred.
2. To make your client more robust, you'd better:
  - check http status code. We are trying to always return 200, but it is not guaranteed.
  - check the `reason` field in the response. For example, `queue_timeout` means the workers are busy. You should reduce the concurrent requests.
  - make sure you have set timeout for the request(i.e.`requests.post(..., timeout=...)`).
3. You should check the log of the api and workers to see if there are any errors.
