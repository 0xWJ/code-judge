import logging
from typing import Awaitable
from time import time

import redis
import socket

logger = logging.getLogger(__name__)


class RedisQueue:
    def __init__(self, redis_uri, queue_name, *, enable_log=False, socket_timeout: int = None, is_async: bool = False):
        self.redis_uri = redis_uri
        self.is_async = is_async
        self.queue_name = queue_name
        self.socket_timeout = socket_timeout
        if self.socket_timeout is not None and self.socket_timeout < 5:
            raise ValueError('socket_timeout must be at least 5 seconds')
        self.redis: redis.Redis | redis.asyncio.Redis = self._init_redis(socket_timeout)

    def _init_redis(self, socket_timeout) -> redis.Redis | redis.asyncio.Redis:
        if '+cluster://' in self.redis_uri:
            Redis = redis.RedisCluster if not self.is_async else redis.asyncio.RedisCluster
            redis_uri = self.redis_uri.replace('+cluster://', '://')
        else:
            Redis = redis.Redis if not self.is_async else redis.asyncio.Redis
            redis_uri = self.redis_uri

        return Redis.from_url(
            redis_uri,
            socket_connect_timeout=120,
            socket_timeout=socket_timeout,
            socket_keepalive=True,
            health_check_interval=30,
            socket_keepalive_options={socket.TCP_KEEPIDLE: 2, socket.TCP_KEEPINTVL: 1, socket.TCP_KEEPCNT: 2}
        )

    def push(self, queue_name, value):
        return self.redis.rpush(queue_name, value)

    def pop(self, queue_name):
        return self.redis.lpop(queue_name)

    def _block_pop_sync(self, queue_name, timeout=0) -> tuple[str, bytes] | None:
        start = time()
        while True:
            if timeout > 0:
                effective_timeout = timeout - int(time() - start)
                if effective_timeout <= 0:
                    break
            else:
                effective_timeout = 0
            try:
                result = self.redis.blpop(queue_name, timeout=effective_timeout)
                if result:
                    return result
            except redis.exceptions.TimeoutError:
                continue
        return None

    async def _block_pop_async(self, queue_name, timeout=0) -> tuple[str, bytes] | None:
        start = time()
        while True:
            if timeout > 0:
                effective_timeout = timeout - int(time() - start)
                if effective_timeout <= 0:
                    break
            else:
                effective_timeout = 0
            try:
                result = await self.redis.blpop(queue_name, timeout=effective_timeout)
                if result:
                    return result
            except redis.exceptions.TimeoutError:
                continue
        return None

    def block_pop(self, queue_name, timeout=0) -> tuple[str, bytes] | None | Awaitable[tuple[str, bytes]] | Awaitable[None]:
        if self.is_async:
            return self._block_pop_async(queue_name, timeout)
        else:
            return self._block_pop_sync(queue_name, timeout)

    def expire(self, key, timeout):
        return self.redis.expire(key, timeout)

    def delete(self, key):
        return self.redis.delete(key)
