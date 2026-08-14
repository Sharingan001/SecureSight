"""Redis-based JWT Token Blocklist for explicit logout / revocation.

Redis is initialized LAZILY inside each function call, not at import time.
This prevents Celery workers from crashing if Redis is unavailable at startup.
"""

import logging

logger = logging.getLogger(__name__)

_redis_client = None


def _get_redis():
    """Lazy-initialize and return the Redis client singleton."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        from app.config import settings
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        # Test connectivity immediately on first use
        _redis_client.ping()
    except Exception as e:
        logger.warning(f"Token blocklist Redis unavailable: {e}. Revocation disabled.")
        _redis_client = None
    return _redis_client


def block_token(jti: str, expires_in_seconds: int) -> None:
    """Add a token's JTI to the blocklist until it expires."""
    client = _get_redis()
    if not client:
        return
    try:
        client.setex(f"blocklist:{jti}", max(1, expires_in_seconds), "revoked")
    except Exception as e:
        logger.error(f"Failed to block token {jti}: {e}")


def is_token_blocked(jti: str) -> bool:
    """Check if a token's JTI is in the blocklist."""
    client = _get_redis()
    if not client:
        return False
    try:
        return client.exists(f"blocklist:{jti}") == 1
    except Exception as e:
        logger.error(f"Failed to check token {jti} against blocklist: {e}")
        return False
