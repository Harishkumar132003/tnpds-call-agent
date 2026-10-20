"""Operations portal: agent prompt/temperature and call tracking with latencies.

Runs as its own process (``python portal.py``) with its own connection pool,
so browsing calls never competes with the voice pipeline.
"""
