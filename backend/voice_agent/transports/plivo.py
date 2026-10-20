"""Plivo telephony transport parameters.

Plivo's Audio Streaming connects to the runner's ``/ws`` websocket and carries
8 kHz mu-law audio. The runner parses the stream/call IDs from the first
messages and attaches ``PlivoFrameSerializer`` to these params itself; it reads
``PLIVO_AUTH_ID`` / ``PLIVO_AUTH_TOKEN`` from the environment for hang-up.

Run with ``python bot.py -t plivo --proxy <public-host>``; the runner then
answers Plivo's webhook with the ``<Stream>`` XML pointing at ``wss://<public-host>/ws``.
"""

from __future__ import annotations

from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams


def plivo_transport_params() -> FastAPIWebsocketParams:
    """Audio-only, bidirectional websocket transport for Plivo calls."""
    return FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)
