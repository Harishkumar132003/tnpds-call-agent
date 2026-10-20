"""Browser (WebRTC) transport parameters for local development and testing.

The Pipecat development runner (``pipecat.runner.run``) owns the FastAPI
server, the ``/api/offer`` signalling endpoint and the prebuilt web client at
``/client``; this module only declares how audio should flow for that transport.
"""

from __future__ import annotations

from pipecat.transports.base_transport import TransportParams


def webrtc_transport_params() -> TransportParams:
    """Audio-only, bidirectional WebRTC transport."""
    return TransportParams(audio_in_enabled=True, audio_out_enabled=True)
