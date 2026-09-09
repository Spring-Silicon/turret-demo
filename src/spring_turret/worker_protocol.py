"""Bounded raw-JPEG requests; legacy base64 JSON remains readable."""
import base64
import json

JPEG_BYTES = "jpeg-bytes-v1"
MAX_JPEG = 8 * 1024 * 1024
MAX_LINE = 12 * 1024 * 1024  # Accommodate an 8 MiB legacy base64 image.


def encode_request(metadata, jpeg, transport):
    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= MAX_JPEG:
        raise ValueError("Worker JPEG must contain 1..8 MiB of bytes")
    if transport == JPEG_BYTES:
        header = json.dumps({**metadata, "transport": JPEG_BYTES, "jpeg_length": len(jpeg)},
                            separators=(",", ":")).encode() + b"\n"
        if len(header) > 65536:
            raise ValueError("Worker header exceeds 64 KiB")
        return header + jpeg
    return json.dumps({**metadata, "jpeg": base64.b64encode(jpeg).decode()},
                      separators=(",", ":")).encode() + b"\n"


def iter_requests(stream):
    """One ordered frame in flight; malformed framing terminates the worker."""
    while True:
        line = stream.readline(MAX_LINE + 1)
        if not line:
            return
        if len(line) > MAX_LINE or not line.endswith(b"\n"):
            raise ValueError("Worker request header too large or truncated")
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError("Worker request must be an object")
        if request.get("transport") == JPEG_BYTES:
            length = request.get("jpeg_length")
            if len(line) > 65536 or "jpeg" in request or type(length) is not int or not 0 < length <= MAX_JPEG:
                raise ValueError("Invalid binary JPEG header")
            parts = []
            remaining = length
            while remaining:
                chunk = stream.read(remaining)
                if not chunk:
                    raise EOFError("Truncated binary JPEG")
                parts.append(chunk)
                remaining -= len(chunk)
            jpeg = b"".join(parts)
        elif "transport" in request:
            raise ValueError("Unknown worker request transport")
        else:
            jpeg = base64.b64decode(request["jpeg"], validate=True)
            if not 0 < len(jpeg) <= MAX_JPEG:
                raise ValueError("Invalid legacy JPEG size")
        yield {**request, "jpeg": jpeg}
