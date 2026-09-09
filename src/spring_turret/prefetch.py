"""Bounded speculative CPU work. Never runs inference or touches servo state."""
import queue
import threading

if __package__:
    from .worker_protocol import iter_requests
else:
    from worker_protocol import iter_requests


class LatestPreparation:
    def __init__(self, prepare):
        self.prepare = prepare
        self.condition = threading.Condition()
        self.pending = self.ready = None
        self.generation = 0
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="cpu-prepare", daemon=True)
        self.thread.start()

    def submit(self, token, jpeg):
        with self.condition:
            if self.closed:
                return
            self.generation += 1
            self.pending = (self.generation, token, jpeg)
            self.ready = None
            self.condition.notify()

    def take(self, token, jpeg):
        with self.condition:
            candidate, self.ready = self.ready, None
        if candidate and candidate[0] == token and candidate[1] == jpeg:
            return candidate[2]
        return None  # Never wait for a late preparation or substitute a frame.

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                generation, token, jpeg = self.pending
                self.pending = None
            try:
                prepared = self.prepare(jpeg)
            except Exception:
                # Speculation is optional. The normal path reports any actual
                # requested-frame error; never retain/use a partial preparation.
                continue
            with self.condition:
                if not self.closed and generation == self.generation:
                    self.ready = (token, jpeg, prepared)

    def close(self):
        with self.condition:
            self.closed = True
            self.ready = self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout=1)


class RequestInbox:
    """Read the input pipe while the main thread is executing the GPU graph."""
    def __init__(self, stream, preparation):
        self.requests = queue.Queue(maxsize=1)
        self.preparation = preparation
        self.thread = threading.Thread(target=self._read, args=(stream,), name="worker-input", daemon=True)
        self.thread.start()

    def _read(self, stream):
        try:
            for request in iter_requests(stream):
                if request.get("kind") == "prepare":
                    token = request.get("token")
                    if not isinstance(token, str) or not 0 < len(token) <= 128:
                        raise ValueError("Invalid CPU preparation token")
                    self.preparation.submit(token, request["jpeg"])
                else:
                    self.requests.put(request)
            self.requests.put(None)
        except Exception as error:
            self.requests.put(error)

    def __iter__(self):
        while True:
            item = self.requests.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
