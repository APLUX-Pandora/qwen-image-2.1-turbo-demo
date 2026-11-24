"""Application signal coordination for a cancellable SDK owner."""
import signal
import threading


class StopControl:
    """Consume TERM/INT in an ordinary thread; never destroy from a handler."""
    def __init__(self):
        self.stopped = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.engine = None
        self.first_signal = 0
        self.thread = None

    def __enter__(self):
        self.signals = {signal.SIGTERM, signal.SIGINT}
        self.mask = signal.pthread_sigmask(signal.SIG_BLOCK, self.signals)
        self.thread = threading.Thread(target=self._receive, name="image-stop-signals")
        try:
            self.thread.start()
        except BaseException:
            signal.pthread_sigmask(signal.SIG_SETMASK, self.mask)
            raise
        return self

    def _receive(self):
        while not self.done.is_set():
            event = signal.sigtimedwait(self.signals, .1)
            if event is not None:
                if not self.first_signal:
                    self.first_signal = event.si_signo
                self.stopped.set()
                with self.lock:
                    if self.engine is not None:
                        self.engine.cancel()

    def publish(self, engine):
        with self.lock:
            self.engine = engine

    def __exit__(self, *_):
        # The caller's synchronous generation/context manager has returned.
        with self.lock:
            self.engine = None
        self.done.set()
        self.thread.join()
        signal.pthread_sigmask(signal.SIG_SETMASK, self.mask)

    @property
    def exit_code(self):
        return 128 + self.first_signal if self.first_signal else 0
