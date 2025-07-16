import time

class RateLimiter:
    """
    Rate limiter enforcing both requests-per-minute (RPM) and tokens-per-minute (TPM) limits.

    Usage:
        limiter = RateLimiter(rpm=60, tpm=150000)
        limiter.acquire(tokens=N, timeout=5)  # blocks up to 5s
    """
    def __init__(self, rpm: int, tpm: int):
        if not isinstance(rpm, int) or rpm <= 0:
            raise ValueError("rpm must be a positive integer")
        if not isinstance(tpm, int) or tpm <= 0:
            raise ValueError("tpm must be a positive integer")

        self.rpm = rpm
        self.tpm = tpm
        self._req_tokens = rpm
        self._tok_tokens = tpm
        now = time.time()
        self._last_req_refill = now
        self._last_tok_refill = now

    def _refill(self):
        now = time.time()
        # Refill request bucket
        elapsed_req = now - self._last_req_refill
        cycles_req = int(elapsed_req // 60)
        if cycles_req > 0:
            self._req_tokens = min(self.rpm, self._req_tokens + cycles_req * self.rpm)
            self._last_req_refill += cycles_req * 60
        # Refill token bucket
        elapsed_tok = now - self._last_tok_refill
        cycles_tok = int(elapsed_tok // 60)
        if cycles_tok > 0:
            self._tok_tokens = min(self.tpm, self._tok_tokens + cycles_tok * self.tpm)
            self._last_tok_refill += cycles_tok * 60

    def acquire(self, tokens: int = 0, timeout: float = None) -> bool:
        """
        Block until both a request slot and the requested token budget are available.

        Args:
            tokens (int): Number of tokens this request will consume.
            timeout (float): Maximum seconds to wait (None for infinite).
        Returns:
            True if acquired, False if timeout.
        """
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        start = time.time()
        while True:
            self._refill()
            if self._req_tokens > 0 and self._tok_tokens >= tokens:
                self._req_tokens -= 1
                self._tok_tokens -= tokens
                return True
            if timeout is not None and (time.time() - start) >= timeout:
                return False
            time.sleep(0.1)

    def get_status(self) -> dict:
        """
        Return current rate-limiter status, including RPM and TPM details.
        """
        now = time.time()
        elapsed_req = now - self._last_req_refill
        elapsed_tok = now - self._last_tok_refill
        return {
            "rpm_limit": self.rpm,
            "requests_remaining": self._req_tokens,
            "tpm_limit": self.tpm,
            "tokens_remaining": self._tok_tokens,
            "seconds_since_req_refill": round(elapsed_req, 2),
            "seconds_since_tok_refill": round(elapsed_tok, 2),
            "req_refill_in": max(0, round(60 - (elapsed_req % 60), 2)),
            "tok_refill_in": max(0, round(60 - (elapsed_tok % 60), 2)),
        }