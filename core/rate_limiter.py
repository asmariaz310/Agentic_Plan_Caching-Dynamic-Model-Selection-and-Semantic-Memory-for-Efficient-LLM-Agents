import time
import threading
from collections import deque
from typing import Optional

class RateLimiter:
    """Rate limiter for Groq API to handle request limits."""

    def __init__(self, requests_per_minute: int = 30, burst_limit: int = 10):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_limit: Maximum burst requests allowed
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.request_times = deque(maxlen=requests_per_minute)
        self.last_request_time = 0
        self.burst_times = deque(maxlen=burst_limit)
        self.lock = threading.Lock()

    def wait_if_needed(self) -> float:
        """
        Wait if necessary to respect rate limits.

        Returns:
            Time waited in seconds
        """
        with self.lock:
            current_time = time.time()

            # Clean old requests from deques
            self._clean_old_requests(current_time)

            # Check burst limit (requests in the last second)
            burst_wait = 0.0
            if len(self.burst_times) >= self.burst_limit:
                oldest_burst = self.burst_times[0]
                burst_wait = max(0.0, 1.0 - (current_time - oldest_burst))

            # Check per-minute limit
            minute_wait = 0.0
            if len(self.request_times) >= self.requests_per_minute:
                oldest_request = self.request_times[0]
                minute_wait = max(0.0, 60.0 - (current_time - oldest_request))

            total_wait = max(burst_wait, minute_wait)
            if total_wait > 0:
                time.sleep(total_wait)
                current_time = time.time()
                self._clean_old_requests(current_time)

            # Record this request
            self.request_times.append(current_time)
            self.burst_times.append(current_time)

            return total_wait

    def _clean_old_requests(self, current_time: float):
        """Clean requests older than the time windows."""
        # Clean minute-old requests
        while self.request_times and (current_time - self.request_times[0]) > 60.0:
            self.request_times.popleft()

        # Clean burst-old requests (1 second window)
        while self.burst_times and (current_time - self.burst_times[0]) > 1.0:
            self.burst_times.popleft()

    def get_remaining_capacity(self) -> dict:
        """Get remaining capacity information."""
        current_time = time.time()
        self._clean_old_requests(current_time)

        next_reset = 0.0

        if self.request_times:
            next_reset = 60.0 - (current_time - self.request_times[0])

        return {
            "requests_remaining_minute": self.requests_per_minute - len(self.request_times),
            "burst_capacity_remaining": self.burst_limit - len(self.burst_times),
            "next_minute_reset": next_reset
        }

class GroqAPIManager:
    """Manages Groq API interactions with rate limiting and error handling."""

    def _get_model_limits(self, model_name: str):
        """Define Groq model-specific rate limits (safe conservative values)."""
        limits = {
            "llama-3.1-8b-instant": {"rpm": 30, "burst": 5},
            "llama-3.3-70b-versatile": {"rpm": 30, "burst": 3},
            "groq/compound-mini": {"rpm": 30, "burst": 8},
            "compound-mini": {"rpm": 30, "burst": 8}  # Alternative name
        }
        return limits.get(model_name, {"rpm": 30, "burst": 3})


    def __init__(self, model_name: str = None, requests_per_minute: int = None):
        """Initialize Groq API manager with flexible rate limiting."""

        # handle both model-based and manual limits
        if model_name:
            limits = self._get_model_limits(model_name)
            rpm = limits["rpm"]
            burst = limits["burst"]
        else:
            rpm = requests_per_minute or 30
            burst = 5

        self.rate_limiter = RateLimiter(
            requests_per_minute=rpm,
            burst_limit=burst
        )

        self.model_name = model_name
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = 0


    def make_request(self, llm_call_func, *args, **kwargs):
        """
        Make a rate-limited LLM request with error handling and increased retries.

        Args:
            llm_call_func: The LLM function to call (e.g., chatgroq.invoke)
            *args, **kwargs: Arguments for the LLM function

        Returns:
            Response from the LLM call
        """
        # Wait for rate limit
        wait_time = self.rate_limiter.wait_if_needed()
        if wait_time > 0:
            print(f"Rate limited, waited {wait_time:.2f}s")

        min_interval = 60.0 / self.rate_limiter.requests_per_minute
        time_since_last = time.time() - self.last_request_time

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)

        try:
            self.request_count += 1
            self.last_request_time = time.time()

            # Make the actual request
            response = llm_call_func(*args, **kwargs)

            return response

        except Exception as e:
            self.error_count += 1
            error_msg = str(e).lower()

            # Handle specific Groq errors with increased retries
            if "rate limit" in error_msg or "429" in error_msg:
                print(f"Rate limit hit, backing off with retries...")
                # Retry up to 4 times for rate limits
                for retry_count in range(4):
                    time.sleep((2 ** retry_count) * 3) 
                    try:
                        print(f"  [Retry {retry_count + 1}/4]")
                        return llm_call_func(*args, **kwargs)
                    except Exception as retry_error:
                        if retry_count == 3:  # Last retry
                            raise retry_error
                        continue

            elif "quota" in error_msg or "402" in error_msg:
                raise Exception("API quota exceeded")

            elif "timeout" in error_msg or "500" in error_msg or "503" in error_msg:
                print(f"Server error, retrying with exponential backoff...")
                # Retry up to 4 times for server errors
                for retry_count in range(4):
                    time.sleep((retry_count + 1) * 3)  # Exponential backoff: 3s, 6s, 9s, 12s
                    try:
                        print(f"  [Retry {retry_count + 1}/4]")
                        return llm_call_func(*args, **kwargs)
                    except Exception as retry_error:
                        if retry_count == 3:  # Last retry
                            raise retry_error
                        continue

            else:
                # Re-raise other errors
                raise e

    def get_stats(self) -> dict:
        """Get API usage statistics."""
        capacity = self.rate_limiter.get_remaining_capacity()

        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "requests_remaining_minute": capacity["requests_remaining_minute"],
            "burst_capacity_remaining": capacity["burst_capacity_remaining"],
            "time_to_minute_reset": capacity["next_minute_reset"],
            "last_request_seconds_ago": time.time() - self.last_request_time
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.request_count = 0
        self.error_count = 0