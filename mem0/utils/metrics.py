import os
from datadog import initialize
from datadog.dogstatsd.base import statsd
import logging

logger = logging.getLogger(__name__)

options = {"statsd_host": "127.0.0.1", "statsd_port": 8125}

initialize(**options)


class DatadogMetrics:
    _instance = None

    def __init__(self):
        env = os.getenv("ENV", "local")
        service_name = "mem0"
        self.default_tags = [f"env:{env.value}", f"service:{service_name}"]
        self.prefix = "oliva.mem0."

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def gauge(self, name, value, tags=None):
        try:
            final_tags = self.default_tags + (tags if tags else [])
            statsd.gauge(f"{self.prefix}{name}", value, tags=final_tags)
        except Exception as e:
            logger.error(f"Error sending gauge metric: {e}")

    def increment(self, name, value=1, tags=None):
        try:
            final_tags = self.default_tags + (tags if tags else [])
            statsd.increment(f"{self.prefix}{name}", value, tags=final_tags)
        except Exception as e:
            logger.error(f"Error incrementing metric: {e}")

    def histogram(self, name, value, tags=None):
        try:
            final_tags = self.default_tags + (tags if tags else [])
            statsd.histogram(f"{self.prefix}{name}", value, tags=final_tags)
        except Exception as e:
            logger.error(f"Error sending histogram metric: {e}")

    def distribution(self, name, value, tags=None):
        try:
            final_tags = self.default_tags + (tags if tags else [])
            statsd.distribution(f"{self.prefix}{name}", value, tags=final_tags)
        except Exception as e:
            logger.error(f"Error sending distribution metric: {e}")

    def flush(self):
        # The Python client handles flushing automatically, but you can force it if necessary
        statsd.flush()
