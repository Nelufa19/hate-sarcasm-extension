"""
Structured (JSON-lines) logging. Each log line is one JSON object, so a
future log aggregator (or even `jq` on a log file) can query/filter fields
like latency_ms or language without parsing free-form text.
"""
import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger("hate_sarcasm_api")
    root.setLevel(logging.INFO)
    root.handlers = [handler]
    root.propagate = False
