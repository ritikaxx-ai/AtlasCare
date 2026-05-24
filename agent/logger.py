"""
Structured JSON logger for AtlasCare.

Every log line is a JSON object — machine-readable, grep-able, splunkable.
Import `log` from here everywhere; never use print() in production code.

Log file: logs/atlascare.log  (also streams to stdout)
"""
import logging
import re
import sys
import os
from datetime import datetime, timezone
from pythonjsonlogger import jsonlogger

# ── ensure logs/ dir exists ──────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "atlascare.log")


class _AtlasCareFormatter(jsonlogger.JsonFormatter):
    """Add a fixed 'service' field and ISO timestamp to every record."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["service"] = "atlascare"
        log_record["level"] = record.levelname
        log_record.setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat()
        )
        # Remove noisy default fields
        log_record.pop("color_message", None)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("atlascare")
    if logger.handlers:          # already initialised (e.g. reload)
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = _AtlasCareFormatter(
        "%(timestamp)s %(level)s %(message)s",
        timestamp=True,
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    # rotating file handler — plain RotatingFileHandler keeps it simple
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(
        _LOG_FILE, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)   # file gets DEBUG; stdout gets INFO

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


log = _build_logger()


# ── sensitive-data masking ───────────────────────────────────────────────────
_MASK_PATTERNS = [
    # 16-digit card numbers
    (re.compile(r'\b(\d{4})\d{8}(\d{4})\b'), r'\1********\2'),
    # 10-digit Indian phone numbers
    (re.compile(r'\b(\d{2})\d{6}(\d{2})\b'), r'\1******\2'),
    # email addresses
    (re.compile(r'([a-zA-Z0-9_.+-]{1,3})[a-zA-Z0-9_.+-]*@([a-zA-Z0-9-]+\.[a-zA-Z]{2,})'),
     r'\1***@\2'),
]


def mask(text: str) -> str:
    """Mask card numbers, phone numbers, emails before logging."""
    for pattern, replacement in _MASK_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def mask_dict(d: dict) -> dict:
    """Recursively mask sensitive values in a dict (for tool outputs)."""
    import json
    raw = json.dumps(d, default=str)
    return json.loads(mask(raw))
