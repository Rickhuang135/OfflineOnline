import re
import sys
import threading

FILTER_PATTERNS = [
    re.compile(r"DEPRECATED_ENDPOINT"),
    re.compile(r"Fontconfig error"),
    re.compile(r"unknown libva error"),
    re.compile(r"The name org.freedesktop.UPower was not provided"),
    re.compile(r"Xlib.xauth: warning, no xauthority details available"),
    re.compile(r"rejected by interface blink.mojom.Widget"),
    re.compile(r"Registration response error message: QUOTA_EXCEEDED"),
]

def should_suppress(line: str) -> bool:
    return any(p.search(line) for p in FILTER_PATTERNS)

def forward_filtered_stream(stream):
    for raw in iter(stream.readline, ''):
        line = raw.rstrip("\n")
        if not should_suppress(line):
            print(line, file=sys.stderr, flush=True)

def filter_proc(proc):
    t = threading.Thread(target=forward_filtered_stream, args=(proc.stderr,), daemon=True)
    t.start()