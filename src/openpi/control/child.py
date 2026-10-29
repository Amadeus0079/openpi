"""Linux child lifecycle: exit when the console's persistent spawn thread dies."""

import ctypes
import os
import signal
import sys


def main():
    parent = int(sys.argv[1])
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot register parent-death signal")
    if os.getppid() != parent:
        raise SystemExit("Console parent already exited")
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
