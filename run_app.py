"""Launch and supervise the SgSL recogniser and Streamlit interface."""

from pathlib import Path
import os
import signal
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTOR_PATH = PROJECT_ROOT / "recognition" / "predict.py"
STREAMLIT_PATH = PROJECT_ROOT / "app.py"
SHUTDOWN_TIMEOUT_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.2


def child_process_options() -> dict:
    """Place each child in a group that the launcher can stop safely."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def start_child(name: str, command: list[str]) -> subprocess.Popen:
    """Start one child with output attached to the current terminal."""
    print(f"Starting {name}: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        **child_process_options(),
    )
    print(f"{name} started (PID {process.pid}).", flush=True)
    return process


def signal_child(process: subprocess.Popen, interrupt: bool = True) -> None:
    """Signal a child process group if it is still running."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            if interrupt:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
        else:
            os.killpg(
                process.pid,
                signal.SIGINT if interrupt else signal.SIGTERM,
            )
    except ProcessLookupError:
        pass


def stop_children(children: list[tuple[str, subprocess.Popen]]) -> None:
    """Stop all children gracefully, escalating only when required."""
    running = [(name, process) for name, process in children if process.poll() is None]
    if not running:
        return

    print("Stopping recogniser and Streamlit...", flush=True)
    for _, process in running:
        signal_child(process)

    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
    while time.monotonic() < deadline and any(
        process.poll() is None for _, process in running
    ):
        time.sleep(0.1)

    for name, process in running:
        if process.poll() is None:
            print(f"{name} did not stop after Ctrl+C; terminating it.", flush=True)
            signal_child(process, interrupt=False)

    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
    while time.monotonic() < deadline and any(
        process.poll() is None for _, process in running
    ):
        time.sleep(0.1)

    for name, process in running:
        if process.poll() is None:
            print(f"{name} did not terminate; killing its process group.", flush=True)
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    for _, process in children:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    print("All app processes stopped.", flush=True)


def run() -> int:
    """Start both app processes and supervise them until shutdown."""
    for required_path in (PREDICTOR_PATH, STREAMLIT_PATH):
        if not required_path.is_file():
            print(f"Cannot start app; missing file: {required_path}", file=sys.stderr)
            return 1

    python = sys.executable
    children: list[tuple[str, subprocess.Popen]] = []
    stopped_child = None
    try:
        children.append((
            "SgSL recogniser",
            start_child("SgSL recogniser", [python, "recognition/predict.py"]),
        ))
        children.append((
            "Streamlit UI",
            start_child(
                "Streamlit UI",
                [python, "-m", "streamlit", "run", "app.py"],
            ),
        ))
        print("App is running. Press Ctrl+C here to stop both processes.", flush=True)

        while True:
            for name, process in children:
                exit_code = process.poll()
                if exit_code is not None:
                    stopped_child = name
                    print(
                        f"{name} exited unexpectedly with code {exit_code}.",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nCtrl+C received.", flush=True)
        return 0
    except OSError as error:
        print(f"Could not start the app: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        stop_children(children)
        if stopped_child:
            print(f"Stopped remaining processes after {stopped_child} exited.", flush=True)
        else:
            print("Shutdown complete.", flush=True)


if __name__ == "__main__":
    raise SystemExit(run())
