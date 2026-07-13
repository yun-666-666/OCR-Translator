"""Small dependency-free helpers for safely publishing user-owned files."""

import os
import uuid


def write_atomically(
    file_path,
    writer,
    *,
    mode,
    encoding=None,
    open_file=None,
    replace_file=None,
    fsync_file=None,
    remove_file=None,
    token_factory=None,
):
    """Write a unique same-directory temporary file, then publish it by replace."""
    if mode not in {"x", "xb"}:
        raise ValueError("atomic write mode must be 'x' or 'xb'")

    open_file = open if open_file is None else open_file
    replace_file = os.replace if replace_file is None else replace_file
    fsync_file = os.fsync if fsync_file is None else fsync_file
    remove_file = os.remove if remove_file is None else remove_file
    token_factory = uuid.uuid4 if token_factory is None else token_factory

    absolute_path = os.path.abspath(os.fspath(file_path))
    temporary_path = os.path.join(
        os.path.dirname(absolute_path),
        f".{os.path.basename(absolute_path)}.{token_factory().hex}.tmp",
    )
    open_kwargs = {} if mode.endswith("b") else {"encoding": encoding}
    active_error = None
    try:
        with open_file(temporary_path, mode, **open_kwargs) as temporary_file:
            writer(temporary_file)
            temporary_file.flush()
            fsync_file(temporary_file.fileno())
        replace_file(temporary_path, absolute_path)
        temporary_path = None
    except BaseException as error:
        active_error = error
        raise
    finally:
        if temporary_path is not None:
            try:
                remove_file(temporary_path)
            except FileNotFoundError:
                pass
            except BaseException:
                if active_error is None:
                    raise


def write_text_atomically(file_path, content, *, encoding, **kwargs):
    """Atomically publish text with the requested encoding."""
    write_atomically(
        file_path,
        lambda temporary_file: temporary_file.write(content),
        mode="x",
        encoding=encoding,
        **kwargs,
    )


def write_bytes_atomically(file_path, content, **kwargs):
    """Atomically publish raw bytes."""
    write_atomically(
        file_path,
        lambda temporary_file: temporary_file.write(content),
        mode="xb",
        **kwargs,
    )
