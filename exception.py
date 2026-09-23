import sys
from typing import Any


def error_message_detail(error: Any, error_detail: sys):
    try:
        exc_type, exc_obj, exc_tb = error_detail.exc_info()
    except Exception:
        exc_tb = None

    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "<unknown>"
        line_number = 0

    return (
        f"Error occurred in script [{file_name}] "
        f"line [{line_number}] "
        f"message [{str(error)}]"
    )


class CustomException(Exception):
    def __init__(self, error_message: Any, error_details: sys = None):
        super().__init__(str(error_message))
        if error_details is None:
            import sys as _sys
            error_details = _sys

        self.error_message = error_message_detail(
            error_message, error_detail=error_details
        )

    def __str__(self):
        return self.error_message
