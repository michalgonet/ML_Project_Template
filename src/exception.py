import sys

from logger import logging

def error_msg_details(error, error_details: sys) -> str:
    _, _, exc_tp = error_details.exc_info()
    file_name = exc_tp.tb_frame.f_code.co_filename
    line_no = exc_tp.tb_lineno
    return f'Error occurred in python script: [{file_name}], line: [{line_no}], error: [{str(error)}]'


class CustomException(Exception):
    def __init__(self, error_msg, error_detail: sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_details(error_msg, error_detail)

    def __str__(self):
        return self.error_msg


