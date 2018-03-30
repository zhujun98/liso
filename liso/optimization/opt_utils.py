from datetime import datetime


def timestamp(func):
    """Decorator"""
    def timestamp_wrapper(*args, **kwargs):
        _start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        func(*args, **kwargs)
        _end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return timestamp_wrapper
