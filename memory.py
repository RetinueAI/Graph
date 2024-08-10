import sys
import math


def get_memory_usage(obj, seen=None):
    """Recursively find the memory usage of objects."""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        size += sum([get_memory_usage(v, seen) for v in obj.values()])
        size += sum([get_memory_usage(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_memory_usage(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_memory_usage(i, seen) for i in obj])
    
    return size


def human_readable_size(size_bytes):
    """Convert bytes to a more human-readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_name[i]}"