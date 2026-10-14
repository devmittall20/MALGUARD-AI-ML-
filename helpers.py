import time
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and log execution time of functions."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def machine_type_to_str(machine_val: int) -> str:
    """Converts PE File Header machine value to readable name."""
    mapping = {
        0x014c: "i386 (32-bit x86)",
        0x8664: "AMD64 (64-bit x86-64)",
        0x0200: "IA64 (Intel Itanium)",
        0x01c0: "ARM (ARM Little Endian)",
        0xaa64: "ARM64"
    }
    return mapping.get(machine_val, f"Unknown (0x{machine_val:04x})")

def optional_header_magic_to_str(magic_val: int) -> str:
    """Converts PE Optional Header Magic to readable name."""
    mapping = {
        0x10b: "PE32 (32-bit)",
        0x20b: "PE32+ (64-bit / PE32 Plus)",
        0x107: "ROM"
    }
    return mapping.get(magic_val, f"Unknown (0x{magic_val:04x})")
