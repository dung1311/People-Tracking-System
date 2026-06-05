import subprocess
import os
import logging
from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)

def get_cpu_info():
    try:
        import psutil
        return {
            "usage_percent": psutil.cpu_percent(interval=0.1),
            "cores": psutil.cpu_count()
        }
    except ImportError:
        # Fallback using /proc/stat or simple shell
        try:
            # Quick estimation of CPU load using top
            cmd = "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'"
            out = subprocess.check_output(cmd, shell=True).decode().strip()
            cores_cmd = "nproc"
            cores = int(subprocess.check_output(cores_cmd, shell=True).decode().strip())
            return {
                "usage_percent": float(out) if out else 0.0,
                "cores": cores
            }
        except Exception as e:
            logger.warning(f"Failed to get fallback CPU info: {e}")
            return {"usage_percent": 0.0, "cores": 1}

def get_ram_info():
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "free_gb": round(mem.available / (1024**3), 2),
            "usage_percent": mem.percent
        }
    except ImportError:
        # Fallback reading /proc/meminfo or free command
        try:
            cmd = "free -b | grep Mem"
            out = subprocess.check_output(cmd, shell=True).decode().strip().split()
            # free -b outputs: Mem: total used free shared buff/cache available
            total = int(out[1])
            used = int(out[2])
            free = int(out[3])
            available = int(out[6]) if len(out) > 6 else free
            return {
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(available / (1024**3), 2),
                "usage_percent": round((used / total) * 100, 1) if total > 0 else 0.0
            }
        except Exception as e:
            logger.warning(f"Failed to get fallback RAM info: {e}")
            return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "usage_percent": 0.0}

def get_gpu_info():
    gpus = []
    try:
        # Run nvidia-smi
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        if output:
            for line in output.split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpus.append({
                        "name": parts[0],
                        "memory_total_mb": int(parts[1]),
                        "memory_used_mb": int(parts[2]),
                        "memory_free_mb": int(parts[3]),
                        "utilization_percent": int(parts[4])
                    })
    except Exception:
        # nvidia-smi failed or not present, return empty or dummy if simulated (but empty is cleaner for non-GPU system)
        pass
    return gpus

@router.get("/status")
async def get_system_status():
    return {
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "gpu": get_gpu_info()
    }
