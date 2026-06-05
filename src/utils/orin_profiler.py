import subprocess
import re
import time
import os

class OrinProfiler:
    """
    Captures real-time tegrastats memory, GPU, and thermal consumption on NVIDIA Orin modules.
    Provides a fallback for non-Orin environments for development.
    """
    def __init__(self, interval: int = 1000):
        self.interval = interval
        self.is_orin = self._check_orin()
        self.stats = []

    def _check_orin(self) -> bool:
        try:
            return os.path.exists("/sys/module/tegra_fuse")
        except:
            return False

    def get_current_stats(self) -> dict:
        if not self.is_orin:
            return {"gpu_util": 0.0, "vram_mb": 0.0, "temp_c": 45.0, "power_mw": 5000}
        
        try:
            # Run tegrastats for a single iteration
            proc = subprocess.run(["tegrastats", "--interval", str(self.interval), "--count", "1"], 
                                  capture_output=True, text=True)
            output = proc.stdout
            
            # Parse output (simplified example)
            stats = {
                "gpu_util": self._parse_val(output, r"GR3D_FREQ (\d+)%"),
                "vram_mb": self._parse_val(output, r"RAM (\d+)/"),
                "temp_c": self._parse_val(output, r"thermal (\d+\.?\d*)"),
                "power_mw": self._parse_val(output, r"POM_5V_IN (\d+)")
            }
            return stats
        except Exception as e:
            return {"error": str(e)}

    def _parse_val(self, text, pattern):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 0.0
