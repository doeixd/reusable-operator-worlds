"""Machine workaround: make platform's WMI query fail fast.

On this Windows machine the WMI service intermittently stalls, which hangs
`platform.machine()` (called during `import torch`) for minutes per
process. Forcing the WMI path to raise immediately makes `platform` fall
back to environment-variable answers, which are correct here. Harmless on
machines without the stall. Loaded automatically because `src/` is on
sys.path via the editable install.
"""

import platform

try:
    platform._wmi_query = lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("wmi disabled by sitecustomize")
    )
except Exception:
    pass
