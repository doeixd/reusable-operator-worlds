"""Launch the SO1 grid after checking CONCURRENCY_PLAN.md Amendment 1's host precondition."""
import json, subprocess, sys, time
import psutil

RSS_MIB = 389
RESERVE_MIB = 4096
HARD_CAP = 6


def pagefile_mib():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-c",
         "(Get-CimInstance Win32_PageFileUsage | Measure-Object CurrentUsage -Sum).Sum"],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out or 0)


def free_mib():
    return psutil.virtual_memory().available // 2**20


first = {"free_mib": free_mib(), "pagefile_mib": pagefile_mib(), "time": time.time()}
print("precondition sample 1:", json.dumps(first), flush=True)
time.sleep(60)
second = {"free_mib": free_mib(), "pagefile_mib": pagefile_mib(), "time": time.time()}
print("precondition sample 2:", json.dumps(second), flush=True)

need = RESERVE_MIB + 2 * RSS_MIB
if min(first["free_mib"], second["free_mib"]) <= need:
    sys.exit(f"PRECONDITION FAILED: free {min(first['free_mib'], second['free_mib'])} MiB <= {need} MiB")
if second["pagefile_mib"] > first["pagefile_mib"]:
    sys.exit(f"PRECONDITION FAILED: page file grew {first['pagefile_mib']} -> {second['pagefile_mib']} MiB")
print(f"PRECONDITION OK: free >= {min(first['free_mib'], second['free_mib'])} MiB, "
      f"page file stable at {second['pagefile_mib']} MiB, cap<= {HARD_CAP}", flush=True)

code = subprocess.run([sys.executable, "-m", "row.experiments.audit_so1_budget_bracket",
                       "--measured-rss-mib", str(RSS_MIB), "--hard-cap", str(HARD_CAP)]).returncode
print(f"SO1_EXIT={code}", flush=True)
