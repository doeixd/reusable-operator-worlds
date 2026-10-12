import sys, json, time
from row.experiments import census_b2h_root as c
r = c.job(("ALLLOOP", sys.argv[1], 54, 2))
r.pop("canonical_per_task")
print(json.dumps(r), flush=True)
