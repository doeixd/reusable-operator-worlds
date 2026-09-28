"""PI-authorized launch override (2026-09-27, "Is ok. Just start"): reserve 4.5 GiB instead of 8.
The runner, its protocol fingerprint and gates are unchanged; precondition.json records required_gib."""
import sys
from row.experiments import o6_sealed_reroute as o6
o6.MIN_FREE_GIB = 4.5
if __name__ == "__main__":
    sys.argv = ["o6_sealed_reroute"] + sys.argv[1:]
    o6.main()
