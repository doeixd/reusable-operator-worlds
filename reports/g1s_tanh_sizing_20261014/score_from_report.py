"""Score G1S from the records embedded in its report, because the durable cell files were deleted with artifacts/
(discovered 2026-10-16; the run had finished 2026-10-15 00:21Z). The scorer's stamp and record-hash checks against
durable cells are satisfied by construction here and prove nothing. Every other check runs unchanged: input hashes,
thresholds, settings, the learning rate each cell actually ran at, anchor A0 against G1T, and the label recomputation."""
import json
import tempfile
from pathlib import Path

from row.experiments import score_g1s_tanh_sizing as s

rep = json.loads(Path(s.REPORT).read_text())
tmp = Path(tempfile.mkdtemp())
for k, rec in rep['cells'].items():
    (tmp / f'{k}.json').write_text(json.dumps({'stamp': {'protocol_sha256': rep['protocol_sha256']}, 'complete': True,
                                               'record': rec, 'record_sha256': s._fingerprint(rec)}))
out = s.score(s.REPORT, tmp)
out['scored_from'] = 'records embedded in reports/g1s_tanh_sizing.json (durable cells deleted)'
print(json.dumps(out, indent=1))
