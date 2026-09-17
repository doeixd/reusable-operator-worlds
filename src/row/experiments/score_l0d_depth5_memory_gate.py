import argparse,json,math
from pathlib import Path
from row.experiments.so1_storage import digest,fingerprint
def score(root=Path('artifacts/l0d_depth5_memory_gate_v2'),output=Path('reports/l0d_depth5_memory_gate_v2.json')):
    root,output=Path(root),Path(output); m=json.loads((root/'manifest.json').read_text()); o=json.loads(output.read_text()); p,s=m['protocol'],m['protocol_sha256']
    if fingerprint(p)!=s or o['protocol']!=p or o['protocol_sha256']!=s: raise ValueError('protocol mismatch')
    for path,e in p['input_sha256'].items():
        if digest(Path(path))!=e: raise ValueError(f'input changed: {path}')
    if digest(Path(__file__).with_name('l0d_depth5_memory_gate.py'))!=p['implementation_sha256']: raise ValueError('runner changed')
    c=json.loads((root/'cell.json').read_text())
    if not c.get('complete') or c['stamp']!={'protocol_sha256':s} or fingerprint(c['record'])!=c['record_sha256'] or c['record']!=o['cell']: raise ValueError('cell mismatch')
    r=c['record']
    if r['blocks']!=243 or not r['support_mse_equal'] or not math.isfinite(r['selected_support_mse']): raise ValueError('gate failed')
    if json.loads((root/'status.json').read_text())['state']!='complete' or json.loads((root/'exit.json').read_text())['exit_code']!=0: raise ValueError('operational failure')
    return {'valid':True,'blocks':r['blocks'],'largest_terminal_bytes':r['largest_terminal_bytes'],'selected_route':r['selected_route'],'selected_support_mse':r['selected_support_mse'],'seconds':r['seconds'],'interpretation':o['summary']['interpretation']}
def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path('artifacts/l0d_depth5_memory_gate'));a.add_argument('--report',type=Path,default=Path('reports/l0d_depth5_memory_gate.json'));x=a.parse_args();print(json.dumps(score(x.root,x.report),indent=2))
if __name__=='__main__':main()
