"""Bounded depth-five exhaustive evaluator; feasibility gate only."""
from __future__ import annotations
import argparse, itertools, json, math, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from row.experiments import audit_j2a_staged_library as j2a
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_so1r_route_only import unflatten
from row.experiments.l0d_depth4_execution_gate import DepthLibrary
from row.experiments.preflight_l0d import load_source
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN=Path('L0D_DEPTH5_MEMORY_GATE_PLAN.md'); ROOT=Path('artifacts/l0d_depth5_memory_gate'); OUTPUT=Path('reports/l0d_depth5_memory_gate.json')
NAME, WORLD, MODEL_SEED, DEPTH, TASKS, SEED, BLOCK = 'STAGED5000', 0, 5000, 5, 1, 2705, 1024

def protocol():
    return {'id':'l0d-depth5-memory-gate-v1','git_commit':git_commit(),'name':NAME,'world':WORLD,'model_seed':MODEL_SEED,
            'depth':DEPTH,'tasks':TASKS,'seed':SEED,'block':BLOCK,'slots':12,'support':128,
            'input_sha256':{p.as_posix():digest(p) for p in (PLAN,Path('configs/v1.yaml'),j2a.OUTPUT)},
            'implementation_sha256':digest(Path(__file__))}

def task(cfg, world):
    choices=list(itertools.product(range(cfg.world.teacher_primitives),repeat=DEPTH)); rng=np.random.default_rng(np.random.SeedSequence([SEED,WORLD,DEPTH])); choice=choices[int(rng.integers(len(choices)))]
    x=np.random.default_rng(np.random.SeedSequence([SEED,WORLD,1])).normal(size=(cfg.world.examples_per_task,cfg.world.state_dim)); q=np.random.default_rng(np.random.SeedSequence([SEED,WORLD,10000])).normal(size=(cfg.world.evaluation_examples,cfg.world.state_dim)); y=x.copy(); qy=q.copy(); lib=world.tasks[0].teacher_library
    for p in choice: y=lib[p](y); qy=lib[p](qy)
    return {'program':tuple(choice),'train_x':x,'train_y':y,'eval_x':q,'eval_y':qy}

def chunked_mse(library,x,y,depth,block):
    n,d=x.shape; total=library.slots**depth; values=[]; largest=0; blocks=0
    with torch.no_grad():
        for start in range(0,total,block):
            count=min(block,total-start); ids=np.arange(start,start+count,dtype=np.int64); routes=np.empty((count,depth),dtype=np.int64); digits=ids.copy()
            for pos in range(depth-1,-1,-1): routes[:,pos]=digits%library.slots; digits//=library.slots
            z=x.unsqueeze(0).expand(count,n,d).reshape(-1,d)
            for pos in range(depth):
                candidates=library.candidates(z).reshape(count,n,library.slots,d)
                z=candidates[torch.arange(count)[:,None],torch.arange(n)[None,:],torch.tensor(routes[:,pos])].reshape(-1,d)
            values.append(torch.mean((z.reshape(count,n,d)-y.unsqueeze(0))**2,dim=(1,2)))
            largest=max(largest,int(z.numel()*z.element_size())); blocks+=1
    return torch.cat(values), blocks, largest

def direct_support(library,x,y,route):
    with torch.no_grad(): return float(torch.mean((library.hard(x,route)-y)**2))

def measure():
    started=time.perf_counter(); cfg,world,model,_,_=load_source(NAME,WORLD); library=DepthLibrary(model); before=library_sha(model); t=task(cfg,world); x=torch.tensor(t['train_x'],dtype=torch.float32); y=torch.tensor(t['train_y'],dtype=torch.float32)
    values,blocks,largest=chunked_mse(library,x,y,DEPTH,BLOCK); best=int(torch.argmin(values)); route=unflatten(best,library.slots,DEPTH); selected=float(values[best]); direct=direct_support(library,x,y,route)
    if library_sha(model)!=before: raise ValueError('library changed')
    return {'name':NAME,'world':WORLD,'model_seed':MODEL_SEED,'depth':DEPTH,'program':list(t['program']),'library_sha256':before,'blocks':blocks,'expected_blocks':math.ceil(library.slots**DEPTH/BLOCK),'selected_route':route,'selected_support_mse':selected,'direct_support_mse':direct,'support_mse_equal':selected==direct,'largest_terminal_bytes':largest,'seconds':time.perf_counter()-started,'terminal_lower_bound_bytes':128*12**5*16*4}

def validate(r):
    if r['blocks']!=r['expected_blocks'] or r['expected_blocks']!=243 or not r['support_mse_equal']: raise ValueError('block/equivalence gate failed')
    for k in ('selected_support_mse','direct_support_mse','seconds'):
        if not math.isfinite(r[k]) or r[k]<0: raise ValueError('invalid metric')
    if len(r['selected_route'])!=DEPTH: raise ValueError('route depth')

def run():
    ROOT.mkdir(parents=True,exist_ok=True); expected=protocol(); sha=fingerprint(expected); manifest={'protocol':expected,'protocol_sha256':sha}
    with writer_lock(ROOT/'launcher.lock'):
        if (ROOT/'manifest.json').exists() and json.loads((ROOT/'manifest.json').read_text())!=manifest: raise ValueError('protocol mismatch')
        atomic_json(ROOT/'manifest.json',manifest)
        try:
            p=ROOT/'cell.json'
            if p.exists():
                s=json.loads(p.read_text());
                if not s.get('complete') or s['stamp']!={'protocol_sha256':sha} or fingerprint(s['record'])!=s['record_sha256']: raise ValueError('cell integrity')
                r=s['record']; log_line(ROOT/'run.log','reused validated cell')
            else:
                log_line(ROOT/'run.log',f'LAUNCH {sha}'); r=measure(); validate(r); atomic_json(p,{'stamp':{'protocol_sha256':sha},'complete':True,'record':r,'record_sha256':fingerprint(r),'finished_utc':now()}); log_line(ROOT/'run.log',f'cell finished {r["seconds"]:.3f}s')
            validate(r); atomic_json(OUTPUT,{**manifest,'complete':True,'cell':r,'summary':{'classification':'MEMORY_SAFE_SEARCH','interpretation':'depth-five memory gate only; no PX7 verdict'},'finished_utc':now()}); atomic_json(ROOT/'status.json',{'state':'complete','cells_done':1,'cells_total':1,'updated_utc':now()}); atomic_json(ROOT/'exit.json',{'exit_code':0,'complete':True,'finished_utc':now()})
        except BaseException:
            atomic_json(ROOT/'status.json',{'state':'failed','cells_done':0,'cells_total':1,'updated_utc':now()}); atomic_json(ROOT/'exit.json',{'exit_code':1,'complete':False,'finished_utc':now()}); atomic_json(ROOT/'error.json',{'traceback':traceback.format_exc(),'finished_utc':now()}); raise

def main():
    argparse.ArgumentParser().parse_args(); torch.set_num_threads(1); require_clean_code(OUTPUT)
    for c in ('tools/check_prereg.py','tools/check_invalid.py'): subprocess.run([sys.executable,c],check=True)
    run(); print(f'Depth-five memory gate report: {OUTPUT}')
if __name__=='__main__': main()
