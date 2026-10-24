"""Print or execute an RQ1 grid; every run remains a separate process."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
WEB={'quito_hour':[48,288,512],'extended_wikipedia':[7,28,59],'tailedts_202401':[24,48,96]}
GENERAL={d:[96,192,336,720] for d in ['ETTh1','ETTh2','ETTm1','ETTm2','Exchange','Weather']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['web','general'],required=True)
    p.add_argument('--data-root',type=Path,help='External data root; omit to allow bundled data')
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--datasets',nargs='+')
    p.add_argument('--horizons',type=int,nargs='+')
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--methods',nargs='+',default=['RAFT','SHARP','CREST','S&C'],choices=['RAFT','SHARP','CREST','S&C'])
    p.add_argument('--scale',type=int,default=200000)
    p.add_argument('--execute',action='store_true',help='Run the printed grid sequentially; otherwise only print it')
    p.add_argument('overrides',nargs=argparse.REMAINDER,help='Extra reproduce.py arguments after --')
    a=p.parse_args();grid=WEB if a.suite=='web' else GENERAL
    datasets=a.datasets or list(grid);seeds=a.seeds if a.seeds is not None else [0,1,42] if a.suite=='web' else [0,1,2]
    if set(datasets)-set(grid):p.error('dataset outside selected suite')
    extra=a.overrides[1:] if a.overrides[:1]==['--'] else a.overrides
    reserved=['--dataset','--method','--seed','--pred_len','--scale','--output','--config']
    if any(x.split('=')[0] in reserved for x in extra):p.error('set grid identity using sweep options, not trailing overrides')
    for ds in datasets:
        for h in a.horizons or grid[ds]:
            for seed in seeds:
                for method in a.methods:
                    out=a.output_root/ds/f'H{h}'/f'seed{seed}'/method.replace('&','_')
                    cmd=[sys.executable,str(ROOT/'reproduce.py'),'--dataset',ds,'--pred_len',str(h),
                        '--seed',str(seed),'--method',method,'--scale',str(a.scale),'--output',str(out.resolve())]
                    if a.data_root is not None:
                        cmd+=['--data_root',str(a.data_root.resolve())]
                    cmd+=extra
                    print(shlex.join(cmd),flush=True)
                    if a.execute:subprocess.run(cmd,check=True)


if __name__=='__main__':main()
