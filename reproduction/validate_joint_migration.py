"""Validate the completed mixed R2R/RxR experiment and both exact eval sets."""
import gzip
import json
import math
import time
import sys
from pathlib import Path

RUN = Path('/mnt/autodl_tmp2/zhuyanhao/runs/navida_qwen3_20261102')

def validate():
    train = json.loads((RUN/'joint_r2r_rxr/train_results.json').read_text())
    assert train['epoch'] >= .999 and train['train_samples'] == 817098
    assert math.isfinite(train['train_loss'])
    assert (RUN/'artifacts/train_joint_r2r_rxr.done').exists()
    state = json.loads((RUN/'joint_r2r_rxr/trainer_state.json').read_text())
    checkpoint = RUN/'joint_r2r_rxr'/f"checkpoint-{state['global_step']}"
    sys.path.insert(0,str(RUN/"code"))
    from src.train.checkpointing import validate_full_checkpoint
    full = validate_full_checkpoint(checkpoint, 2)
    assert (checkpoint/'COMPLETE.json').exists()
    report = {'checked_unix':time.time(), 'train':train, 'checkpoint':full, 'eval':{}}
    for dataset, source, expected_n in [
        ('r2r', 'R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz',1839),
        ('rxr', 'RxR_VLNCE_v0/val_unseen/val_unseen_guide.json.gz',3669)]:
        name = 'eval_joint_'+dataset
        assert (RUN/'artifacts'/f'{name}.done').exists()
        with gzip.open(RUN/'data'/source) as f: episodes=json.load(f)['episodes']
        if dataset == 'rxr': episodes=[e for e in episodes if e['instruction']['language'] in ['en-US','en-IN']]
        expected={str(e['episode_id']) for e in episodes}
        rows=[json.loads(p.read_text()) for p in (RUN/name/'log').glob('stats_*.json')]
        assert len(rows)==len(expected)==expected_n
        assert {str(r['id']) for r in rows}==expected
        for row in rows:
            assert row['success'] in [0,1] and 0<=row['spl']<=1
            assert all(math.isfinite(float(v)) for k,v in row.items() if k!='id')
        report['eval'][dataset]={'episodes':len(rows),'metrics':{k:sum(float(r[k]) for r in rows)/len(rows) for k in rows[0] if k!='id'}}
    sr=100*report['eval']['r2r']['metrics']['success']
    spl=100*report['eval']['r2r']['metrics']['spl']
    report.update(r2r_sr_percent=sr,r2r_spl_percent=spl,paper_joint_sr_percent=47.7,paper_joint_spl_percent=41.5,meets_reference_sr=sr>=47.7,
        protocol_notes=['R2R+official StreamVLN RxR natural-count VLN+IDM mix from BASE, one epoch.', 'Qwen3-VL-4B replaces paper Qwen2.5-VL-3B; projector+LLM trained.', 'Linear scheduler follows paper Table7; R2R-only prior run used released-code cosine.', 'Table3 ablation protocol: 3 chunks, no entropy; greedy; history200; seed41.', 'No ScaleVLN or DAgger; threshold comparison does not isolate model architecture.'])
    target=RUN/'artifacts/joint_migration_verdict.json'
    target.write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
