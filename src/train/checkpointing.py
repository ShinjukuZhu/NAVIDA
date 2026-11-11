"""Validate full DeepSpeed checkpoints before advertising them as resumable."""
import json,os,time
from pathlib import Path
import torch
from transformers import TrainerCallback
from safetensors import safe_open


def validate_full_checkpoint(path,world_size):
    path=Path(path)
    state=json.loads((path/'trainer_state.json').read_text())
    index=json.loads((path/'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        with safe_open(str(path/shard),framework='np') as f:
            assert set(f.keys())=={k for k,v in index['weight_map'].items() if v==shard}
    models=list(path.glob('global_step*/*model_states.pt'))
    optim=list(path.glob('global_step*/*optim_states.pt'))
    assert models and len(optim)==world_size,(models,optim,world_size)
    for p in optim:assert p.stat().st_size>1024**2,p
    model_state=torch.load(models[0],map_location='cpu',weights_only=False,mmap=True)
    assert model_state.get('lr_scheduler') is not None or (path/'scheduler.pt').exists(),'Missing scheduler state'
    del model_state
    rng=list(path.glob('rng_state*.pth'));assert len(rng)==world_size,(rng,world_size)
    for name in ['config.json','preprocessor_config.json','tokenizer_config.json']:assert (path/name).is_file()
    return {'global_step':state['global_step'],'epoch':state['epoch'],'world_size':world_size,'optimizer_files':[p.name for p in optim],'rng_files':[p.name for p in rng],'total_bytes':sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),'checked_unix':time.time(),'model_optimizer_scheduler_rng_and_trainer_state':True}


def latest_complete_checkpoint(output_dir):
    candidates=sorted(Path(output_dir).glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1,reverse=True)
    for path in candidates:
        marker=path/'COMPLETE.json'
        if not marker.exists():continue
        report=json.loads(marker.read_text())
        validate_full_checkpoint(path,report['world_size'])
        return str(path)
    if candidates:raise RuntimeError('Checkpoint directories exist without a validated COMPLETE marker; inspect before restarting from BASE')
    return None


class FullCheckpointCallback(TrainerCallback):
    def on_train_begin(self,args,state,control,**kwargs):
        # A resumed TrainerState can retain the previous checkpoint interval.
        state.save_steps = args.save_steps
        return control

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==10:control.should_save=True
        return control

    def on_save(self,args,state,control,**kwargs):
        if torch.distributed.is_initialized():torch.distributed.barrier()
        if state.is_world_process_zero:
            path=Path(args.output_dir)/f'checkpoint-{state.global_step}'
            report=validate_full_checkpoint(path,args.world_size)
            tmp=path/'COMPLETE.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(path/'COMPLETE.json')
            print('FULL_CHECKPOINT_VALIDATED',json.dumps(report),flush=True)
        return control
