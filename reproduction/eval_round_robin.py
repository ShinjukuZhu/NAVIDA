"""Eight deterministic Habitat shards, assigned to healthy idle GPUs round-robin."""
import gzip,json,math,os,socket,subprocess,time,urllib.request
from pathlib import Path

SHARDS=8

def assignments(gpus):
    assert 1<=len(gpus)<=4
    return [(shard,gpus[shard%len(gpus)]) for shard in range(SHARDS)]

def summarize(run,code,dataset,name):
    source='R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz' if dataset=='r2r' else 'RxR_VLNCE_v0/val_unseen/val_unseen_guide.json.gz'
    eps=json.load(gzip.open(code/'data'/source))['episodes']
    if dataset=='rxr':eps=[e for e in eps if e['instruction']['language'] in ['en-US','en-IN']]
    expected={str(e['episode_id']) for e in eps}
    rows=[json.loads(p.read_text()) for p in (run/name/'log').glob('stats_*.json')]
    assert len(rows)==len(expected) and {str(r['id']) for r in rows}==expected,'Incomplete/duplicate eval coverage'
    assert len(rows)==(1839 if dataset=='r2r' else 3669)
    metrics={k:sum(float(r[k]) for r in rows)/len(rows) for k in rows[0] if k!='id'}
    assert all(math.isfinite(v) for v in metrics.values())
    report={'n':len(rows),'metrics':metrics,'shards':SHARDS,'dispatch':'round_robin'}
    (run/'artifacts'/f'{name}_summary.json').write_text(json.dumps(report,indent=2))
    return report

def evaluate_once(d,model,dataset,name):
    done=d.RUN/'artifacts'/f'{name}.done'
    if done.exists():return
    selected=d.wait_available(1,allowed_counts=[1,2,3,4])
    mapping=assignments(selected)
    gpu_indices={line.split(',')[1].strip():int(line.split(',')[0]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
    manifest={'created_unix':time.time(),'model':str(model),'dataset':dataset,'split_num':SHARDS,'seed':41,'assignment':'gpu[split_id % number_of_gpus]','mapping':[{'split_id':i,'gpu_uuid':gpu,'gpu_index':gpu_indices[gpu]} for i,gpu in mapping],'protocol':{'chunks':3,'entropy':False,'max_history':200,'max_new_tokens':128,'attention':'flash_attention_2','generation':'greedy'}}
    (d.RUN/'artifacts'/f'{name}_dispatch.json').write_text(json.dumps(manifest,indent=2))
    d.status(name,selected_gpus=selected,split_num=SHARDS,mapping=manifest['mapping'])
    servers={};workers=[];logs=[]
    try:
        # One model per GPU; Habitat workers on that GPU share its HTTP server.
        for gpu in selected:
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            f=(d.RUN/'logs'/f'{name}_server_gpu{gpu_indices[gpu]}.log').open('a');logs.append(f)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4','PYTHONPATH':'.'}
            p=subprocess.Popen([d.PY3,'-u','src/eval/model_server.py','--model-path',str(model),'--gpu',gpu,'--port',str(port)],cwd=d.CODE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            servers[gpu]=(p,port)
            (d.RUN/'logs'/f'{name}_server_gpu{gpu_indices[gpu]}.pid').write_text(str(p.pid))
        pending=set(selected);deadline=time.monotonic()+600
        while pending:
            for gpu in list(pending):
                p,port=servers[gpu]
                if p.poll() is not None:raise RuntimeError(f'Model server exited on GPU{gpu_indices[gpu]}: {p.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:health=json.load(response)
                except (OSError,ValueError):continue
                assert health['pid']==p.pid and health['model_path']==str(model),'Unexpected server identity'
                assert health['attention']=='flash_attention_2'
                pending.remove(gpu)
            if time.monotonic()>deadline:raise TimeoutError('Model server startup; inspect logs before retrying')
            if pending:time.sleep(2)
        for shard,gpu in mapping:
            marker=d.RUN/'artifacts'/f'{name}_split{shard}.done'
            if marker.exists():continue
            port=servers[gpu][1]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'NAVIDA_MODEL_URL':f'http://127.0.0.1:{port}','NAVIDA_SAVE_VIDEO':'0','NAVIDA_EXEC_CHUNKS':'3','PYTHONPATH':'.','OMP_NUM_THREADS':'2'}
            cmd=[d.PYH,'-u','src/eval/eval.py','--exp-config',f'config/vln_{dataset}.yaml','--split-num',str(SHARDS),'--split-id',str(shard),'--model-path',str(model),'--result-path',str(d.RUN/name),'--max-action-history','200']
            f=(d.RUN/'logs'/f'{name}_split{shard}.log').open('a');logs.append(f)
            p=subprocess.Popen(cmd,cwd=d.CODE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append((p,shard,marker))
            (d.RUN/'logs'/f'{name}_split{shard}.pid').write_text(str(p.pid))
        remaining=workers.copy()
        while remaining:
            for gpu,(p,port) in servers.items():
                if p.poll() is not None:raise RuntimeError(f'Model server exited during evaluation: GPU{gpu_indices[gpu]}')
            for item in remaining.copy():
                p,shard,marker=item;code=p.poll()
                if code is None:continue
                if code:raise subprocess.CalledProcessError(code,p.args)
                marker.write_text(str(time.time()));remaining.remove(item)
            if remaining:time.sleep(2)
        summarize(d.RUN,d.CODE,dataset,name)
        done.write_text(str(time.time()))
    finally:
        # These Popen objects and process groups were created by this invocation.
        for p,_,_ in workers:d.stop_created_process(p)
        for p,_ in servers.values():d.stop_created_process(p)
        for f in logs:f.close()
