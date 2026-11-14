import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--ready-file',type=Path,required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    assert not args.ready_file.exists()
    with (root/'parameld-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert not active
        parent = psutil.Process(int((root/'parameld-campaign.pid').read_text()))
        assert parent.cwd()==str(root) and any(value.endswith('run-parameld-campaign.py') for value in parent.cmdline())
        while True:
            children = [process for process in parent.children() if any(value.endswith('run-parameld-context.py')
                for value in process.cmdline())]
            assert len(children)==1
            child = children[0]
            wait = Path(f'/proc/{child.pid}/wchan').read_text().strip()
            if not child.children(recursive=True) and wait in ('locks_lock_inode_wait','hrtimer_nanosleep'):
                parent.suspend()
                child.suspend()
                if not child.children(recursive=True):
                    break
                child.resume()
                parent.resume()
            time.sleep(0.2)
        command = child.cmdline()
        model = command[command.index('--model')+1]
        sequence = command[command.index('--sequence')+1]
        scope = command[command.index('--scope')+1]
        for process in (parent,child):
            process.terminate()
            process.send_signal(signal.SIGCONT)
        while any(process.is_running() and process.status()!=psutil.STATUS_ZOMBIE for process in (parent,child)):
            time.sleep(0.2)
        state = dict(status='SOURCE_TRANSFER_READY',old_campaign_pid=parent.pid,old_context_pid=child.pid,
            model=model,sequence=sequence,scope=scope,interrupted_gpu_processes=0,
            original_pod_preserved=True,observed_unix_seconds=time.time())
        status_path = root/'frontier-source-migration.json'
        status_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(state,ensure_ascii=False),flush=True)
        while not args.ready_file.is_file():
            time.sleep(0.5)
        sys.path.insert(0,str(root/'parameld-source/source/paramorph'))
        from agents.pvc_mailbox import OwnedMailbox
        from storage.experiment_store import ExperimentStore
        OwnedMailbox(root/'agent-mailbox').activate()
        case = root/('parameld-cp-fsdp' if scope=='cp-fsdp' else 'parameld')/f'{model}-s{sequence}'
        store = ExperimentStore(case/'experiments.sqlite')
        store.add_search_state('SEARCH_CONTROLLER_MIGRATION',dict(state,source=json.loads(args.ready_file.read_text())))
        for request in store.profile_requests():
            if request['state']=='RUNNING':
                store.update_profile_request(request['request_id'],state='PENDING')
        store.connection.close()
        subprocess.run(['bash',str(root/'launch-full-campaign.sh')],check=True)
        state.update(status='RESUMED',new_campaign_pid=int((root/'parameld-campaign.pid').read_text()))
        status_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
