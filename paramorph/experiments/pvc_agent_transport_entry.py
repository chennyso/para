import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('PARAMELD_SOURCE_ROOT',ROOT/'parameld-source/source'))
sys.path.insert(0,str(SOURCE/'paramorph'))

from agents import pvc_transport as bridge

CONFIG = json.loads((ROOT/'bridge-config.json').read_text())
CONFIG['pod'] = os.environ.get('PARAMELD_BRIDGE_POD', CONFIG['pod'])
bridge.POD = CONFIG['pod']
bridge.MODEL = CONFIG['model']
bridge.MAILBOX = Path(CONFIG['mailbox'])
bridge.LEASE = None
bridge.ARTIFACT_DIRECTORY = ROOT
OUTBOX = ROOT/'agent-relay-outbox'


def remote(*arguments, data=None, timeout=18):
    action = str(arguments[-1])
    assert action in ('pull','push')
    receipt = None
    if action=='push':
        receipt = json.loads(data)
        assert len(receipt['id'])==64 and all(character in '0123456789abcdef' for character in receipt['id'])
        OUTBOX.mkdir(exist_ok=True)
        cached = OUTBOX/(receipt['id']+'.json')
        if cached.is_file():
            assert json.loads(cached.read_text())==receipt
        else:
            bridge.atomic_write(cached,receipt)
    argv = ['kubectl','--context','prd','--request-timeout=12s','exec']
    if data is not None:
        argv.append('-i')
    argv += ['-n','default',CONFIG['pod'],'-c','trainer','--','python','-B',
             CONFIG['remote_entry'],action]
    failures = 0
    transient = ('timeout','deadline exceeded','connection reset','error dialing backend',
        'error reading from error stream','unknown stream id','unexpected eof','tls handshake',
        'connection refused','i/o timeout','unable to connect','stream error','http2:',
        'error trying to reach service: sync from client')
    while True:
        try:
            result = subprocess.run(argv,input=data,text=True,capture_output=True,
                timeout=timeout,env=dict(os.environ,KUBECTL_REMOTE_COMMAND_WEBSOCKETS='false',GODEBUG='http2client=0'))
        except subprocess.TimeoutExpired:
            result = None
        if result is not None and result.returncode == 0:
            if action == 'pull':
                envelope = json.loads(result.stdout)
                result.stdout = bridge.encoded(bridge.decode_transport_payload(envelope))
            if receipt is not None:
                (OUTBOX/(receipt['id']+'.sent')).touch()
            if failures:
                print(bridge.encoded(dict(status='TRANSPORT_RECOVERED',action=action,attempts=failures)),flush=True)
            return result.stdout
        detail = '' if result is None else result.stderr.lower()
        empty_server_response=detail.strip()=='error from server:'
        if result is not None and not empty_server_response and not any(marker in detail for marker in transient):
            print(result.stderr,file=sys.stderr,flush=True)
            raise subprocess.CalledProcessError(result.returncode,argv,output=result.stdout,stderr=result.stderr)
        failures += 1
        print(bridge.encoded(dict(status='TRANSPORT_RECONNECTING',action=action,attempt=failures,
            reason='EMPTY_SERVER_RESPONSE' if empty_server_response else detail.strip())),flush=True)
        time.sleep(min(10,failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['relay','pull','push','smoke'])
    parser.add_argument('--codex',default='codex')
    args = parser.parse_args()
    if args.mode == 'relay':
        bridge.command = remote
        OUTBOX.mkdir(exist_ok=True)
        for cached in sorted(OUTBOX.glob('*.json')):
            if not cached.with_suffix('.sent').is_file():
                remote('push',data=cached.read_text())
        bridge.relay(args.codex)
        return
    bridge.MAILBOX.mkdir(exist_ok=True)
    if args.mode == 'pull':
        print(bridge.encoded(bridge.encode_transport_payload(bridge.pull())))
    elif args.mode == 'push':
        bridge.push(json.loads(sys.stdin.read()))
    else:
        schema = dict(type='object',properties=dict(status=dict(type='string',enum=['ready'])),
                      required=['status'],additionalProperties=False)
        request = bridge.submit('BridgeVerification','Return status ready.',schema,'low',None)
        receipt = bridge.wait_for(request)
        assert receipt['response'] == {'status':'ready'} and receipt['ephemeral']
        print(bridge.encoded(dict(status='VERIFIED',model=receipt['model'],usage=receipt['usage'],
                                 request_id=request['id'])),flush=True)


if __name__=='__main__':
    main()
