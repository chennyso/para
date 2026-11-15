import hashlib
import base64
import gzip
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time
import tomllib

from agents.pvc_mailbox import OwnedMailbox


MODEL = None
MAILBOX = None
ARTIFACT_DIRECTORY = None
ROLES = {'PerformanceAnalysisAgent','JointStrategyAgent','ConstraintValidationAgent',
    'ExperimentInvestigationAgent','BridgeVerification'}
LIMIT = 512*1024
LEASE = None


class AgentResponseFailure(RuntimeError):
    def __init__(self,receipt):
        self.receipt = receipt
        super().__init__(receipt['detail'])


def encoded(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def encode_transport_payload(value):
    content = encoded(value).encode()
    return dict(encoding='gzip_base64_v1', raw_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        payload=base64.b64encode(gzip.compress(content, mtime=0)).decode())


def decode_transport_payload(envelope):
    if envelope['encoding'] != 'gzip_base64_v1':
        raise ValueError('Agent传输编码版本无效')
    content = gzip.decompress(base64.b64decode(envelope['payload'], validate=True))
    if len(content) != envelope['raw_bytes'] or hashlib.sha256(content).hexdigest() != envelope['sha256']:
        raise ValueError('Agent传输内容长度或摘要不同')
    return json.loads(content)


def atomic_write(path,value):
    path = Path(path)
    staging = path.with_suffix(f'.{time.time_ns()}.staging')
    with staging.open('x') as stream:
        stream.write(encoded(value)+'\n')
    staging.rename(path)


def validate(request):
    if (set(request) != {'id','role','model','effort','prompt','schema','deadline'}
            or re.fullmatch('[a-f0-9]{64}',request['id']) is None
            or request['role'] not in ROLES or request['model'] != MODEL
            or request['effort'] not in ('low','medium')
            or not isinstance(request['prompt'],str) or not request['prompt']
            or not isinstance(request['schema'],dict) or len(encoded(request).encode()) > LIMIT):
        raise ValueError('请求超出已经登记的实际Agent模型和结构化输入范围')
    if request['deadline'] is not None or LEASE is not None:
        raise ValueError('当前PVC Agent中转不设置请求或运行时限')


def submit(role,prompt,schema,effort,timeout,on_submit=None):
    if timeout is not None:
        raise ValueError('当前PVC Agent请求需要使用None时限')
    body = dict(role=role,model=MODEL,effort=effort,prompt=prompt,schema=schema)
    identifier = hashlib.sha256(encoded(dict(body,attempt=time.time_ns())).encode()).hexdigest()
    request = dict(body,id=identifier,deadline=None)
    validate(request)
    OwnedMailbox(MAILBOX).record_owner(identifier,role)
    if on_submit is not None:
        on_submit(identifier)
    atomic_write(Path(MAILBOX)/(identifier+'.request.json'),request)
    return request


def wait_for(request):
    validate(request)
    response_path = Path(MAILBOX)/(request['id']+'.response.json')
    cancelled_path = Path(MAILBOX)/(request['id']+'.cancelled.json')
    while True:
        if response_path.is_file():
            response = json.loads(response_path.read_text())
            if response['id'] != request['id']:
                raise ValueError('实际Agent响应身份不一致')
            if 'error' in response:
                raise AgentResponseFailure(response)
            return response
        if cancelled_path.is_file():
            raise RuntimeError('当前Agent请求的控制器归属已经终止')
        time.sleep(0.5)


def pull():
    return OwnedMailbox(MAILBOX).pull(validate,LIMIT)


def push(response):
    validate(json.loads((Path(MAILBOX)/(response['id']+'.request.json')).read_text()))
    destination = Path(MAILBOX)/(response['id']+'.response.json')
    if destination.is_file():
        if json.loads(destination.read_text()) != response:
            raise ValueError('相同Agent请求已有内容不同的真实响应')
        return
    atomic_write(destination,response)


def infer(executable,request):
    validate(request)
    started = time.monotonic()
    inbox = queue.Queue()
    directory = Path(ARTIFACT_DIRECTORY)/'agent-api-progress'
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/(request['id']+'.json')
    stream_path = directory/(request['id']+'.agent-message.jsonl')
    messages = {}
    progress = dict(request_id=request['id'],role=request['role'],model=request['model'],status='INITIALIZING')
    atomic_write(path,progress)
    settings_path = Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'config.toml'
    settings = tomllib.loads(settings_path.read_text()) if settings_path.is_file() else {}
    disabled = [option for name in settings.get('mcp_servers', {})
        for option in ('-c',f'mcp_servers.{json.dumps(name)}.enabled=false')]
    command = [executable,*disabled,'-c','web_search="disabled"','-c','features.shell_tool=false',
        '-c','features.hooks=false','-c','skills.include_instructions=false','-c','features.apps=false',
        '-c','features.plugins=false','-c','mcp_servers={}','-c','default_permissions="parameld_pvc"',
        '-c','permissions.parameld_pvc.filesystem={":minimal"="read"}',
        '-c','permissions.parameld_pvc.network.enabled=false','app-server','--listen','stdio://']
    with subprocess.Popen(command,cwd='/dev',stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,text=True,encoding='utf-8',bufsize=1,start_new_session=True) as process:
        def reader():
            for line in process.stdout:
                inbox.put(line)
            inbox.put(None)

        def send(document):
            process.stdin.write(encoded(document)+'\n')
            process.stdin.flush()

        def receive():
            line = inbox.get()
            if line is None:
                raise RuntimeError('实际Agent API在交付完整响应前退出')
            message = json.loads(line)
            method = message.get('method')
            if method=='item/agentMessage/delta':
                params = message['params']
                delta = params['delta']
                identifier = params['itemId']
                messages.setdefault(identifier,[]).append(delta)
                progress['output_characters'] = progress.get('output_characters',0)+len(delta)
                progress['last_output_tail'] = delta[-256:]
                with stream_path.open('a') as stream:
                    stream.write(encoded(dict(item_id=identifier,delta=delta))+'\n')
            if method=='thread/tokenUsage/updated':
                progress['token_usage'] = message['params']['tokenUsage']['total']
            if method in ('turn/completed','thread/tokenUsage/updated') or time.monotonic()-progress.get('last_write',started) >= 10:
                progress.update(last_event=method or 'rpc_reply',observed_unix_seconds=time.time(),last_write=time.monotonic())
                if method=='turn/completed':
                    progress['status'] = message['params']['turn']['status']
                atomic_write(path,progress)
            if 'method' in message and 'id' in message:
                send(dict(id=message['id'],error=dict(code=-32601,message='No external tools are authorized')))
                raise RuntimeError('实际Agent请求调用了未授权的外部工具')
            return message

        def reply(identifier):
            while True:
                message = receive()
                if message.get('id')==identifier:
                    if 'error' in message:
                        raise RuntimeError(str(message['error'].get('message','Agent API请求失败'))[:350])
                    return message['result']

        threading.Thread(target=reader,daemon=True).start()
        try:
            send(dict(id=1,method='initialize',params=dict(clientInfo=dict(
                name='parameld_pvc',title='ParaMeld PVC',version='2'),capabilities=dict(experimentalApi=True))))
            reply(1)
            send(dict(method='initialized'))
            send(dict(id=2,method='thread/start',params=dict(model=request['model'],cwd='/dev',
                approvalPolicy='never',permissions='parameld_pvc',ephemeral=True,serviceName='parameld_pvc',
                baseInstructions='Use supplied ParaMeld evidence and return the requested JSON. Do not call tools.',
                developerInstructions='Preserve unknown evidence. Never invent measurements, task identifiers or predictions.')))
            thread = reply(2)['thread']
            if thread.get('ephemeral') is not True:
                raise RuntimeError('实际Agent API没有确认独立内存会话')
            send(dict(id=3,method='turn/start',params=dict(threadId=thread['id'],model=request['model'],
                effort=request['effort'],cwd='/dev',approvalPolicy='never',
                input=[dict(type='text',text=request['prompt'])],outputSchema=request['schema'])))
            turn_id = reply(3)['turn']['id']
            progress.update(status='TURN_RUNNING',thread_id=thread['id'],turn_id=turn_id,
                api_process_pid=process.pid,observed_unix_seconds=time.time())
            atomic_write(path,progress)
            response,usage = None,None
            while True:
                message = receive()
                method,params = message.get('method'),message.get('params', {})
                if params.get('turnId',turn_id) != turn_id:
                    continue
                if method=='item/completed':
                    item = params['item']
                    if item['type'] in ('commandExecution','fileChange','mcpToolCall','dynamicToolCall','webSearch'):
                        raise RuntimeError('实际Agent执行了未授权工具')
                    if item['type']=='agentMessage' and item.get('phase')!='commentary':
                        response = item['text']
                elif method=='thread/tokenUsage/updated':
                    usage = params['tokenUsage']['total']
                elif method=='turn/completed':
                    turn = params['turn']
                    actual_usage = None if usage is None else dict(input_tokens=usage['inputTokens'],
                        output_tokens=usage['outputTokens'],cached_input_tokens=usage['cachedInputTokens'])
                    if turn['status']!='completed' or response is None or usage is None:
                        detail = turn.get('error') or {}
                        receipt = dict(id=request['id'],error='AgentResponseFailure',
                            detail=str(detail.get('message','实际Agent没有交付完整结果与用量'))[:350],
                            error_details=detail,
                            status=turn['status'],usage=actual_usage,token_usage_complete=usage is not None,
                            turn_id=turn_id,model=request['model'],ephemeral=True,
                            incomplete_output={key:''.join(parts) for key,parts in messages.items()},
                            elapsed_seconds=time.monotonic()-started)
                        atomic_write(directory/(request['id']+'.failed-receipt.json'),receipt)
                        raise AgentResponseFailure(receipt)
                    return dict(id=request['id'],response=json.loads(response),
                        usage=actual_usage,model=request['model'],
                        turn_id=turn_id,ephemeral=True,elapsed_seconds=time.monotonic()-started)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL)
            process.wait()


def command(action,data=None):
    raise RuntimeError('PVC中转需要配置已经授权的Pod读取和响应交付接口')


def relay(executable):
    while True:
        request = json.loads(command('pull'))
        if request is None:
            time.sleep(2)
            continue
        try:
            response = infer(executable,request)
        except AgentResponseFailure as error:
            command('push',data=encoded(error.receipt))
            raise
        except (ValueError,RuntimeError,queue.Empty) as error:
            response = dict(id=request['id'],error=type(error).__name__,detail=str(error)[:350])
            command('push',data=encoded(response))
            raise
        command('push',data=encoded(response))
        print(encoded(dict(status='COMPLETED',role=request['role'],request_id=request['id'])),flush=True)
