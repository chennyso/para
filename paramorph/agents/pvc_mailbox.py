import json
import os
from pathlib import Path
import re
import time

import psutil


class OwnedMailbox:
    def __init__(self,directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)

    def write(self,path,document):
        staging = path.with_suffix(f'.{time.time_ns()}.staging')
        with staging.open('x') as stream:
            json.dump(document,stream,ensure_ascii=False,sort_keys=True)
        staging.rename(path)

    def identifier(self,value):
        if re.fullmatch('[a-f0-9]{64}',value) is None:
            raise ValueError('PVC请求身份无效')
        return value

    def record_owner(self,identifier,role):
        identifier = self.identifier(identifier)
        process = psutil.Process()
        owner = dict(schema_version=1,pod=os.environ['HOSTNAME'],pid=process.pid,
            process_created=process.create_time(),role=role)
        path = self.directory/f'{identifier}.owner.json'
        if path.is_file():
            if json.loads(path.read_text()) != owner:
                raise ValueError('相同请求的控制器归属发生冲突')
            return
        self.write(path,owner)

    def activate(self):
        self.write(self.directory/'ownership-protocol.json',dict(schema_version=1,
            ownership_required=True,activated_unix_seconds=time.time()))

    def active(self,identifier):
        owner_path = self.directory/f'{identifier}.owner.json'
        if not owner_path.is_file():
            protocol = self.directory/'ownership-protocol.json'
            return not (protocol.is_file() and json.loads(protocol.read_text())['ownership_required'])
        owner = json.loads(owner_path.read_text())
        if owner['pod'] != os.environ['HOSTNAME']:
            return None
        return any(process.info['pid']==owner['pid'] and process.info['create_time']==owner['process_created']
            and process.info['status']!=psutil.STATUS_ZOMBIE
            for process in psutil.process_iter(['pid','create_time','status']))

    def pull(self,validate,limit):
        for path in sorted(self.directory.glob('*.request.json'),key=lambda item:item.stat().st_mtime_ns):
            identifier = self.identifier(path.name.removesuffix('.request.json'))
            cancelled = self.directory/f'{identifier}.cancelled.json'
            if ((self.directory/f'{identifier}.response.json').is_file() or cancelled.is_file()
                    or path.stat().st_size > limit):
                continue
            active = self.active(identifier)
            if active is None:
                continue
            if not active:
                self.write(cancelled,dict(status='CANCELLED_REQUEST_OWNER_UNAVAILABLE',request_id=identifier,
                    observed_unix_seconds=time.time(),original_request_preserved=True))
                continue
            request = json.loads(path.read_text())
            validate(request)
            if request['deadline'] is None or request['deadline'] > time.time():
                return request
        return None
