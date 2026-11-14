import hashlib
import json
from pathlib import Path
import time

from openai.lib._pydantic import to_strict_json_schema
from agents.pvc_mailbox import OwnedMailbox


class PVCStructuredProvider:
    def __init__(self,bridge,directory,*,guidance='',scope='native_full_search'):
        self.bridge = bridge
        self.directory = Path(directory)
        self.guidance = guidance
        self.scope = scope

    def verified_response(self,role,prompt,schema,instructions,observation,output_type):
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        for path in self.directory.glob('*/provenance.json'):
            provenance = json.loads(path.read_text())
            if provenance['prompt_sha256'] != digest or provenance['model_identifier'] != self.bridge.MODEL:
                continue
            directory = path.parent
            context = json.loads((directory/'request-context.json').read_text())
            if context != dict(role=role,scope=self.scope,instructions=instructions,observation=observation,schema=schema):
                continue
            raw = Path(self.bridge.MAILBOX)/(provenance['request_id']+'.response.json')
            if not raw.is_file():
                continue
            receipt = json.loads(raw.read_text())
            response = json.loads((directory/'response.json').read_text())
            if (receipt.get('ephemeral') is not True or receipt.get('id') != provenance['request_id']
                    or receipt.get('model') != self.bridge.MODEL or receipt.get('turn_id') != provenance['turn_id']
                    or receipt.get('response') != response or receipt.get('usage') != provenance['usage']
                    or hashlib.sha256(self.bridge.encoded(response).encode()).hexdigest() != provenance['response_sha256']):
                raise ValueError('已有Agent响应与实际认证回执或完整输入不一致')
            replay = dict(provenance,replayed_verified_response=True,
                source_usage=receipt['usage'],usage=dict(input_tokens=0,output_tokens=0,cached_input_tokens=0))
            return output_type.model_validate(response),replay
        return None

    def generate(self,role,instructions,observation,output_type,*,on_start,on_finish,on_usage,
                 max_output_tokens):
        schema = to_strict_json_schema(output_type)
        prompt = self.guidance+'\n'+instructions+'\n'+self.bridge.encoded(observation)
        if max_output_tokens is not None:
            prompt += f'\nMaximum output tokens: {max_output_tokens}.'
        cached = self.verified_response(role,prompt,schema,instructions,observation,output_type)
        if cached is not None:
            return cached
        def submitted(identifier):
            OwnedMailbox(self.bridge.MAILBOX).record_owner(identifier,role)
            on_start(identifier)

        request = self.bridge.submit(role,prompt,schema,'low',None,on_submit=submitted)
        directory = self.directory/request['id']
        directory.mkdir(parents=True,exist_ok=False)
        (directory/'request-context.json').write_text(json.dumps(dict(role=role,scope=self.scope,
            instructions=instructions,observation=observation,schema=schema),ensure_ascii=False,indent=2)+'\n')
        started = time.monotonic()
        try:
            receipt = self.bridge.wait_for(request)
        finally:
            on_finish(request['id'],time.monotonic()-started)
        on_usage(request['id'],receipt['usage'])
        response = output_type.model_validate(receipt['response'])
        provenance = dict(provider='authenticated_codex_app_server_pvc_bridge',
            model_identifier=receipt['model'],request_id=request['id'],turn_id=receipt['turn_id'],
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            response_sha256=hashlib.sha256(self.bridge.encoded(receipt['response']).encode()).hexdigest(),
            usage=receipt['usage'])
        for name,value in (('provenance.json',provenance),('response.json',receipt['response']),('receipt.json',receipt)):
            (directory/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        return response,provenance
