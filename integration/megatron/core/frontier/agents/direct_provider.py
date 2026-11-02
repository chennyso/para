import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

DEFAULT_BASE_URL = 'https://api.wkkapi.com'
DEFAULT_MODEL = 'gpt-5.5'


class DirectAPIProvider:
    def __init__(self, directory, model=None, timeout_seconds=600,
                 base_url=None, api_key=None, reasoning_effort=None):
        self.directory = Path(directory).resolve()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.base_url = (base_url or os.environ.get('PARAMORPH_LLM_BASE_URL')
                         or DEFAULT_BASE_URL).rstrip('/')
        self.api_key = (api_key or os.environ.get('PARAMORPH_LLM_API_KEY')
                        or os.environ.get('OPENAI_API_KEY'))
        if not self.api_key:
            raise RuntimeError('DirectAPIProvider 需要 API key（PARAMORPH_LLM_API_KEY 或 OPENAI_API_KEY）')
        self.directory.mkdir(parents=True, exist_ok=True)

    def generate(self, role, instructions, observation, output_type, *, on_start=None, on_finish=None,
                 on_usage=None, max_output_tokens=None):
        invocation_started = time.perf_counter()
        specification = to_strict_json_schema(output_type)
        requested_model = self.model or DEFAULT_MODEL
        identity = dict(role=role, instructions=instructions,
            observation=observation, schema=specification, model=requested_model,
            provider='direct_api', base_url=self.base_url)
        if self.reasoning_effort is not None:
            identity['reasoning_effort'] = self.reasoning_effort
        identifier = hashlib.sha256(json.dumps(identity,
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        directory = self.directory / identifier
        prompt = ('Use only the supplied observation. Do not call tools, access files, edit source code, '
                  'or invent measurements. Return one object matching the output schema.\n'
                  + instructions + '\n' + json.dumps(observation, ensure_ascii=False, allow_nan=False))
        required = ('prompt.txt', 'schema.json', 'response.json', 'provenance.json')
        if directory.exists() and not all((directory / name).is_file() for name in required):
            incomplete = self.directory / 'incomplete-calls'
            incomplete.mkdir(exist_ok=True)
            directory.rename(incomplete / (identifier + '-' + str(time.time_ns())))
        if directory.exists():
            provenance = json.loads((directory / 'provenance.json').read_text())
            if ((directory / 'prompt.txt').read_text() != prompt
                    or json.loads((directory / 'schema.json').read_text()) != specification
                    or provenance['model_override'] != self.model
                    or provenance.get('reasoning_effort') != self.reasoning_effort
                    or provenance['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest()
                    or provenance['response_sha256'] != hashlib.sha256((directory / 'response.json').read_bytes()).hexdigest()):
                raise RuntimeError('existing actual agent response differs from its registered prompt, schema or provenance')
            output = output_type.model_validate_json((directory / 'response.json').read_text())
            return output, provenance | dict(replayed_verified_response=True,
                invocation_seconds=time.perf_counter() - invocation_started,
                new_llm_seconds=0.0, new_input_tokens=0, new_output_tokens=0)
        directory.mkdir(exist_ok=False)
        (directory / 'schema.json').write_text(json.dumps(specification, indent=2) + '\n')
        (directory / 'prompt.txt').write_text(prompt)
        started = time.perf_counter()
        attempt_id = identifier + ':' + str(time.time_ns())
        try:
            payload = dict(
                model=requested_model,
                store=False,
                input=[dict(role='user', content=[dict(type='input_text', text=prompt)])],
                text=dict(format=dict(type='json_schema', name='result',
                    schema=specification, strict=True)),
            )
            if max_output_tokens is not None:
                if type(max_output_tokens) is not int or max_output_tokens < 1:
                    raise ValueError('LLM输出Token上限必须为正整数')
                payload['max_output_tokens'] = max_output_tokens
            if self.reasoning_effort is not None:
                payload['reasoning'] = dict(effort=self.reasoning_effort)
            (directory / 'request.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
            body = self._post_with_retry(payload, directory, on_start=on_start)
        finally:
            if on_finish is not None:
                on_finish(attempt_id, time.perf_counter() - started)
        (directory / 'provider_response.json').write_text(json.dumps(body, ensure_ascii=False, indent=2) + '\n')
        usage = body.get('usage', {}) or {}
        if on_usage is not None:
            on_usage(identifier + ':' + str(body.get('id', hashlib.sha256(
                json.dumps(body, sort_keys=True).encode()).hexdigest())), usage)
        raw = self._extract_text(body)
        (directory / 'response.json').write_text(raw)
        output = output_type.model_validate_json(raw)
        provenance = {'role': role, 'model_override': self.model,
                      'reasoning_effort': self.reasoning_effort,
                      'provider': 'direct_http_api', 'base_url': self.base_url,
                      'model_identifier': requested_model,
                      'response_model_identifier': body.get('model'),
                      'model_identifier_source': 'explicit_provider_argument',
                      'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                      'usage': dict(input_tokens=usage.get('input_tokens', 0),
                                    output_tokens=usage.get('output_tokens', 0)),
                      'elapsed_seconds': time.perf_counter() - started,
                      'response_sha256': hashlib.sha256((directory / 'response.json').read_bytes()).hexdigest()}
        provenance.update(replayed_verified_response=False,
            invocation_seconds=time.perf_counter() - invocation_started,
            new_llm_seconds=provenance['elapsed_seconds'],
            new_input_tokens=provenance['usage']['input_tokens'],
            new_output_tokens=provenance['usage']['output_tokens'])
        (directory / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        return output, provenance

    def _post_with_retry(self, payload, directory, attempts=4, on_start=None):
        if self.reasoning_effort is not None:
            if on_start is not None:
                on_start(directory.name + ':' + str(time.time_ns()))
            with OpenAI(api_key=self.api_key, base_url=self.base_url + '/v1',
                        timeout=self.timeout_seconds, max_retries=0) as client:
                response = None
                with client.responses.create(**payload, stream=True) as stream:
                    with (directory / 'stream-events.jsonl').open('w') as events:
                        for event in stream:
                            events.write(event.model_dump_json() + '\n')
                            events.flush()
                            if event.type == 'response.completed':
                                response = event.response
                            elif event.type in ('response.failed', 'response.incomplete', 'error'):
                                raise RuntimeError('Responses API返回失败事件: ' + event.model_dump_json())
            if response is None:
                raise RuntimeError('Responses API没有返回完整完成事件')
            if response.status != 'completed':
                raise RuntimeError('Responses API未完成请求: ' + str(response.status))
            return response.model_dump(mode='json')
        last_error = None
        for attempt in range(1, attempts + 1):
            if on_start is not None:
                on_start(directory.name + ':' + str(time.time_ns()))
            try:
                response = httpx.post(
                    self.base_url + '/v1/responses',
                    headers={'Authorization': 'Bearer ' + self.api_key, 'Content-Type': 'application/json'},
                    json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in (429, 500, 502, 503, 504):
                    raise
                last_error = error
                (directory / 'retry.log').open('a').write(
                    f'attempt {attempt}/{attempts}: HTTP {error.response.status_code}\n')
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 20))
            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError,
                    httpx.ReadError, httpx.WriteError, httpx.PoolTimeout) as error:
                last_error = error
                (directory / 'retry.log').open('a').write(
                    f'attempt {attempt}/{attempts} failed: {type(error).__name__}: {error}\n')
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 20))
        raise last_error

    @staticmethod
    def _extract_text(body):
        # Responses API：优先 output_text，其次遍历 output[].content[].text
        if isinstance(body.get('output_text'), str) and body['output_text'].strip():
            return body['output_text']
        parts = [part['text'] for item in body.get('output', []) or []
                 for part in item.get('content', []) or [] if isinstance(part.get('text'), str)]
        if parts:
            return ''.join(parts)
        choices = body.get('choices')
        if choices:
            message = choices[0].get('message', {})
            if isinstance(message.get('content'), str):
                return message['content']
        raise RuntimeError('LLM 响应中没有可解析的文本内容')
