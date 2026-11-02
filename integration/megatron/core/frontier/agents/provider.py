import hashlib
import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import getproxies

from openai.lib._pydantic import to_strict_json_schema


class CodexJSONProvider:
    def __init__(self, directory, model=None, timeout_seconds=600, base_url=None,
                 reasoning_effort='medium'):
        self.directory = Path(directory).resolve()
        self.model = model
        self.base_url = base_url
        if reasoning_effort not in ('low', 'medium', 'high', 'xhigh', 'max', 'ultra'):
            raise ValueError('Agent推理强度不在支持的范围内')
        self.reasoning_effort = reasoning_effort
        if base_url is not None:
            address = urlparse(base_url)
            if address.scheme != 'https' or not address.hostname or address.username or address.password:
                raise ValueError('Agent服务地址必须使用HTTPS且不得包含凭据')
        self.timeout_seconds = timeout_seconds
        self.executable = shutil.which('codex')
        if self.executable is None:
            raise RuntimeError('Codex CLI is required for the configured real LLM provider')
        self.directory.mkdir(parents=True, exist_ok=True)

    def generate(self, role, instructions, observation, output_type, *, on_start=None, on_finish=None,
                 on_usage=None, max_output_tokens=None):
        invocation_started = time.perf_counter()
        specification = to_strict_json_schema(output_type)
        requested_specification = specification
        json_text_transport = len(json.dumps(specification)) > 12000
        if json_text_transport:
            specification = dict(type='object', properties=dict(json_object=dict(type='string')),
                required=['json_object'], additionalProperties=False)
        configuration = tomllib.loads((Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'config.toml').read_text())
        requested_model = self.model or configuration['model']
        prompt = ('Use only the supplied observation and backend query results. '
                  'To request registered backend information, return JSON data in tool_requests '
                  'when that field exists in the output schema. The host executes those requests '
                  'and returns their results. Do not invoke external tools, run shell commands, '
                  'access files, edit source code, or invent measurements. '
                  'Return one object matching the output schema.\n'
                  + instructions + '\n' + json.dumps(observation, ensure_ascii=False, allow_nan=False,
                    separators=(',', ':')))
        if json_text_transport:
            prompt += ('\nReturn the complete decision as JSON encoded in the json_object string. '
                'The decoded decision must match this schema and will be validated by the host:\n'
                + json.dumps(requested_specification, ensure_ascii=False, separators=(',', ':')))
        identity = dict(role=role, prompt=prompt, schema=specification,
            model=requested_model, base_url=self.base_url)
        if self.reasoning_effort != 'medium':
            identity['reasoning_effort'] = self.reasoning_effort
        identifier = hashlib.sha256(json.dumps(identity,
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        directory = self.directory / identifier
        required = ('prompt.txt', 'schema.json', 'response.json', 'provenance.json', 'events.jsonl')
        if directory.exists() and not all((directory / name).is_file() for name in required):
            incomplete = self.directory / 'incomplete-calls'
            incomplete.mkdir(exist_ok=True)
            directory.rename(incomplete / (identifier + '-' + str(time.time_ns())))
        if directory.exists():
            files = [directory / name for name in ('prompt.txt', 'schema.json', 'response.json', 'provenance.json', 'events.jsonl')]
            if not all(path.is_file() for path in files):
                raise RuntimeError('existing proposal has no complete authenticated response record')
            provenance = json.loads((directory / 'provenance.json').read_text())
            events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines() if line.strip()]
            completions = [event for event in events if event['type'] == 'turn.completed']
            if ((directory / 'prompt.txt').read_text() != prompt
                    or json.loads((directory / 'schema.json').read_text()) != specification
                    or provenance['model_override'] != self.model
                    or provenance.get('base_url') != self.base_url
                    or provenance.get('reasoning_effort', 'medium') != self.reasoning_effort
                    or provenance['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest()
                    or provenance['response_sha256'] != hashlib.sha256((directory / 'response.json').read_bytes()).hexdigest()
                    or len(completions) != 1 or completions[0]['usage'] != provenance['usage']
                    or any(event.get('item', {}).get('type') in ('command_execution', 'mcp_tool_call', 'file_change') for event in events)):
                raise RuntimeError('existing actual agent response differs from its registered prompt, schema or provenance')
            raw_response = (directory / 'response.json').read_text()
            output = output_type.model_validate_json(
                json.loads(raw_response)['json_object'] if json_text_transport else raw_response)
            return output, provenance | dict(replayed_verified_response=True,
                invocation_seconds=time.perf_counter() - invocation_started,
                new_llm_seconds=0.0, new_input_tokens=0, new_output_tokens=0)
        directory.mkdir(exist_ok=False)
        schema = directory / 'schema.json'
        schema.write_text(json.dumps(specification, indent=2) + '\n')
        (directory / 'prompt.txt').write_text(prompt)
        command = [self.executable, 'exec', '--ephemeral',
                   '--skip-git-repo-check', '--sandbox', 'read-only',
                   '-c', f'model_reasoning_effort="{self.reasoning_effort}"',
                   '--json', '--color', 'never', '--output-schema', str(schema),
                   '--output-last-message', str(directory / 'response.json')]
        if self.base_url is None:
            command.extend(('-c', 'model_provider="parameld_chatgpt"',
                '-c', 'model_providers.parameld_chatgpt={name="OpenAI",wire_api="responses",requires_openai_auth=true,supports_websockets=false}'))
        command.extend(('--model', requested_model))
        command.append('-')
        environment = os.environ.copy()
        environment.pop('OPENAI_API_KEY', None)
        system_proxies = getproxies()
        for scheme, variable in (('http', 'HTTP_PROXY'), ('https', 'HTTPS_PROXY'), ('all', 'ALL_PROXY')):
            if scheme in system_proxies:
                environment.setdefault(variable, system_proxies[scheme])
        if self.base_url is not None:
            environment['OPENAI_API_KEY'] = configuration['model_providers']['custom']['experimental_bearer_token']
        environment['TMPDIR'] = str(directory)
        started = time.perf_counter()
        attempt_id = identifier + ':' + str(time.time_ns())
        if on_start is not None:
            on_start(attempt_id)
        with (directory / 'events.jsonl').open('w') as events_file, (directory / 'provider.log').open('w') as log_file:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events_file,
                stderr=log_file, text=True, cwd=directory, env=environment)
            try:
                process.communicate(input=prompt, timeout=self.timeout_seconds)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if on_finish is not None:
                    on_finish(attempt_id, time.perf_counter() - started)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines() if line.strip()]
        if any(event.get('item', {}).get('type') in ('command_execution', 'mcp_tool_call', 'file_change')
               for event in events):
            raise RuntimeError('proposal generation invoked an unauthorized tool')
        completions = [event for event in events if event['type'] == 'turn.completed']
        if len(completions) != 1:
            raise RuntimeError('LLM provider did not complete exactly one response')
        if on_usage is not None:
            on_usage(attempt_id, completions[0]['usage'])
        raw_response = (directory / 'response.json').read_text()
        provenance = {'role': role, 'model_override': self.model, 'provider': 'authenticated_codex_cli',
                      'base_url': self.base_url,
                      'model_identifier': requested_model, 'model_identifier_source': 'explicit_codex_cli_argument',
                      'reasoning_effort': self.reasoning_effort,
                      'response_transport': 'json_text' if json_text_transport else 'direct_schema',
                      'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                      'usage': completions[0]['usage'], 'elapsed_seconds': time.perf_counter() - started,
                      'response_sha256': hashlib.sha256((directory / 'response.json').read_bytes()).hexdigest()}
        provenance.update(replayed_verified_response=False,
            invocation_seconds=time.perf_counter() - invocation_started,
            new_llm_seconds=provenance['elapsed_seconds'],
            new_input_tokens=provenance['usage']['input_tokens'],
            new_output_tokens=provenance['usage']['output_tokens'])
        (directory / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        output = output_type.model_validate_json(
            json.loads(raw_response)['json_object'] if json_text_transport else raw_response)
        return output, provenance
