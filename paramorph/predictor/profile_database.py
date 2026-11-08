import hashlib
import json
import math
import sqlite3
import statistics
from dataclasses import asdict
from pathlib import Path
from core_data_models import MemoryPolicy
from ddp_overlap import DDP_OVERLAP_FIELDS, resolve_ddp_overlap

from .measured_modules import annotate_measurement_semantics


EXECUTION_FIELDS = ('model', 'checkpoint', 'data', 'tp', 'cp', 'pp', 'dp', 'ep', 'vpp',
    'group', 'sequence', 'micro_batch', 'global_batch', 'offload',
    'optimizer_offload_fraction', 'optimizer_transfer_overlap', 'recompute',
    'node_count', 'gpus_per_node', 'cross_entropy_impl', 'memory_policy',
    'bucket_elements', 'overlap_p2p', 'cuda_device_max_connections')
EXECUTION_FIELDS += ('etp', 'expert_dp', 'rank_mapping', 'pipeline_schedule', 'optimizer_sharding')
EXECUTION_FIELDS += ('precision', 'optimizer', 'optimizer_config', 'attention_backend',
    'gradient_path', 'fused_cross_entropy', 'mlp_recompute_every', 'communication_executor_protocol',
    'hardware_fingerprint', 'backend_fingerprint', 'random_seed', 'loss_normalization',
    'precision_aware_optimizer')
EXECUTION_FIELDS += ('execution_backend', 'fsdp_sharding_strategy', 'dense_sharding_strategy',
    'expert_sharding_strategy', 'dp_inner_size', 'dp_outer_size', 'outer_sharding_strategy',
    'expert_dp_inner_size', 'expert_dp_outer_size', 'expert_outer_sharding_strategy',
    'megatron_fsdp_version', 'fsdp_unit', 'fsdp_prefetch',
    'fsdp_prefetch_distance', 'fsdp_reshard_after_forward', 'fsdp_double_buffer',
    'fsdp_bucket_elements', 'checkpoint_format', 'checkpoint_reshard', 'optimizer_checkpoint_mode')
EXECUTION_FIELDS += ('pipeline_dtype', 'sequence_parallel', 'clip_grad')
EXECUTION_FIELDS += ('main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype')
EXECUTION_FIELDS += DDP_OVERLAP_FIELDS
PROFILE_SIGNATURE_REVISION = 'native_backend_effective_execution_v6'


def resolved_cuda_connections(request):
    value = request.get('cuda_device_max_connections',
        2 if request.get('execution_backend', 'megatron_ddp') == 'megatron_fsdp' else 1)
    if type(value) is not int or value < 1:
        raise ValueError('CUDA连接数量必须为正整数')
    return value


def resolved_clip_grad(request):
    value = request.get('clip_grad', 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('clip_grad需要有限非负数值')
    return float(value)


def resolved_bucket_elements(request):
    value = request.get('bucket_elements')
    if value is None:
        value = max(40000000, 1000000 * request['dp'])
    if type(value) is not int or value <= 0:
        raise ValueError('梯度Bucket元素数量必须为正整数')
    return value


def normalized_execution(request):
    resolved = {key: request.get(key) for key in EXECUTION_FIELDS}
    if 'fp16_bounded_adam_sha256' in request:
        resolved['fp16_bounded_adam_sha256'] = request['fp16_bounded_adam_sha256']
    if 'optimizer_schedule' in request:
        resolved['optimizer_schedule'] = request['optimizer_schedule']
        resolved['optimizer_hyperparameters'] = request['optimizer_hyperparameters']
    if request.get('native_uniform_pipeline_layout', False):
        resolved['native_uniform_pipeline_layout'] = True
    if 'training_data_contract' in request:
        resolved['training_data_contract'] = request['training_data_contract']
    if request.get('record_native_schedule'):
        overlap = request.get('overlap_p2p', False)
        resolved['batch_p2p_comm'] = request.get('batch_p2p_comm', not overlap)
        resolved['overlap_p2p_comm_warmup_flush'] = request.get('overlap_p2p_comm_warmup_flush', False)
    if request.get('gpus_per_worker', request['gpus_per_node']) != request['gpus_per_node']:
        resolved['gpus_per_worker'] = request['gpus_per_worker']
    if request.get('memory_policy') is not None:
        resolved['memory_policy'] = asdict(MemoryPolicy(**request['memory_policy']))
    resolved['optimizer'] = request.get('optimizer', 'adam')
    resolved['clip_grad'] = resolved_clip_grad(request)
    resolved['bucket_elements'] = resolved_bucket_elements(request)
    resolved['cuda_device_max_connections'] = resolved_cuda_connections(request)
    world = request['node_count'] * request['gpus_per_node']
    resolved['etp'] = request.get('etp', request['tp'])
    resolved['expert_dp'] = request.get('expert_dp', world // (resolved['etp'] * request['ep'] * request['pp']))
    resolved['rank_mapping'] = request.get('rank_mapping', list(range(world)))
    resolved['pipeline_schedule'] = request.get('pipeline_schedule', 'interleaved_1f1b' if request['vpp'] > 1 else '1f1b')
    resolved['optimizer_sharding'] = request.get('optimizer_sharding', 'distributed')
    resolved['optimizer_offload_fraction'] = float(request.get('optimizer_offload_fraction', 0.0))
    if resolved['optimizer_offload_fraction'] > 0 and request.get('optimizer_pin_cpu_params') is False:
        resolved['optimizer_pin_cpu_params'] = False
    if resolved['optimizer_offload_fraction'] > 0 and request.get('optimizer_cpu_copy_mode') is not None:
        resolved['optimizer_cpu_copy_mode'] = request['optimizer_cpu_copy_mode']
    if request.get('pinned_host_allocator') is not None:
        resolved['pinned_host_allocator'] = request['pinned_host_allocator']
        resolved['pinned_host_register_threads'] = request['pinned_host_register_threads']
    resolved['optimizer_transfer_overlap'] = bool(request.get('optimizer_transfer_overlap', False))
    resolved['precision'] = request.get('precision', 'bfloat16').removeprefix('torch.')
    resolved['precision_aware_optimizer'] = bool(request.get('precision_aware_optimizer', False))
    resolved['pipeline_dtype'] = request.get('pipeline_dtype', resolved['precision']).removeprefix('torch.')
    resolved['sequence_parallel'] = request.get('sequence_parallel', request['tp'] > 1)
    backend = request.get('execution_backend', 'megatron_ddp')
    resolved['execution_backend'] = backend
    resolved.update(resolve_ddp_overlap(request))
    for name in ('main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype'):
        value = request.get(name, 'float32' if backend == 'megatron_ddp' else None)
        resolved[name] = value.removeprefix('torch.') if isinstance(value, str) else value
    resolved['dense_sharding_strategy'] = request.get('dense_sharding_strategy',
        request.get('fsdp_sharding_strategy', 'no_shard')) if backend == 'megatron_fsdp' else 'no_shard'
    resolved['expert_sharding_strategy'] = request.get('expert_sharding_strategy',
        request.get('fsdp_sharding_strategy', 'no_shard')) if backend == 'megatron_fsdp' else 'no_shard'
    resolved['fsdp_sharding_strategy'] = resolved['dense_sharding_strategy']
    resolved['megatron_fsdp_version'] = request.get('megatron_fsdp_version', 1) if backend == 'megatron_fsdp' else 0
    resolved['dp_inner_size'] = request.get('dp_inner_size', request['dp'] * request['cp'])
    resolved['dp_outer_size'] = request.get('dp_outer_size', 1)
    resolved['outer_sharding_strategy'] = request.get('outer_sharding_strategy', 'no_shard')
    resolved['expert_dp_inner_size'] = request.get('expert_dp_inner_size', resolved['expert_dp'])
    resolved['expert_dp_outer_size'] = request.get('expert_dp_outer_size', 1)
    resolved['expert_outer_sharding_strategy'] = request.get('expert_outer_sharding_strategy', 'no_shard')
    resolved['checkpoint_format'] = request.get('checkpoint_format', 'torch_dist')
    resolved['checkpoint_reshard'] = request.get('checkpoint_reshard', True)
    resolved['optimizer_checkpoint_mode'] = request.get('optimizer_checkpoint_mode', 'disabled')
    if backend != 'megatron_fsdp':
        for name in ('fsdp_unit', 'fsdp_prefetch', 'fsdp_prefetch_distance',
                     'fsdp_reshard_after_forward', 'fsdp_double_buffer', 'fsdp_bucket_elements'):
            resolved[name] = None
    else:
        for name in ('megatron_fsdp_main_params_dtype', 'megatron_fsdp_main_grads_dtype'):
            dtype = request.get(name)
            resolved[name] = dtype.removeprefix('torch.') if isinstance(dtype, str) else dtype
    return resolved


def effective_execution_config(request):
    from scheduling.native_microbatches import native_schedule_settings

    effective = normalized_execution(request)
    schedule = native_schedule_settings(request)
    for key in ('group', 'overlap_p2p', 'batch_p2p_comm', 'overlap_p2p_comm_warmup_flush'):
        effective[key] = schedule[key]
    effective['recompute_num_layers'] = request.get('recompute_num_layers') if request['recompute'] == 'full' else None
    return effective


def signature(request):
    return hashlib.sha256(json.dumps(effective_execution_config(request),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def plan_signature(request):
    return hashlib.sha256(json.dumps(dict(execution_signature=signature(request),
        decoder_chunks=request['decoder_chunks']), sort_keys=True).encode()).hexdigest()


def same_training_plan(first, second):
    left, right = effective_execution_config(first), effective_execution_config(second)
    # 发现阶段的身份字段独立核验，不能使同一训练配置绕过测量排除。
    for field in ('backend_fingerprint', 'hardware_fingerprint'):
        left.pop(field)
        right.pop(field)
    return left == right and first['decoder_chunks'] == second['decoder_chunks']


def observed_ddp_overlap(directory, world):
    paths = [Path(directory) / f'rank{rank}.manifest.json' for rank in range(world)]
    if not all(path.is_file() for path in paths):
        return {}
    manifests = [json.loads(path.read_text()) for path in paths]
    values = {}
    for name in DDP_OVERLAP_FIELDS:
        observed = [row.get(name) for row in manifests]
        if all(value is None for value in observed):
            continue
        if any(type(value) is not bool or value is not observed[0] for value in observed):
            raise ValueError(f'逐 Rank DDP 配置证据不一致：{name}')
        values[name] = observed[0]
    return values


def profile_execution_request(document, source=None):
    request = document['request']
    bound = document.get('runtime_ddp_overlap')
    if bound is None:
        for directory in dict.fromkeys((source, document.get('source_directory'))):
            if not directory:
                continue
            bound = observed_ddp_overlap(Path(directory) / 'artifacts',
                request['node_count'] * request['gpus_per_node'])
            if bound:
                break
    return dict(request, **(bound or {}))


def backend_reuse_evidence(request, profile):
    target = request.get('backend_fingerprint')
    hashes = profile.get('source_hashes')
    if not hashes:
        return dict(status='UNVERIFIED', target_fingerprint=target,
            source_fingerprint=None, reason='实测记录没有后端源码校验值')
    source = hashlib.sha256(json.dumps(hashes, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if target is None:
        return dict(status='UNVERIFIED', target_fingerprint=None,
            source_fingerprint=source, reason='当前请求没有backend_fingerprint')
    return dict(status='MATCHED' if target == source else 'MISMATCHED',
        target_fingerprint=target, source_fingerprint=source,
        reason=None if target == source else '当前后端身份与实测源码校验值不同')


class ProfileDatabase:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute('CREATE TABLE IF NOT EXISTS profiles '
            '(id TEXT PRIMARY KEY, signature TEXT NOT NULL, document TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS predictions '
            '(id TEXT PRIMARY KEY, document TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS validations '
            '(id TEXT PRIMARY KEY, signature TEXT NOT NULL, document TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS profile_sources '
            '(profile_id TEXT PRIMARY KEY, directory TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS communication_profiles '
            '(id TEXT PRIMARY KEY, document TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS overlap_profiles '
            '(id TEXT PRIMARY KEY, document TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS operator_costs '
            '(id TEXT PRIMARY KEY, cost_key TEXT NOT NULL, document TEXT NOT NULL)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS operator_cost_index ON operator_costs(cost_key)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS operator_families '
            '(id TEXT PRIMARY KEY REFERENCES operator_costs(id), family_key TEXT NOT NULL)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS operator_family_index ON operator_families(family_key)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS operator_models '
            '(id TEXT PRIMARY KEY, family_key TEXT NOT NULL, document TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS database_metadata '
            '(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        family_revision = self.connection.execute(
            'SELECT value FROM database_metadata WHERE key=?', ('compute_family_revision',)).fetchone()
        if family_revision is None or family_revision[0] != 'shape_conditioned_v2':
            from .compute_cost import compute_family
            rows = self.connection.execute('SELECT id, document FROM operator_costs').fetchall()
            self.connection.execute('DELETE FROM operator_families')
            self.connection.executemany('INSERT INTO operator_families VALUES (?, ?)',
                [(identifier, compute_family(json.loads(encoded)['identity']))
                    for identifier, encoded in rows])
            self.connection.execute('INSERT OR REPLACE INTO database_metadata VALUES (?, ?)',
                ('compute_family_revision', 'shape_conditioned_v2'))
        signature_revision = self.connection.execute(
            'SELECT value FROM database_metadata WHERE key=?', ('profile_signature_revision',)).fetchone()
        if signature_revision is None or signature_revision[0] != PROFILE_SIGNATURE_REVISION:
            for identifier, encoded, source in self.connection.execute('SELECT profiles.id, document, directory '
                    'FROM profiles LEFT JOIN profile_sources ON profiles.id=profile_sources.profile_id'):
                self.connection.execute('UPDATE profiles SET signature=? WHERE id=?',
                    (signature(profile_execution_request(json.loads(encoded), source)), identifier))
            self.connection.execute('INSERT OR REPLACE INTO database_metadata VALUES (?, ?)',
                ('profile_signature_revision', PROFILE_SIGNATURE_REVISION))
        self.connection.commit()

    def add(self, document):
        document = annotate_measurement_semantics(document)
        encoded = json.dumps(document, sort_keys=True, allow_nan=False)
        existing = self.connection.execute('SELECT document FROM profiles WHERE id=?',
            (document['id'],)).fetchone()
        if existing is not None:
            previous = annotate_measurement_semantics(json.loads(existing[0]))
            comparable = json.loads(encoded)
            if 'runtime_ddp_overlap' not in previous:
                comparable.pop('runtime_ddp_overlap', None)
            if json.loads(json.dumps(previous, allow_nan=False)) != comparable:
                raise ValueError('同一测量身份对应不同证据')
        self.connection.execute('INSERT OR IGNORE INTO profiles VALUES (?, ?, ?)',
            (document['id'], signature(profile_execution_request(document)), encoded))
        self.connection.commit()

    def add_communication(self, identifier, benchmark):
        if benchmark['source'] not in (
                'actual_nccl_cuda_event_microbenchmark',
                'actual_pipeline_p2p_cuda_event_calibration'):
            raise ValueError('通信成本需要真实NCCL测量来源')
        encoded = json.dumps(benchmark, sort_keys=True, allow_nan=False)
        existing = self.connection.execute('SELECT document FROM communication_profiles WHERE id=?', (identifier,)).fetchone()
        if existing and existing[0] != encoded:
            raise ValueError('同名NCCL成本记录对应不同实际测量')
        self.connection.execute('INSERT OR IGNORE INTO communication_profiles VALUES (?, ?)', (identifier, encoded))
        self.connection.commit()

    def communication_profiles(self):
        generation = (self.connection.total_changes,
            self.connection.execute('PRAGMA data_version').fetchone()[0])
        cached = getattr(self, '_communication_profile_cache', None)
        if cached is None or cached[0] != generation:
            records = [dict(id=identifier, benchmark=json.loads(encoded)) for identifier, encoded
                in self.connection.execute('SELECT id, document FROM communication_profiles ORDER BY id')]
            self._communication_profile_cache = (generation, records)
        return self._communication_profile_cache[1]

    def measured_profiles(self):
        generation = (self.connection.total_changes,
            self.connection.execute('PRAGMA data_version').fetchone()[0])
        cached = getattr(self, '_measured_profile_cache', None)
        if cached is None or cached[0] != generation:
            records = []
            for identifier, encoded, directory in self.connection.execute('SELECT profiles.id, document, directory '
                    'FROM profiles LEFT JOIN profile_sources ON profiles.id=profile_sources.profile_id ORDER BY profiles.id'):
                profile = json.loads(encoded)
                profile['request'] = self.profile_request(profile, directory)
                records.append((identifier, profile))
            self._measured_profile_cache = (generation, records)
        return [(identifier, dict(profile)) for identifier, profile in self._measured_profile_cache[1]]

    def add_overlap(self, identifier, benchmark):
        if benchmark['source'] != 'real_gpu_compute_communication_microbenchmark':
            raise ValueError('计算通信重叠成本需要真实GPU并发测量来源')
        if benchmark['gamma'] < 1 or not benchmark['samples']:
            raise ValueError('计算通信重叠测量需要有效减速系数与全部样本')
        encoded = json.dumps(benchmark, sort_keys=True, allow_nan=False)
        existing = self.connection.execute('SELECT document FROM overlap_profiles WHERE id=?',
            (identifier,)).fetchone()
        if existing and existing[0] != encoded:
            raise ValueError('同名计算通信重叠记录对应不同实际测量')
        self.connection.execute('INSERT OR IGNORE INTO overlap_profiles VALUES (?, ?)',
            (identifier, encoded))
        self.connection.commit()

    def overlap_profiles(self):
        return [dict(id=identifier, benchmark=json.loads(encoded)) for identifier, encoded
            in self.connection.execute('SELECT id, document FROM overlap_profiles ORDER BY id')]

    def add_operator_cost(self, document, *, commit=True):
        from .compute_cost import compute_family
        key = hashlib.sha256(json.dumps(document['identity'], sort_keys=True).encode()).hexdigest()
        encoded = json.dumps(document, sort_keys=True, allow_nan=False)
        existing = self.connection.execute('SELECT document FROM operator_costs WHERE id=?',
            (document['evidence_id'],)).fetchone()
        if existing and existing[0] != encoded:
            raise ValueError('同一算子成本证据对应不同实际测量')
        self.connection.execute('INSERT OR IGNORE INTO operator_costs VALUES (?, ?, ?)',
            (document['evidence_id'], key, encoded))
        self.connection.execute('INSERT OR IGNORE INTO operator_families VALUES (?, ?)',
            (document['evidence_id'], compute_family(document['identity'])))
        if commit:
            self.connection.commit()

    def operator_costs(self, identity):
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return [json.loads(row[0]) for row in self.connection.execute(
            'SELECT document FROM operator_costs WHERE cost_key=? ORDER BY id', (key,))]

    def operator_family_costs(self, family_key):
        from .compute_cost import compute_family
        missing = self.connection.execute('SELECT id, document FROM operator_costs WHERE id NOT IN '
            '(SELECT id FROM operator_families)').fetchall()
        self.connection.executemany('INSERT INTO operator_families VALUES (?, ?)',
            [(identifier, compute_family(json.loads(encoded)['identity'])) for identifier, encoded in missing])
        self.connection.commit()
        return [json.loads(row[0]) for row in self.connection.execute(
            'SELECT operator_costs.document FROM operator_costs JOIN operator_families USING(id) '
            'WHERE family_key=? ORDER BY id', (family_key,))]

    def operator_family_aggregates(self, family_key):
        grouped = {}
        cursor = self.connection.execute(
            'SELECT operator_costs.document FROM operator_costs JOIN operator_families USING(id) '
            'WHERE family_key=? ORDER BY id', (family_key,))
        for encoded, in cursor:
            row = json.loads(encoded)
            evidence_id = row['evidence_id']
            run_id = evidence_id.split(':', 1)[0]
            identity_key = hashlib.sha256(json.dumps(row['identity'],
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            key = (run_id, identity_key)
            if key not in grouped:
                grouped[key] = dict(identity=row['identity'], milliseconds=[], evidence_ids=[])
            grouped[key]['milliseconds'].append(row['milliseconds'])
            grouped[key]['evidence_ids'].append(evidence_id)
        return [dict(identity=row['identity'], milliseconds=statistics.median(row['milliseconds']),
            evidence_id=run_id + ':aggregate:' + identity_key,
            evidence_ids=row['evidence_ids'], sample_count=len(row['milliseconds']))
            for (run_id, identity_key), row in grouped.items()]

    def add_operator_model(self, document):
        required = {'model_id', 'family_key', 'model_kind', 'kernel_tile', 'concurrent_tiles',
            'launch_ms', 'wave_k_step_ms', 'shape_bounds', 'evidence_refs', 'kernel_configuration_source'}
        if set(document) != required or document['model_kind'] != 'gemm_tile':
            raise ValueError('Kernel模型需要完整Tile配置、形状范围和真实标定来源')
        if any(type(document['kernel_tile'][name]) is not int or document['kernel_tile'][name] < 1 for name in ('B_M', 'B_N', 'B_K')):
            raise ValueError('Kernel Tile尺寸必须为正整数')
        if type(document['concurrent_tiles']) is not int or document['concurrent_tiles'] < 1:
            raise ValueError('Kernel并发Tile数量必须为正整数')
        if not document['kernel_configuration_source'] or not document['evidence_refs']:
            raise ValueError('Tile模型需要Kernel配置来源和真实算子测量身份')
        if any(not math.isfinite(document[name]) or document[name] < 0 for name in ('launch_ms', 'wave_k_step_ms')) or document['wave_k_step_ms'] == 0:
            raise ValueError('Tile模型需要有限且合法的标定时延参数')
        if set(document['shape_bounds']) != {'M', 'N', 'K'} or any(
            len(bounds) != 2 or any(type(value) is not int or value < 1 for value in bounds) or bounds[0] > bounds[1]
            for bounds in document['shape_bounds'].values()):
            raise ValueError('Tile模型需要明确的M、N、K形状范围')
        known = {row[0] for row in self.connection.execute('SELECT id FROM operator_costs')}
        if not set(document['evidence_refs']) <= known:
            raise ValueError('Tile模型引用了不存在的真实算子测量')
        encoded = json.dumps(document, sort_keys=True, allow_nan=False)
        previous = self.connection.execute('SELECT document FROM operator_models WHERE id=?', (document['model_id'],)).fetchone()
        if previous and previous[0] != encoded:
            raise ValueError('同一Kernel标定模型身份对应不同内容')
        self.connection.execute('INSERT OR IGNORE INTO operator_models VALUES (?, ?, ?)',
            (document['model_id'], document['family_key'], encoded))
        self.connection.commit()

    def operator_models(self, family_key):
        return [json.loads(row[0]) for row in self.connection.execute(
            'SELECT document FROM operator_models WHERE family_key=? ORDER BY id', (family_key,))]

    def get(self, profile_id, request, *, blind=False):
        row = self.connection.execute('SELECT signature, document FROM profiles WHERE id=?',
            (profile_id,)).fetchone()
        if row is None or row[0] != signature(request):
            raise ValueError('没有与模型、工作负载、拓扑及执行配置匹配的成本记录')
        document = json.loads(row[1])
        source = self.connection.execute('SELECT directory FROM profile_sources WHERE profile_id=?',
            (profile_id,)).fetchone()
        if source and Path(source[0]).is_dir():
            document['source_directory'] = source[0]
        document['request'] = self.profile_request(document)
        if blind and same_training_plan(document['request'], request):
            raise ValueError('盲测候选不能使用自身的成本、Anchor或显存校准记录')
        return document

    def profile_request(self, document, source=None):
        cache = self.__dict__.setdefault('_profile_runtime_request_cache', {})
        key = (document['id'], str(source or document.get('source_directory') or ''))
        if key not in cache:
            cache[key] = profile_execution_request(document, source)
        return dict(cache[key])

    def register_source(self, profile_id, directory):
        row = self.connection.execute('SELECT document FROM profiles WHERE id=?', (profile_id,)).fetchone()
        if row is None:
            raise ValueError('实际Trace目录缺少已登记成本身份')
        document = json.loads(row[0])
        directory = Path(directory).resolve()
        artifacts = directory / 'artifacts'
        for name, digest in document['evidence_files'].items():
            path = artifacts / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Trace目录文件与实际成本证据不一致')
        existing = self.connection.execute('SELECT directory FROM profile_sources WHERE profile_id=?',
            (profile_id,)).fetchone()
        if existing and existing[0] != str(directory):
            raise ValueError('成本证据的实际Trace来源目录发生变化')
        self.connection.execute('INSERT OR IGNORE INTO profile_sources VALUES (?, ?)', (profile_id, str(directory)))
        self.connection.execute('UPDATE profiles SET signature=? WHERE id=?',
            (signature(self.profile_request(document, directory)), profile_id))
        self.connection.commit()

    def matches(self, profile_id, request):
        row = self.connection.execute('SELECT signature FROM profiles WHERE id=?', (profile_id,)).fetchone()
        if row is None:
            raise ValueError('指定成本记录身份不存在')
        return row[0] == signature(request)

    def select(self, request, *, blind=False):
        rows = self.connection.execute('SELECT id, document FROM profiles WHERE signature=? ORDER BY id',
            (signature(request),)).fetchall()
        if not rows:
            return None
        documents = [(identifier, json.loads(document)) for identifier, document in rows]
        if blind:
            documents = [(identifier, document) for identifier, document in documents
                if not same_training_plan(self.profile_request(document), request)]
        if not documents:
            return None
        return max(documents, key=lambda item: (
            item[1]['request']['decoder_chunks'] == request['decoder_chunks'], item[1]['anchor'] is not None,
            sum(len(windows) for windows in item[1]['measured_windows'].values())))[0]

    def reusable_compute_profiles(self, request, *, blind=False):
        from cost.decomposition import COMPUTE_SHAPE_FIELDS
        identity_fields = ('model', 'checkpoint', 'precision', 'attention_backend',
            'cross_entropy_impl', 'optimizer_sharding', 'execution_backend',
            'dense_sharding_strategy', 'expert_sharding_strategy', 'sequence_parallel',
            'pipeline_dtype', 'clip_grad', *DDP_OVERLAP_FIELDS)
        output = []
        target = normalized_execution(request)
        for identifier, encoded in self.connection.execute('SELECT id, document FROM profiles ORDER BY id'):
            document = json.loads(encoded)
            measured_request = self.profile_request(document)
            previous = normalized_execution(measured_request)
            if blind and same_training_plan(measured_request, request):
                continue
            if any(previous.get(key) != target.get(key) for key in identity_fields):
                continue
            if any(previous.get(key) != target.get(key) for key in COMPUTE_SHAPE_FIELDS):
                continue
            backend_evidence = backend_reuse_evidence(request, document)
            if backend_evidence['status'] == 'MISMATCHED':
                continue
            semantics = annotate_measurement_semantics(document)['measurement_semantics']
            output.append(dict(profile_id=identifier, source_hashes=document['source_hashes'],
                hardware=document['hardware'], layers=document['layers'],
                backend_identity_evidence=backend_evidence,
                measurement_semantics=semantics,
                reuse_scope='measured_layer_composite_compute_shapes',
                requires_schedule_and_communication_revalidation=True))
        return output

    def save_prediction(self, prediction):
        self.connection.execute('INSERT OR REPLACE INTO predictions VALUES (?, ?)',
            (prediction['prediction_id'], json.dumps(prediction, allow_nan=False)))
        self.connection.commit()

    def cost_database_revision(self):
        digest = hashlib.sha256()
        identity_columns = {
            'profiles': 'id, signature',
            'communication_profiles': 'id',
            'overlap_profiles': 'id',
            'operator_costs': 'id, cost_key',
            'operator_families': 'id, family_key',
            'operator_models': 'id, family_key',
            'validations': 'id, signature',
            'database_metadata': 'key, value',
        }
        for table, columns in identity_columns.items():
            digest.update(table.encode())
            for row in self.connection.execute(
                    f'SELECT {columns} FROM {table} ORDER BY 1'):
                digest.update(json.dumps(row, sort_keys=True).encode())
        return digest.hexdigest()

    def cached_dag_projection(self, cache_identity):
        rows = [json.loads(row[0]) for row in self.connection.execute(
            'SELECT document FROM predictions ORDER BY rowid DESC')]
        return next((row for row in rows if row.get('projection_cache_identity') == cache_identity), None)

    def record_validation(self, request, prediction, measurement):
        if prediction['plan_signature'] != plan_signature(request) or measurement['run_id'] != request['run_id']:
            raise ValueError('预测误差校准需要同一候选的真实结果')
        if measurement['status'] != 'PASS' or not measurement['performance_eligible']:
            raise ValueError('预测误差校准需要正式成功测量')
        record = dict(prediction_id=prediction['prediction_id'], measured_run_id=measurement['run_id'],
            execution_request=request,
            plan_signature=plan_signature(request),
            time_residual_ms=measurement['step_seconds'] * 1000 - prediction['performance']['step_time_ms'],
            memory_residual_bytes=measurement['peak_allocated_gib'] * 2**30 -
                max(prediction['memory']['peak_per_rank_bytes'].values()),
            evidence_directory=measurement['raw_artifacts'])
        existing = self.connection.execute('SELECT document FROM validations WHERE id=?',
            (measurement['run_id'],)).fetchone()
        encoded = json.dumps(record, allow_nan=False)
        if existing and existing[0] != encoded:
            raise ValueError('同一真实测量身份的预测验证记录发生变化')
        self.connection.execute('INSERT OR IGNORE INTO validations VALUES (?, ?, ?)',
            (measurement['run_id'], signature(request), json.dumps(record, allow_nan=False)))
        self.connection.commit()
        return record

    def validation_residuals(self, request):
        records = [json.loads(row[0]) for row in self.connection.execute(
            'SELECT document FROM validations WHERE signature=? AND id != ?',
            (signature(request), request['run_id'])).fetchall()]
        target = plan_signature(request)
        independent = []
        for record in records:
            if 'execution_request' not in record:
                continue
            identity = plan_signature(record['execution_request'])
            if identity is None:
                row = self.connection.execute('SELECT document FROM predictions WHERE id=?',
                    (record['prediction_id'],)).fetchone()
                if row is None:
                    continue
                identity = json.loads(row[0])['plan_signature']
            if identity != target:
                independent.append(record)
        return independent
