from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import time

from contracts import fingerprint


class ExperimentStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.cost_clock = None
        self.connection.execute('PRAGMA foreign_keys=ON')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS trials (
                id TEXT PRIMARY KEY, plan_hash TEXT UNIQUE NOT NULL, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS charges (
                id TEXT PRIMARY KEY, resource TEXT NOT NULL, amount REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS scratchpad (
                id TEXT PRIMARY KEY, trial_id TEXT NOT NULL REFERENCES trials(id),
                kind TEXT NOT NULL, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS profiling_requests (
                id TEXT PRIMARY KEY, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS search_rounds (
                id INTEGER PRIMARY KEY, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS agent_sessions (
                id TEXT PRIMARY KEY, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS agent_context_metrics (
                id TEXT PRIMARY KEY, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS search_sessions (
                id TEXT PRIMARY KEY, elapsed_seconds REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS search_state_repository (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS actual_cost_checkpoints (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL, identifier TEXT NOT NULL, document TEXT NOT NULL);
        ''')
        self.connection.commit()

    def add_search_state(self, kind, document):
        if kind not in ('INITIAL_CONFIGURATION', 'INITIAL_CONFIGURATION_RESULT',
                        'BASELINE_REUSE_DECISION', 'BASELINE_RUNTIME_BINDING', 'CHECKPOINT_VALIDATION_IN_TRAINING',
                        'AGENT_RESPONSE', 'PLANNING_INVESTIGATION', 'PLANNING_FEEDBACK',
                        'PLANNING_PORTFOLIO_QUEUE', 'PLANNING_PORTFOLIO_DISPATCH', 'PLANNING_EVALUATION',
                        'PLANNING_EXECUTION_CONTEXT', 'COST_MEASUREMENT_AVAILABLE_TO_AGENT',
                        'COST_REQUEST_DEFERRED_BEFORE_GPU_START'):
            raise ValueError('Search State Repository记录类型未注册')
        encoded = json.dumps(document, sort_keys=True, allow_nan=False)
        identifier = fingerprint(dict(kind=kind, document=document))
        previous = self.connection.execute(
            'SELECT kind, document FROM search_state_repository WHERE id=?', (identifier,)).fetchone()
        if previous is not None and previous != (kind, encoded):
            raise ValueError('Search State Repository记录身份对应不同内容')
        self.connection.execute(
            'INSERT OR IGNORE INTO search_state_repository (id, kind, document) VALUES (?, ?, ?)',
            (identifier, kind, encoded))
        self.connection.commit()
        return identifier

    def search_states(self, kind=None):
        if kind is None:
            rows = self.connection.execute(
                'SELECT sequence, id, kind, document FROM search_state_repository ORDER BY sequence')
        else:
            rows = self.connection.execute(
                'SELECT sequence, id, kind, document FROM search_state_repository '
                'WHERE kind=? ORDER BY sequence', (kind,))
        return [dict(sequence=row[0], state_id=row[1], kind=row[2],
            document=json.loads(row[3])) for row in rows]

    def put_trial(self, plan, hypothesis=None, delta=None, rewrite=None):
        by_identifier = self.connection.execute('SELECT plan_hash, document FROM trials WHERE id=?',
            (plan.plan_id,)).fetchone()
        if by_identifier is not None:
            from contracts import TrainingPlan
            document = json.loads(by_identifier[1])
            recalculated = TrainingPlan.model_validate(document['plan']).plan_hash
            if recalculated != plan.plan_hash:
                raise ValueError('同一Trial ID对应不同训练语义')
            if by_identifier[0] != recalculated:
                document['plan_hash'] = recalculated
                self.connection.execute('UPDATE trials SET plan_hash=?, document=? WHERE id=?',
                    (recalculated, json.dumps(document, allow_nan=False), plan.plan_id))
                self.connection.commit()
            return document, False
        existing = self.connection.execute('SELECT document FROM trials WHERE plan_hash=?', (plan.plan_hash,)).fetchone()
        if existing:
            return json.loads(existing[0]), False
        document = dict(trial_id=plan.plan_id, parent_trial_id=plan.parent_plan_id,
            plan=plan.model_dump(mode='json'), plan_hash=plan.plan_hash, hypothesis=hypothesis,
            graph_delta=delta, validation=None, prediction=None, measurement=None,
            rewrite=rewrite,
            state='PENDING', created_at=datetime.now(timezone.utc).isoformat())
        self.connection.execute('INSERT INTO trials VALUES (?, ?, ?)',
            (plan.plan_id, plan.plan_hash, json.dumps(document, allow_nan=False)))
        self.connection.commit()
        return document, True

    def update_trial(self, trial_id, **fields):
        row = self.connection.execute('SELECT document FROM trials WHERE id=?', (trial_id,)).fetchone()
        if row is None:
            raise ValueError('实验记录不存在')
        document = json.loads(row[0])
        if set(fields) - {'validation', 'prediction', 'measurement', 'state', 'failure', 'trace', 'completed_at'}:
            raise ValueError('实验更新包含未知字段')
        document.update(fields)
        self.connection.execute('UPDATE trials SET document=? WHERE id=?',
            (json.dumps(document, allow_nan=False), trial_id))
        if fields.get('measurement') is not None:
            self.record_actual_cost_checkpoint('measurement_completed', trial_id)
        self.connection.commit()
        return document

    def add_evidence(self, identifier, kind, document):
        if kind not in ('MEASURED', 'PREDICTED', 'PROJECTED', 'HYPOTHESIS', 'FAILURE'):
            raise ValueError('证据类型未注册')
        if kind in ('PREDICTED', 'PROJECTED') and 'prediction_id' in document:
            document = dict(document)
            document.pop('evaluation_seconds', None)
            document['diagnosis'] = {key: value for key, value in document.get('diagnosis', {}).items()
                if key not in ('full_replays', 'prediction_cache_hit', 'reused_region_costs', 'new_region_costs')}
        encoded = json.dumps(document, sort_keys=True, allow_nan=False)
        previous = self.connection.execute('SELECT kind, document FROM evidence WHERE id=?', (identifier,)).fetchone()
        if previous and previous != (kind, encoded):
            raise ValueError('同一证据身份的内容发生变化')
        self.connection.execute('INSERT OR IGNORE INTO evidence VALUES (?, ?, ?)', (identifier, kind, encoded))
        self.connection.commit()

    def charge(self, identifier, resource, amount):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError('资源成本必须为有限非负实测数值')
        existing = self.connection.execute('SELECT resource, amount FROM charges WHERE id=?', (identifier,)).fetchone()
        if existing and existing != (resource, amount):
            raise ValueError('已登记资源成本发生变化')
        self.connection.execute('INSERT OR IGNORE INTO charges VALUES (?, ?, ?)', (identifier, resource, amount))
        if existing is None:
            self.record_actual_cost_checkpoint('resource_charged', identifier)
        self.connection.commit()

    def set_cost_clock(self, session_id, started):
        if not math.isfinite(started) or started > time.perf_counter():
            raise ValueError('实际费用时钟需要已经开始的单调计时')
        self.cost_clock = (session_id, started)

    def record_actual_cost_checkpoint(self, kind, identifier):
        if self.cost_clock is None:
            return
        session, started = self.cost_clock
        elapsed = time.perf_counter() - started
        self.connection.execute('INSERT INTO search_sessions VALUES (?, ?) '
            'ON CONFLICT(id) DO UPDATE SET elapsed_seconds=MAX(elapsed_seconds, excluded.elapsed_seconds)',
            (session, elapsed))
        completed = [row['trial_id'] for row in self.trials() if row.get('measurement') is not None]
        document = dict(at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=self.search_seconds(), actual_usage=self.usage(),
            completed_trial_ids=completed, timing_scope='actual_search_sessions_including_agent_and_tools')
        self.connection.execute('INSERT INTO actual_cost_checkpoints (kind, identifier, document) VALUES (?, ?, ?)',
            (kind, identifier, json.dumps(document, allow_nan=False)))

    def actual_cost_checkpoints(self):
        return [dict(sequence=row[0], kind=row[1], identifier=row[2], document=json.loads(row[3]))
            for row in self.connection.execute(
                'SELECT sequence, kind, identifier, document FROM actual_cost_checkpoints ORDER BY sequence')]

    def charged(self, identifier):
        return self.connection.execute('SELECT resource, amount FROM charges WHERE id=?', (identifier,)).fetchone()

    def usage(self):
        return dict(self.connection.execute('SELECT resource, SUM(amount) FROM charges GROUP BY resource'))

    def record_session_time(self, identifier, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('搜索运行时间必须为有限非负数值')
        self.connection.execute('INSERT INTO search_sessions VALUES (?, ?) '
            'ON CONFLICT(id) DO UPDATE SET elapsed_seconds=MAX(elapsed_seconds, excluded.elapsed_seconds)',
            (identifier, seconds))
        self.connection.commit()

    def search_seconds(self):
        return self.connection.execute('SELECT COALESCE(SUM(elapsed_seconds), 0) FROM search_sessions').fetchone()[0]

    def family_records(self):
        from search.family_state import FamilyLedger
        return FamilyLedger(self).records()

    def get_round(self, index):
        row = self.connection.execute('SELECT document FROM search_rounds WHERE id=?', (index,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_round(self, index, **fields):
        document = self.get_round(index) or dict(round_index=index, stage='CREATED')
        document.update(fields)
        self.connection.execute('INSERT INTO search_rounds VALUES (?, ?) '
            'ON CONFLICT(id) DO UPDATE SET document=excluded.document',
            (index, json.dumps(document, allow_nan=False)))
        self.connection.commit()
        return document

    def replace_round(self, index, document):
        if document.get('round_index') != index:
            raise ValueError('搜索轮次文档与轮次身份不一致')
        self.connection.execute('INSERT INTO search_rounds VALUES (?, ?) '
            'ON CONFLICT(id) DO UPDATE SET document=excluded.document',
            (index, json.dumps(document, allow_nan=False)))
        self.connection.commit()
        return document

    def get_agent_session(self, identifier):
        row = self.connection.execute('SELECT document FROM agent_sessions WHERE id=?', (identifier,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_agent_session(self, identifier, document):
        self.connection.execute('INSERT INTO agent_sessions VALUES (?, ?) '
            'ON CONFLICT(id) DO UPDATE SET document=excluded.document',
            (identifier, json.dumps(document, allow_nan=False)))
        self.connection.commit()

    def record_agent_context(self, identifier, **fields):
        row = self.connection.execute(
            'SELECT document FROM agent_context_metrics WHERE id=?', (identifier,)).fetchone()
        document = json.loads(row[0]) if row else dict(context_id=identifier)
        document.update(fields)
        self.connection.execute('INSERT INTO agent_context_metrics VALUES (?, ?) '
            'ON CONFLICT(id) DO UPDATE SET document=excluded.document',
            (identifier, json.dumps(document, allow_nan=False)))
        self.connection.commit()
        return document

    def agent_context_metrics(self):
        return [json.loads(row[0]) for row in self.connection.execute(
            'SELECT document FROM agent_context_metrics ORDER BY rowid')]

    def add_note(self, trial_id, kind, content, evidence_refs):
        if kind not in ('MEASURED', 'PREDICTED', 'PROJECTED', 'HYPOTHESIS', 'FAILURE') or not evidence_refs:
            raise ValueError('经验记录需要类型和证据引用')
        known = {row[0] for row in self.connection.execute('SELECT id FROM evidence')}
        if not set(evidence_refs) <= known:
            raise ValueError('经验记录引用了不存在的证据')
        document = dict(content=content, evidence_refs=evidence_refs)
        identifier = fingerprint(dict(trial=trial_id, kind=kind, **document))
        self.connection.execute('INSERT OR IGNORE INTO scratchpad VALUES (?, ?, ?, ?)',
            (identifier, trial_id, kind, json.dumps(document, allow_nan=False)))
        self.connection.commit()
        return identifier

    def trials(self):
        return [json.loads(row[0]) for row in self.connection.execute('SELECT document FROM trials ORDER BY rowid')]

    def put_profile_request(self, request):
        from contracts import ProfilingRequest
        request = ProfilingRequest.model_validate(request)
        existing = self.connection.execute('SELECT document FROM profiling_requests WHERE id=?',
            (request.request_id,)).fetchone()
        document = request.model_dump(mode='json')
        if existing:
            previous = json.loads(existing[0])
            for key in ('measurement_type', 'execution_request'):
                if previous[key] != document[key]:
                    raise ValueError('同一Profiling请求身份对应不同测量条件')
            previous['trial_ids'] = sorted(set(previous['trial_ids'] + document['trial_ids']))
            previous['cost_keys'] = sorted(set(previous['cost_keys'] + document['cost_keys']))
            document = previous
        self.connection.execute('INSERT INTO profiling_requests VALUES (?, ?) '
            'ON CONFLICT(id) DO UPDATE SET document=excluded.document',
            (request.request_id, json.dumps(document, allow_nan=False)))
        self.connection.commit()
        return document

    def update_profile_request(self, identifier, **fields):
        from contracts import ProfilingRequest
        row = self.connection.execute('SELECT document FROM profiling_requests WHERE id=?', (identifier,)).fetchone()
        if row is None:
            raise ValueError('Profiling请求身份不存在')
        if set(fields) - {'state', 'result', 'reason'}:
            raise ValueError('Profiling请求更新包含未知字段')
        document = ProfilingRequest.model_validate(json.loads(row[0]) | fields).model_dump(mode='json')
        self.connection.execute('UPDATE profiling_requests SET document=? WHERE id=?',
            (json.dumps(document, allow_nan=False), identifier))
        self.connection.commit()
        return document

    def profile_requests(self):
        return [json.loads(row[0]) for row in self.connection.execute(
            'SELECT document FROM profiling_requests ORDER BY rowid')]

    def notes(self):
        return [dict(id=row[0], trial_id=row[1], kind=row[2], **json.loads(row[3]))
            for row in self.connection.execute('SELECT * FROM scratchpad ORDER BY rowid')]

    def best_measured(self):
        eligible = [row for row in self.trials() if row['measurement'] and row['measurement']['status'] == 'PASS'
            and row['measurement'].get('performance_eligible', False)]
        return max(eligible, key=lambda row: row['measurement']['tokens_per_second'], default=None)

    def export(self, path):
        Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in self.trials()))
