from dataclasses import asdict, dataclass
from itertools import accumulate
import math

from contracts import fingerprint
from predictor.profile_database import plan_signature, signature
from search.module_relations import prefix_support
from search.physical_stage_master import PartitionDomain


@dataclass(frozen=True)
class PrefixBoundaryCertificate:
    identifier: str
    execution_family_id: str
    boundary_identity: str
    prefix: tuple
    suffixes: tuple
    metrics: tuple
    prediction_ids: tuple
    graph_ids: tuple
    cost_revision: str


class ExecutionBoundaryIndex:
    def __init__(self,cost_revision,scope_identity,objective='time_subject_to_capacity'):
        if not cost_revision:
            raise ValueError('状态支配证书需要明确的成本版本')
        self.cost_revision = cost_revision
        if not scope_identity:
            raise ValueError('状态支配证书需要明确的设备、后端和执行模型身份')
        self.scope_identity = scope_identity
        if objective not in ('time_subject_to_capacity','time_and_memory'):
            raise ValueError('状态支配目标没有注册')
        self.objective = objective
        self.records = {}
        self.families = set()
        self.certificates = {}
        self.cached = {}
        self.comparisons = []
        self.revision = 0

    def register(self,request,report):
        if report.get('cost_database_revision') != self.cost_revision:
            return False
        if report.get('plan_signature') != plan_signature(request):
            raise ValueError('状态支配证据的完整训练配置不一致')
        performance,memory = report.get('performance', {}),report.get('memory', {})
        step = performance.get('step_time_ms')
        if (report.get('prediction_status') not in ('PREDICTED_CALIBRATED','PREDICTED')
                or step is None or memory.get('admission') != 'WITHIN_ESTIMATED_CAPACITY'):
            return False
        if not report.get('evidence', {}).get('predictor_revision'):
            return False
        if not math.isfinite(step) or step <= 0:
            raise ValueError('完整DAG支配证据需要有限正数步时')
        graph = report.get('execution_graph')
        if not graph or not graph.get('tasks'):
            return False
        if graph.get('full_task_count',graph.get('task_count',len(graph['tasks']))) != len(graph['tasks']):
            raise ValueError('状态支配证据需要完整执行图')
        peaks = {int(rank):value for rank,value in memory.get('peak_per_rank_bytes', {}).items()}
        headroom = {int(rank):value for rank,value in memory.get('memory_headroom_per_rank_bytes', {}).items()}
        world = request['node_count']*request['gpus_per_node']
        if set(peaks) != set(range(world)) or set(headroom) != set(peaks):
            return False
        if any(type(value) not in (int,float) or not math.isfinite(value) or value < 0
                for value in [*peaks.values(),*headroom.values()]):
            raise ValueError('完整DAG支配证据的逐卡显存数据非法')
        capacities = tuple(peaks[rank]+headroom[rank] for rank in range(world))
        document = dict(request=request,prediction_id=report['prediction_id'],
            graph_id=fingerprint(graph),metrics=(step,*(peaks[rank] for rank in range(world))),
            capacities=capacities,predictor_revision=report.get('evidence', {}).get('predictor_revision'),
            comparison_context=fingerprint({name:report.get('evidence', {}).get(name) for name in
                ('blind','cost_mode','graph_delta_calibration','prediction_independence_scope')}))
        family = signature(request)
        counts = tuple(request['decoder_chunks'])
        key = (family,counts)
        if key in self.records and self.records[key] == document:
            return True
        self.records[key] = document
        self.families.add(family)
        self.cached.clear()
        self.revision += 1
        return True

    @staticmethod
    def unrestricted_domain(request):
        layers,slots = sum(request['decoder_chunks']),request['pp']*request['vpp']
        boundaries = ((0,),)+tuple(tuple(range(index,layers-slots+index+1))
            for index in range(1,slots))+((layers,),)
        return PartitionDomain(request['pp'],request['vpp'],boundaries,
            tuple(accumulate(request['decoder_chunks'],initial=0)))

    @staticmethod
    def suffixes(domain,prefix,relations):
        ends = tuple(accumulate(prefix,initial=0))
        if len(prefix) > domain.pp*domain.vpp or any(value <= 0 for value in prefix):
            raise ValueError('续接证明需要合法的连续模块前缀')
        fixed = {index:value for index,value in enumerate(ends)}
        if any(value not in domain.boundaries[index] for index,value in fixed.items()):
            return
        if domain.complete_fixed(fixed) is None:
            return

        def visit(counts,end):
            if not prefix_support(counts,domain.seed[-1],domain.pp,domain.vpp,relations):
                return
            if len(counts) == domain.pp*domain.vpp:
                if end == domain.seed[-1]:
                    yield counts[len(prefix):]
                return
            for next_end in domain.boundaries[len(counts)+1]:
                if next_end > end:
                    yield from visit(counts+(next_end-end,),next_end)

        yield from visit(tuple(prefix),ends[-1])

    def certify(self,request,prefix,domain=None,relations=()):
        if signature(request) not in self.families:
            return None
        domain = self.unrestricted_domain(request) if domain is None else domain
        relation_records = [row.model_dump(mode='json') if hasattr(row,'model_dump') else row
            for row in relations]
        key = fingerprint(dict(family=signature(request),prefix=prefix,domain=asdict(domain),
            relations=relation_records,revision=self.revision))
        if key in self.cached:
            identifier = self.cached[key]
            return self.certificates.get(identifier)
        records,suffixes = [],[]
        for suffix in self.suffixes(domain,tuple(prefix),relations):
            row = self.records.get((signature(request),tuple(prefix)+suffix))
            if row is None:
                self.cached[key] = None
                return None
            records.append(row)
            suffixes.append(suffix)
        if not records:
            self.cached[key] = None
            return None
        if len({(row['capacities'],row['predictor_revision'],row['comparison_context']) for row in records}) != 1:
            raise ValueError('续接证据的设备容量或执行预测版本不一致')
        boundary = dict(family=signature(request),depth=len(prefix),end=sum(prefix),
            legal_suffixes=suffixes,cost_revision=self.cost_revision,
            objective=self.objective,scope_identity=self.scope_identity,
            comparison_context=records[0]['comparison_context'],
            capacities=records[0]['capacities'],predictor_revision=records[0]['predictor_revision'])
        identity = fingerprint(dict(boundary=boundary,prefix=prefix,
            predictions=[row['prediction_id'] for row in records]))
        certificate = PrefixBoundaryCertificate(identity,signature(request),fingerprint(boundary),tuple(prefix),
            tuple(suffixes),tuple(row['metrics'] for row in records),
            tuple(row['prediction_id'] for row in records),tuple(row['graph_id'] for row in records),
            self.cost_revision)
        self.certificates[identity] = certificate
        self.cached[key] = identity
        return certificate

    def dominates(self,left_id,right_id):
        left,right = self.certificates[left_id],self.certificates[right_id]
        if left.boundary_identity != right.boundary_identity:
            return False
        if left.suffixes != right.suffixes:
            raise ValueError('相同边界身份的合法续接集合不一致')
        pairs = [(a,b) for first,second in zip(left.metrics,right.metrics)
            for a,b in zip(first[:1] if self.objective=='time_subject_to_capacity' else first,
                second[:1] if self.objective=='time_subject_to_capacity' else second)]
        dominates = all(a <= b for a,b in pairs) and any(a < b for a,b in pairs)
        if dominates:
            self.comparisons.append(dict(left_certificate=left_id,right_certificate=right_id,
                kept_prefix=left.prefix,pruned_prefix=right.prefix,
                boundary_identity=left.boundary_identity,complete_suffixes=len(left.suffixes),
                proof_scope='all_legal_suffixes_complete_dag_and_per_rank_capacity',objective=self.objective))
        return dominates

    def prunes_completion(self,request,counts):
        return any(certificate.execution_family_id == signature(request)
            and tuple(counts[:len(certificate.prefix)]) == certificate.prefix
            and tuple(counts[len(certificate.prefix):]) in certificate.suffixes
            for row in self.comparisons
            for certificate in (self.certificates[row['right_certificate']],))

    def report(self):
        return dict(cost_revision=self.cost_revision,scope_identity=self.scope_identity,complete_records=len(self.records),
            certificates=[asdict(row) for row in self.certificates.values()],
            dominance_comparisons=self.comparisons,
            objective=self.objective,
            proof_scope='exhaustive_complete_replay_over_declared_legal_suffix_domain',
            unknown_suffixes_pruned=False)
