from itertools import product
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix
from search.module_relations import prefix_support, validate_relations


def task_stage(row,lanes):
    logical = row.get('logical_rank',row.get('metadata', {}).get('logical_rank',row['rank']))
    return logical//lanes


@dataclass(frozen=True)
class PartitionDomain:
    pp: int
    vpp: int
    boundaries: tuple
    seed: tuple

    def __post_init__(self):
        if self.pp < 1 or self.vpp < 1 or len(self.boundaries) != self.pp * self.vpp + 1:
            raise ValueError('分区范围与物理阶段数量不一致')
        if any(not values or tuple(sorted(set(values))) != tuple(values) for values in self.boundaries):
            raise ValueError('端点范围必须递增且没有重复')
        if self.boundaries[0] != (0,) or len(self.boundaries[-1]) != 1:
            raise ValueError('分区必须具有固定的首尾端点')
        if any(type(value) is not int for values in self.boundaries for value in values):
            raise ValueError('层端点必须使用整数')
        self.validate(self.seed)

    @classmethod
    def around(cls, request, radius):
        if type(radius) is not int or radius < 0:
            raise ValueError('端点半径必须为非负整数')
        seed = [0]
        for count in request['decoder_chunks']:
            seed.append(seed[-1] + count)
        size, slots = seed[-1], len(seed) - 1
        domains = [(0,)] + [tuple(range(max(j, value - radius),
            min(size - slots + j, value + radius) + 1))
            for j, value in enumerate(seed[1:-1], 1)] + [(size,)]
        return cls(request['pp'], request['vpp'], tuple(domains), tuple(seed))

    def validate(self, endpoints):
        if (len(endpoints) != len(self.boundaries)
                or any(type(value) is not int or value not in domain
                    for value, domain in zip(endpoints, self.boundaries))
                or any(a >= b for a, b in zip(endpoints, endpoints[1:]))):
            raise ValueError('完整分区的端点超出候选空间或没有完整递增')

    def intervals(self, endpoints, stage):
        self.validate(endpoints)
        if type(stage) is not int or not 0 <= stage < self.pp:
            raise ValueError('物理阶段编号非法')
        return tuple((endpoints[j], endpoints[j + 1])
            for j in range(stage, self.pp * self.vpp, self.pp))

    def witness(self, stage, intervals):
        if type(stage) is not int or not 0 <= stage < self.pp or len(intervals) != self.vpp:
            raise ValueError('候选必须包含同一物理阶段的全部虚拟区间')
        return self.partial_witness(stage, intervals)

    def partial_witness(self, stage, intervals):
        if (type(stage) is not int or not 0 <= stage < self.pp
                or not 0 <= len(intervals) <= self.vpp):
            raise ValueError('部分候选必须依次指定同一物理阶段的虚拟区间')
        fixed = {}
        for j, pair in zip(range(stage, self.pp * self.vpp, self.pp), intervals):
            if len(pair) != 2:
                raise ValueError('候选区间需要起始与结束端点')
            for index, value in ((j, pair[0]), (j + 1, pair[1])):
                if type(value) is not int or value not in self.boundaries[index]:
                    raise ValueError('候选端点超出声明范围')
                if index in fixed and fixed[index] != value:
                    raise ValueError('候选相邻区间没有连接')
                fixed[index] = value
        witness = self.complete_fixed(fixed)
        if witness is None:
            raise ValueError('该物理阶段候选没有可连接的完整分区')
        return witness

    def complete_fixed(self, fixed):
        if any(type(j) is not int or not 0 <= j < len(self.boundaries)
                or type(value) is not int or value not in self.boundaries[j]
                for j, value in fixed.items()):
            raise ValueError('固定端点超出声明范围')
        domains = [tuple([fixed[j]]) if j in fixed else values
            for j, values in enumerate(self.boundaries)]
        suffix = [set() for _ in domains]
        suffix[-1] = set(domains[-1])
        for j in reversed(range(len(domains) - 1)):
            suffix[j] = {a for a in domains[j] if any(a < b for b in suffix[j + 1])}
        if not suffix[0]:
            return None
        chosen = []
        for j, values in enumerate(suffix):
            feasible = [value for value in values if not chosen or chosen[-1] < value]
            chosen.append(min(feasible, key=lambda value: (abs(value - self.seed[j]), value)))
        self.validate(chosen)
        return tuple(chosen)

    def columns(self, stage):
        indices = sorted({index for j in range(stage, self.pp * self.vpp, self.pp)
            for index in (j, j + 1)})
        for values in product(*(self.boundaries[index] for index in indices)):
            fixed = dict(zip(indices, values))
            intervals = tuple((fixed[j], fixed[j + 1])
                for j in range(stage, self.pp * self.vpp, self.pp))
            if any(a >= b for a, b in intervals):
                continue
            # 可达性动态规划排除无法组成完整分区的端点。
            reachable = {0}
            for j, domain in enumerate(self.boundaries[1:], 1):
                choices = [fixed[j]] if j in fixed else domain
                reachable = {b for b in choices if any(a < b for a in reachable)}
            if reachable:
                yield intervals

    def layout_count(self):
        counts = {0: 1}
        for domain in self.boundaries[1:]:
            counts = {b: sum(value for a, value in counts.items() if a < b) for b in domain}
        return sum(counts.values())


@dataclass(frozen=True)
class PhysicalStageColumn:
    identifier: str
    stage: int
    intervals: tuple
    durations: dict
    local_peaks: dict
    local_admitted: bool
    evidence: dict


class NativeEventMaster:
    def __init__(self,domain,tasks,world_size,cost_revision,module_relations=()):
        self.domain = domain
        if world_size % domain.pp:
            raise ValueError('设备数量必须能够被物理阶段数量整除')
        self.lanes = world_size // domain.pp
        self.cost_revision = cost_revision
        self.tasks = tasks
        self.keys = tuple(sorted(tasks))
        self.task_index = {key: index for index, key in enumerate(self.keys)}
        self.edges = sorted({(parent, key) for key, row in tasks.items() for parent in row['dependencies']})
        if any(parent not in tasks or parent == child for parent, child in self.edges):
            raise ValueError('原生事件具有非法依赖')
        successors = {key: set() for key in tasks}
        for parent, child in self.edges:
            successors[parent].add(child)
        self.terminals = tuple(key for key in self.keys if not successors[key])
        self.duration_rows = [parent for parent, _ in self.edges] + list(self.terminals)
        self.boundary_rows = [(j, value) for j in range(1, domain.pp * domain.vpp)
            for value in domain.boundaries[j]]
        self.boundary_index = {key: domain.pp + index for index, key in enumerate(self.boundary_rows)}
        self.columns = {}
        self.interfaces = {}
        self.interface_rows = {}
        self.module_relations = validate_relations(module_relations,domain.seed[-1],domain.pp*domain.vpp)
        self.module_eq_rows = {}
        self.module_ub_rows = {}

    def interval_module_coefficients(self,stage,vp,pair):
        eq,ub = {},{}
        start,end = pair
        for (relation_index,target),index in self.module_eq_rows.items():
            relation = self.module_relations[relation_index]
            applies = (stage==target if relation.relation=='same_physical'
                else vp*self.domain.pp+stage==target)
            if applies:
                left,right = relation.module_ids
                eq[index] = int(start<=left<end)-int(start<=right<end)
        for (relation_index,target),index in self.module_ub_rows.items():
            if stage==target:
                left,right = self.module_relations[relation_index].module_ids
                ub[index] = int(start<=left<end)+int(start<=right<end)
        return eq,ub

    def add(self, column):
        self.domain.witness(column.stage, column.intervals)
        expected = {key for key, row in self.tasks.items() if task_stage(row,self.lanes) == column.stage}
        if set(column.durations) != expected:
            raise ValueError('候选没有完整覆盖所属Rank的原生事件')
        if any(not math.isfinite(value) or value < 0 for value in column.durations.values()):
            raise ValueError('原生事件成本必须有限且非负')
        if column.evidence['cost_revision'] != self.cost_revision:
            raise ValueError('候选成本版本与主问题不同')
        if not column.local_admitted:
            return False
        for relation in self.module_relations:
            if relation.relation=='allowed_virtual':
                for vp,(start,end) in enumerate(column.intervals):
                    if (vp*self.domain.pp+column.stage not in relation.virtual_slots
                            and any(start<=module<end for module in relation.module_ids)):
                        return False
        for identity,row in column.evidence.get('cross_stage_interfaces', {}).items():
            stages = tuple(row['stages'])
            if column.stage not in stages or len(stages) < 2 or not set(stages) <= set(range(self.domain.pp)):
                raise ValueError('通信接口的物理阶段集合非法')
            if identity in self.interfaces and self.interfaces[identity] != stages:
                raise ValueError('固定主问题中的通信接口参与阶段发生改变')
            self.interfaces[identity] = stages
        expected_interfaces = {key for key,stages in self.interfaces.items() if column.stage in stages}
        if set(column.evidence.get('cross_stage_interfaces', {})) != expected_interfaces:
            raise ValueError('候选缺少跨物理阶段的通信接口')
        if column.identifier in self.columns and self.columns[column.identifier] != column:
            raise ValueError('候选身份重复且成本内容不同')
        fresh = column.identifier not in self.columns
        self.columns[column.identifier] = column
        return fresh

    def coefficients(self, column):
        eq = {column.stage: 1.0}
        for vp, (a, b) in enumerate(column.intervals):
            slot = vp * self.domain.pp + column.stage
            if slot:
                index = self.boundary_index[(slot, a)]
                eq[index] = eq.get(index, 0.0) - 1.0
            if slot + 1 < self.domain.pp * self.domain.vpp:
                index = self.boundary_index[(slot + 1, b)]
                eq[index] = eq.get(index, 0.0) + 1.0
        for (identity,value,first,other),index in self.interface_rows.items():
            interface = column.evidence.get('cross_stage_interfaces', {}).get(identity)
            if (column.stage in (first,other) and interface is not None
                    and interface['signature'] == value):
                eq[index] = 1.0 if column.stage == first else -1.0
        ub = {index: column.durations[key] for index, key in enumerate(self.duration_rows)
            if key in column.durations and column.durations[key] != 0}
        for vp,pair in enumerate(column.intervals):
            extra_eq,extra_ub = self.interval_module_coefficients(column.stage,vp,pair)
            for index,value in extra_eq.items():
                eq[index] = eq.get(index,0)+value
            for index,value in extra_ub.items():
                ub[index] = ub.get(index,0)+value
        return eq, ub

    def matrices(self):
        if {column.stage for column in self.columns.values()} != set(range(self.domain.pp)):
            raise ValueError('主问题需要覆盖全部物理阶段的完整种子')
        columns = tuple(sorted(self.columns.values(), key=lambda column: column.identifier))
        for column in columns:
            expected = {key for key,stages in self.interfaces.items() if column.stage in stages}
            if set(column.evidence.get('cross_stage_interfaces', {})) != expected:
                raise ValueError('已有候选没有覆盖当前全部通信接口')
        interface_rows = []
        for identity,stages in sorted(self.interfaces.items()):
            values = sorted({column.evidence['cross_stage_interfaces'][identity]['signature']
                for column in columns if column.stage in stages})
            interface_rows.extend((identity,value,stages[0],stage)
                for value in values for stage in stages[1:])
        interface_start = self.domain.pp + len(self.boundary_rows)
        self.interface_rows = {key:interface_start + index for index,key in enumerate(interface_rows)}
        module_start = interface_start+len(interface_rows)
        module_eq_keys,module_ub_keys = [],[]
        for index,relation in enumerate(self.module_relations):
            if relation.relation in ('same_physical','same_virtual'):
                size = self.domain.pp if relation.relation=='same_physical' else self.domain.pp*self.domain.vpp
                module_eq_keys.extend((index,target) for target in range(size))
            elif relation.relation=='different_physical':
                module_ub_keys.extend((index,target) for target in range(self.domain.pp))
        self.module_eq_rows = {key:module_start+index for index,key in enumerate(module_eq_keys)}
        self.module_ub_rows = {key:len(self.duration_rows)+index for index,key in enumerate(module_ub_keys)}
        ncols, nevents = len(columns), len(self.keys)
        nvars = ncols + nevents + 1
        eq_rows, eq_cols, eq_values = [], [], []
        ub_rows, ub_cols, ub_values = [], [], []
        for index, column in enumerate(columns):
            eq, ub = self.coefficients(column)
            for row, value in eq.items():
                eq_rows.append(row); eq_cols.append(index); eq_values.append(value)
            for row, value in ub.items():
                ub_rows.append(row); ub_cols.append(index); ub_values.append(value)
        for row, (parent, child) in enumerate(self.edges):
            ub_rows.extend((row, row))
            ub_cols.extend((ncols + self.task_index[parent], ncols + self.task_index[child]))
            ub_values.extend((1.0, -1.0))
        for offset, key in enumerate(self.terminals, len(self.edges)):
            ub_rows.extend((offset, offset))
            ub_cols.extend((ncols + self.task_index[key], nvars - 1))
            ub_values.extend((1.0, -1.0))
        eq = coo_matrix((eq_values, (eq_rows, eq_cols)),
            shape=(module_start+len(module_eq_keys),nvars)).tocsr()
        ub = coo_matrix((ub_values, (ub_rows, ub_cols)),
            shape=(len(self.duration_rows)+len(module_ub_keys),nvars)).tocsr()
        self.ub_rhs = np.array([0.0]*len(self.duration_rows)+[1.0]*len(module_ub_keys))
        rhs = np.array([1.0]*self.domain.pp+[0.0]*(len(self.boundary_rows)+len(interface_rows)+len(module_eq_keys)))
        objective = np.zeros(nvars)
        objective[-1] = 1.0
        return columns, objective, eq, rhs, ub

    def solve_lp(self):
        columns, objective, eq, rhs, ub = self.matrices()
        result = linprog(objective,A_ub=ub,b_ub=self.ub_rhs,
            A_eq=eq, b_eq=rhs, bounds=(0, None), method='highs')
        if not result.success:
            raise RuntimeError('事件LP求解失败：' + result.message)
        prices = dict(eq=result.eqlin.marginals.tolist(), ub=result.ineqlin.marginals.tolist())
        errors = [abs(self.reduced_cost(column, prices) - result.lower.marginals[index])
            for index, column in enumerate(columns)]
        if max(errors, default=0) > 1e-5:
            raise ValueError('约化成本与LP求解器的实际列边际不一致')
        return dict(objective_ms=float(result.fun), prices=prices,
            positive_columns=[dict(id=column.identifier, weight=float(result.x[index]))
                for index, column in enumerate(columns) if result.x[index] > 1e-8],
            maximum_reduced_cost_residual=max(errors, default=0))

    def reduced_cost(self, column, prices):
        eq, ub = self.coefficients(column)
        return -sum(value * prices['eq'][index] for index, value in eq.items()) - sum(
            value * prices['ub'][index] for index, value in ub.items())

    def solve_integer(self, visited=(), time_limit=None):
        columns, objective, eq, rhs, ub = self.matrices()
        constraints = [LinearConstraint(eq,rhs,rhs),LinearConstraint(ub,-np.inf,self.ub_rhs)]
        by_id = {column.identifier: index for index, column in enumerate(columns)}
        exclusions = [tuple(ids) for ids in visited if set(ids) <= set(by_id)]
        if exclusions:
            rows, indices = zip(*[(row, by_id[identifier]) for row, ids in enumerate(exclusions)
                for identifier in ids])
            matrix = coo_matrix((np.ones(len(rows)), (rows, indices)),
                shape=(len(exclusions), len(objective))).tocsr()
            constraints.append(LinearConstraint(matrix, -np.inf, self.domain.pp - 1))
        integrality = np.array([1] * len(columns) + [0] * (len(objective) - len(columns)))
        upper = np.array([1.0] * len(columns) + [np.inf] * (len(objective) - len(columns)))
        options = dict(mip_rel_gap=0)
        if time_limit is not None:
            if not math.isfinite(time_limit) or time_limit <= 0:
                raise ValueError('整数求解时限必须为有限正数或None')
            options['time_limit'] = time_limit
        result = milp(objective, integrality=integrality, bounds=Bounds(0, upper),
            constraints=constraints, options=options)
        if result.status == 2:
            return None
        if result.x is None or result.status not in (0, 1):
            raise RuntimeError('事件整数主问题没有取得可行解：' + result.message)
        chosen = [column for index, column in enumerate(columns) if result.x[index] > 0.5]
        if len(chosen) != self.domain.pp or len({column.stage for column in chosen}) != self.domain.pp:
            raise ValueError('整数解未选择每个物理阶段的一项候选')
        endpoints = [None] * len(self.domain.boundaries)
        for column in chosen:
            for vp, (a, b) in enumerate(column.intervals):
                slot = vp * self.domain.pp + column.stage
                for j, value in ((slot, a), (slot + 1, b)):
                    if endpoints[j] is not None and endpoints[j] != value:
                        raise ValueError('整数解区间连接不一致')
                    endpoints[j] = value
        self.domain.validate(endpoints)
        counts = [b-a for a,b in zip(endpoints,endpoints[1:])]
        if not prefix_support(counts,self.domain.seed[-1],self.domain.pp,self.domain.vpp,self.module_relations):
            raise ValueError('整数分区违反模块归属约束')
        # 独立计算最长依赖路径，核查整数事件目标。
        durations = {key: value for column in chosen for key, value in column.durations.items()}
        pending = set(self.keys)
        ends = {}
        while pending:
            ready = [key for key in pending if all(parent in ends for parent in self.tasks[key]['dependencies'])]
            if not ready:
                raise ValueError('原生事件出现循环依赖')
            for key in ready:
                ends[key] = max((ends[parent] for parent in self.tasks[key]['dependencies']), default=0) + durations[key]
                pending.remove(key)
        value = max(ends.values())
        if value > result.fun + 1e-4 or (result.status == 0 and not math.isclose(
                value, result.fun, abs_tol=1e-4, rel_tol=1e-6)):
            raise ValueError('整数目标与独立事件最长路径不同')
        return dict(objective_ms=value, endpoints=endpoints,
            decoder_chunks=[b - a for a, b in zip(endpoints, endpoints[1:])],
            column_ids=[column.identifier for column in chosen], optimal=result.status == 0,
            cross_stage_interfaces={key:next(column.evidence['cross_stage_interfaces'][key]['signature']
                for column in chosen if column.stage in stages) for key,stages in self.interfaces.items()})

    def price_observation(self, lp):
        prices = lp['prices']
        weights = {}
        for row, key in enumerate(self.duration_rows):
            weights[key] = weights.get(key, 0.0) - prices['ub'][row]
        return dict(lp_objective_ms=lp['objective_ms'], cost_revision=self.cost_revision,
            event_prices=[dict(task_id=key, rank=self.tasks[key]['rank'],
                region=self.tasks[key].get('region'), kind=self.tasks[key]['kind'],
                seed_duration_ms=self.tasks[key]['duration_ms'], price=value)
                for key, value in weights.items() if value > 1e-8],
            convexity_prices=prices['eq'][:self.domain.pp],
            interface_prices=[dict(interface_id=identity,signature=value,first_stage=first,
                other_stage=other,price=-prices['eq'][index])
                for (identity,value,first,other),index in self.interface_rows.items()
                if abs(prices['eq'][index]) > 1e-8],
            boundary_prices=[dict(boundary=j, endpoint=value, price=-prices['eq'][index])
                for (j, value), index in self.boundary_index.items() if abs(prices['eq'][index]) > 1e-8],
            generated_columns=[dict(id=column.identifier, stage=column.stage, intervals=column.intervals,
                reduced_cost=self.reduced_cost(column, prices)) for column in self.columns.values()])

    def compact_price_observation(self,lp,limit=16):
        complete = self.price_observation(lp)
        groups = {}
        for row in complete['event_prices']:
            key = (task_stage(self.tasks[row['task_id']],self.lanes),row['region'],row['kind'])
            group = groups.setdefault(key,dict(stage=key[0],region=key[1],kind=key[2],
                task_count=0,price_sum=0.0,priced_work_ms=0.0))
            group['task_count'] += 1
            group['price_sum'] += row['price']
            group['priced_work_ms'] += row['price']*row['seed_duration_ms']
        summary = dict(complete,event_prices=sorted(complete['event_prices'],
            key=lambda row:(-row['price']*row['seed_duration_ms'],row['task_id']))[:limit],
            event_price_aggregates=list(groups.values()),event_price_count=len(complete['event_prices']),
            generated_columns=sorted(complete['generated_columns'],key=lambda row:row['reduced_cost'])[:24],
            generated_column_count=len(complete['generated_columns']),
            detail_scope='complete_prices_remain_in_cpu_solver_and_registered_artifact')
        return complete,summary
