from copy import deepcopy
import math

from contracts import fingerprint


class FragmentCostCache:
    def __init__(self, identity, module_costs):
        self.identity = fingerprint(dict(identity=identity, module_costs=module_costs))
        self.rows = {row['layer']: dict(row) for row in module_costs}
        if len(self.rows) != len(module_costs) or set(self.rows) != set(range(len(self.rows))):
            raise ValueError('模块成本必须连续覆盖全部模型层')
        self.cache = {}
        self.hits = self.misses = 0

    def interval(self, start, end):
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(self.rows)):
            raise ValueError('局部成本区间必须位于实际模型范围')
        key = (start, end)
        if key not in self.cache:
            result = {}
            for field in ('forward_ms', 'backward_ms', 'state_bytes'):
                values = [self.rows[layer][field] for layer in range(start, end)]
                if any(type(value) not in (float, int) or not math.isfinite(value)
                        or value < 0 for value in values):
                    raise ValueError('局部成本需要有限的非负原始数据')
                result[field] = math.fsum(values)
            self.cache[key] = result
            self.misses += 1
        else:
            self.hits += 1
        return deepcopy(self.cache[key])

    def report(self):
        return dict(identity=self.identity, reused_fragment_costs=self.hits,
            constructed_fragments=self.misses,
            scope='same_identity_atomic_module_sums_for_pricing_order',
            complete_replay_reused=False)
