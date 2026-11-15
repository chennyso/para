from collections import defaultdict
import math


class LinearPricingBound:
    def __init__(self,master,oracle,lp,stage):
        self.master,self.oracle,self.prices,self.stage = master,oracle,lp['prices'],stage
        self.seed = master.domain.intervals(master.domain.seed,stage)
        self.base = next(column for column in master.columns.values()
            if column.stage==stage and column.intervals==self.seed)
        self.base_reduced_cost = master.reduced_cost(self.base,self.prices)
        self.weights = defaultdict(float)
        for index,key in enumerate(master.duration_rows):
            self.weights[key] -= self.prices['ub'][index]
        self.state = oracle.fragment_state_bytes(self.seed)
        if self.state <= 0:
            raise ValueError('定价排序下界需要正值的原始状态字节数')
        self.optimizer_weight = math.fsum(self.weights[key]*duration/self.state
            for key,duration in self.base.durations.items()
            if oracle.tasks[key]['kind'] not in ('forward_compute','backward_compute')
            and (oracle.tasks[key]['kind']=='optimizer_step'
                or oracle.tasks[key]['metadata'].get('state_family') in ('dense','expert')))
        self.cache = {}
        self.minimum = []
        for vp in range(master.domain.vpp):
            slot = vp*master.domain.pp+stage
            self.minimum.append(min(self.adjustment(vp,(left,right))
                for left in master.domain.boundaries[slot]
                for right in master.domain.boundaries[slot+1] if left<right))

    def adjustment(self,vp,pair):
        key = (vp,pair)
        if key in self.cache:
            return self.cache[key]
        cache = self.oracle.cost_fragments()
        previous,current = cache.interval(*self.seed[vp]),cache.interval(*pair)
        change = (current['state_bytes']-previous['state_bytes'])*self.optimizer_weight
        for task_id,duration in self.base.durations.items():
            task = self.oracle.tasks[task_id]
            if (task['kind'] not in ('forward_compute','backward_compute')
                    or task['region']//self.master.domain.pp != vp):
                continue
            field = 'forward_ms' if task['kind']=='forward_compute' else 'backward_ms'
            target = max(0.0,duration+current[field]-previous[field])
            change += self.weights[task_id]*(target-duration)
        slot = vp*self.master.domain.pp+self.stage
        for boundary,old,new,sign in ((slot,self.seed[vp][0],pair[0],1),
                (slot+1,self.seed[vp][1],pair[1],-1)):
            if 0 < boundary < self.master.domain.pp*self.master.domain.vpp:
                change += sign*(self.prices['eq'][self.master.boundary_index[boundary,new]]
                    -self.prices['eq'][self.master.boundary_index[boundary,old]])
        old_eq,old_ub = self.master.interval_module_coefficients(self.stage,vp,self.seed[vp])
        new_eq,new_ub = self.master.interval_module_coefficients(self.stage,vp,pair)
        change -= sum((new_eq.get(index,0)-old_eq.get(index,0))*self.prices['eq'][index]
            for index in set(old_eq)|set(new_eq))
        change -= sum((new_ub.get(index,0)-old_ub.get(index,0))*self.prices['ub'][index]
            for index in set(old_ub)|set(new_ub))
        self.cache[key] = change
        return change

    def bound(self,intervals):
        return self.base_reduced_cost+math.fsum(self.adjustment(vp,pair)
            for vp,pair in enumerate(intervals))+math.fsum(self.minimum[len(intervals):])
