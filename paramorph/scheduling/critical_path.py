import math

from scheduling.schedule_simulator import topological_order


class CriticalPathAnalyzer:
    def __init__(self):
        self.earliest_start = {}
        self.latest_finish = {}
        self.slack = {}
        self.makespan = 0.0

    def identify_critical_path(self, dag):
        if not dag.tasks:
            raise ValueError('关键路径分析需要完整任务集合')
        order, predecessors, successors = topological_order(dag.tasks, dag.edges)
        durations = {key: task.duration for key, task in dag.tasks.items()}
        if any(value is None or not math.isfinite(value) or value < 0 for value in durations.values()):
            raise ValueError('关键路径任务缺少合法耗时')
        ends, parents = {}, {}
        self.earliest_start, self.latest_finish, self.slack = {}, {}, {}
        for key in order:
            parent = max(predecessors[key], key=ends.get) if predecessors[key] else None
            start = ends[parent] if parent is not None else 0.0
            self.earliest_start[key] = start
            ends[key] = start + durations[key]
            parents[key] = parent
        last = max(ends, key=ends.get)
        self.makespan = ends[last]
        for key in reversed(order):
            finish = min((self.latest_finish[next_key] - durations[next_key]
                for next_key in successors[key]), default=self.makespan)
            self.latest_finish[key] = finish
            self.slack[key] = finish - ends[key]
        path = []
        while last is not None:
            path.append(last)
            last = parents[last]
        return list(reversed(path))

    def get_critical_path_length(self):
        if not self.latest_finish:
            raise ValueError('尚未计算关键路径')
        return self.makespan

    def get_task_slack(self, task_id):
        return self.slack[task_id]

    def is_on_critical_path(self, task_id):
        return abs(self.slack[task_id]) < 0.001
