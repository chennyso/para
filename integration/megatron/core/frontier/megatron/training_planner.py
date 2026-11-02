from dataclasses import dataclass

from megatron.core.frontier.megatron.model_adapter import PreparedTrainingProgram
from megatron.core.frontier.planner.repair import MandatoryRepair


@dataclass(frozen=True)
class UnsupportedTrainingProgram:
    repair_result: object
    reasons: tuple[str, ...]
    status: str = 'Unsupported'


class NativeRepairPlanner:
    def __init__(self, program, structure, capability_catalog, graph_factory, process_group_resolver,
            pipeline=None, source_graph=None, state_backend=None, initialize_from_backend=False):
        if program is None or not callable(graph_factory) or not callable(process_group_resolver):
            raise ValueError('Native repair planning requires its program and actual model/group registration callables')
        self.program = program
        self.structure = structure
        self.capability_catalog = capability_catalog
        self.graph_factory = graph_factory
        self.process_group_resolver = process_group_resolver
        self.pipeline = pipeline
        self.source_graph = source_graph
        self.state_backend = state_backend
        self.initialize_from_backend = initialize_from_backend

    def __call__(self, context, model_chunks):
        models = tuple(model_chunks)
        if not models or context.optimizer is None:
            raise ValueError('Native repair planning requires actual constructed models and an optimizer')
        graph = self.graph_factory(context, models)
        if graph.scope != 'MODEL' or graph.expected_parameter_keys is None:
            raise ValueError('Native repair planning requires a complete registered model parameter inventory')
        result = MandatoryRepair().close(self.structure, graph, self.capability_catalog,
            pipeline=self.pipeline, context=context, state_backend=self.state_backend,
            source_graph=self.source_graph, initialize_from_backend=self.initialize_from_backend)
        if result.status != 'LEGAL_STRUCTURAL' or not result.after.legal:
            reasons = result.unsupported_reasons or ('The complete training program has unresolved repair obligations',)
            return UnsupportedTrainingProgram(result, reasons)
        groups = self.process_group_resolver(context, models, result)
        return PreparedTrainingProgram(self.program, result, groups)
