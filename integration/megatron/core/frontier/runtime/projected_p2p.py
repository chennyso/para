import torch.distributed as dist

from megatron.core.frontier.runtime.projected_payload import ProjectedPayloadRuntime


def install_projected_p2p(plan, group_state):
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

    if getattr(P2PCommunicator, '_frontier_project_plan', None) is not None:
        raise ValueError('one executable row boundary must own the pipeline communicator')
    original_init = P2PCommunicator.__init__
    original_communicate = P2PCommunicator._communicate
    instances = {}

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._frontier_project_endpoint = dist.get_rank() in plan.source_ranks + plan.destination_ranks

    def adapted(self, **kwargs):
        source = dist.get_rank() in plan.source_ranks
        if not self._frontier_project_endpoint:
            return original_communicate(self, **kwargs)
        bound = (kwargs['tensor_send_next'] is not None or kwargs['recv_next']) if source else (
            kwargs['tensor_send_prev'] is not None or kwargs['recv_prev'])
        if not bound:
            return original_communicate(self, **kwargs)
        other = (kwargs['tensor_send_prev'] is not None or kwargs['recv_prev']) if source else (
            kwargs['tensor_send_next'] is not None or kwargs['recv_next'])
        if other or not kwargs.get('wait_on_reqs', True):
            raise ValueError('row transport requires its compiled one-edge 1F1B communication call')
        shape = kwargs['tensor_shape']
        if shape is not None and tuple(shape) != plan.shard_shape:
            raise ValueError('native logical hidden shape differs from the compiled row boundary')
        if group_state['handoff'] is None:
            raise RuntimeError('handoff groups must exist before the actual model pipeline call')
        rank = dist.get_rank()
        if rank not in instances:
            instances[rank] = ProjectedPayloadRuntime(plan, group_state['handoff'], group_state.get('diagnostic_directory'),
                                                      group_state.get('profiling_origin'))
        incoming, gradient = instances[rank].communicate(
            forward_tensor=kwargs['tensor_send_next'] if source else None,
            backward_tensor=kwargs['tensor_send_prev'] if not source else None,
            receive_forward=kwargs['recv_prev'] if not source else False,
            receive_backward=kwargs['recv_next'] if source else False)
        return incoming, gradient, None

    P2PCommunicator.__init__ = initialize
    P2PCommunicator._communicate = adapted
    P2PCommunicator._frontier_project_plan = plan
    return instances
