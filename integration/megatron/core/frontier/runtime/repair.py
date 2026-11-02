import torch.distributed as dist

from megatron.core.frontier.planner.repair import local_shape


class RepairBoundaryExecutor:
    def __init__(self, result, rank, process_groups):
        if rank != dist.get_rank():
            raise ValueError('Boundary execution must bind the actual current training rank')
        self.bindings = result.bind_collectives(rank, process_groups)
        regions = {region.region_id: region for region in result.graph.regions}
        self.connections = {}
        for connection in result.graph.connections():
            key = (connection.producer, connection.producer_output,
                connection.consumer, connection.consumer_input)
            if key in self.connections:
                raise ValueError('Training boundary execution requires unique actual tensor connections')
            source = regions[connection.producer].training_contract
            destination = regions[connection.consumer].training_contract
            if source is None or destination is None:
                raise ValueError('Training boundary execution requires complete compiled region contracts')
            left = source.forward_outputs[connection.producer_output]
            right = destination.forward_inputs[connection.consumer_input]
            if rank not in left.physical_ranks or rank not in right.physical_ranks:
                continue
            if key not in self.bindings and not left.compatible_with(right):
                raise ValueError('Training tensor redistribution lacks an executable repair implementation')
            self.connections[key] = (left, right)

    def forward(self, connection, tensor):
        key = (connection.producer, connection.producer_output,
            connection.consumer, connection.consumer_input)
        source, destination = self.connections[key]
        if (tuple(tensor.shape) != local_shape(source)
                or str(tensor.dtype).removeprefix('torch.') != source.value.dtype
                or not tensor.is_cuda):
            raise ValueError('Actual training boundary tensor differs from its compiled source contract')
        output = self.bindings[key].forward(tensor) if key in self.bindings else tensor
        if (tuple(output.shape) != local_shape(destination)
                or str(output.dtype).removeprefix('torch.') != destination.value.dtype):
            raise ValueError('Executed communication repair did not satisfy its destination tensor contract')
        return output
