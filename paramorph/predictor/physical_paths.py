from collections import defaultdict
from pathlib import PurePosixPath


def communication_resources(topology, ranks):
    devices = {row.get('physical_rank', row.get('rank')): row for row in topology['devices']}
    if len(set(ranks)) != len(ranks) or any(rank not in devices for rank in ranks):
        raise ValueError('物理通信路径需要完整且唯一的参与Rank')
    nodes = defaultdict(list)
    resources = set()
    paths = []
    missing = []

    def pci_resources(node, path):
        if not path:
            missing.append(f'node:{node}:pci_path')
            return []
        parts = PurePosixPath(path).parts
        bridges = [part for part in parts if part.startswith('pci') or ':' in part]
        if not bridges:
            raise ValueError('实测PCIe路径没有PCI设备或Root Complex')
        return [f'pcie:{node}:{part}' for part in bridges]

    for rank in ranks:
        device = devices[rank]
        node = device.get('node_index')
        if node is None:
            node = device.get('node', topology.get('hostname'))
        if node is None:
            raise ValueError('GPU实测拓扑缺少所属节点')
        node = str(node)
        nodes[node].append(rank)
        local = pci_resources(node, device.get('pci_path'))
        resources.update(local)
        resources.add(f'gpu:{rank}:communication')
        paths.append(dict(physical_rank=rank, node=node, numa_node=device.get('numa_node'),
            pci_path=device.get('pci_path'), resource_ids=local))
    adapters = topology.get('paths') or []
    for node, participants in nodes.items():
        per_node = topology.get('environment_per_node', {})
        environment = per_node.get(node, topology.get('environment', {}))
        if topology.get('environment_per_rank'):
            environment = topology['environment_per_rank'][str(participants[0])]
        if len(nodes) == 1:
            if str(environment.get('NCCL_P2P_DISABLE')) == '1':
                for rank in participants:
                    numa = devices[rank].get('numa_node')
                    if numa is None:
                        missing.append(f'rank:{rank}:host_numa')
                    else:
                        resources.add(f'host_memory:{node}:numa:{numa}')
            continue
        selection = environment.get('NCCL_IB_HCA')
        if not selection or ',' in selection or selection.startswith('^'):
            missing.append(f'node:{node}:observed_active_ib_ports')
            continue
        selected_name = selection.removeprefix('=').partition(':')[0]
        selected = [row for row in adapters if str(row.get('node_index')) == node
                    and row.get('name') == selected_name]
        if len(selected) != 1:
            missing.append(f'node:{node}:adapter:{selected_name}:pci_path')
            continue
        adapter = selected[0]
        local = pci_resources(node, adapter['pci_path'])
        resources.update(local)
        for port in adapter['ports']:
            resources.add(f'ib_port:{node}:{selected_name}:{port}')
        paths.append(dict(node=node, adapter=selected_name, pci_path=adapter['pci_path'],
            numa_node=adapter.get('numa_node'), resource_ids=local))
        missing.append(f'node:{node}:runtime_gdr_or_host_staging')
    return dict(resource_ids=sorted(resources), physical_paths=paths,
        missing_topology=sorted(set(missing)),
        path_source='recorded_sysfs_gpu_and_selected_nccl_adapter_paths',
        sharing_scope='collective_service_resource_envelope',
        assumptions=['PCIe和选定IB端口表示共享资源关系；内部交换网络路径需要独立记录',
            '独立服务时间保持NCCL实测值；资源共享模型单独接受并发测量验证'])
