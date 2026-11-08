from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone


COMMANDS = {
    "gpu_query": ("nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu", "--format=csv,nounits"),
    "compute_apps": ("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"),
    "gpu_topology": ("nvidia-smi", "topo", "-m"),
    "gpu_p2p": ("nvidia-smi", "topo", "-p2p", "n"),
    "numa": ("numactl", "--hardware"),
    "pci": ("lspci", "-tv"),
    "rdma_devices": ("ibdev2netdev",),
    "rdma_status": ("ibstat",),
    "rdma_info": ("ibv_devinfo",),
}


def _run(command: tuple[str, ...]) -> dict:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": list(command), "stdout": "", "stderr": "command not found", "returncode": None}
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "available": True,
        "command": list(command),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "hostname": platform.node(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "pod": os.environ.get("HOSTNAME", ""),
        "commands": {name: _run(command) for name, command in COMMANDS.items()},
    }
    import torch
    query = result['commands']['gpu_query']
    if query['returncode'] != 0:
        raise RuntimeError('实际GPU识别命令没有成功')
    devices = []
    for source in csv.DictReader(io.StringIO(query['stdout']), skipinitialspace=True):
        row = {name.strip(): value.strip() for name, value in source.items()}
        index = int(row['index'])
        devices.append(dict(index=index, uuid=row['uuid'], pci_bus_id=row['pci.bus_id'],
            name=row['name'], capacity_bytes=int(row['memory.total [MiB]']) * 2**20,
            peer_access={peer: torch.cuda.can_device_access_peer(index, peer)
                         for peer in range(torch.cuda.device_count()) if peer != index},
            source='actual_nvidia_smi_and_cuda_peer_access'))
        address = row['pci.bus_id'].lower()
        device_paths = list(Path('/sys/bus/pci/devices').glob('*:' + address.partition(':')[2]))
        if len(device_paths) != 1:
            raise ValueError('实际GPU PCI地址无法唯一对应sysfs设备')
        device_path = device_paths[0].resolve()
        devices[-1].update(pci_path=str(device_path), pci_ancestors=list(device_path.parts),
            numa_node=int((device_path / 'numa_node').read_text()))
    adapters = []
    for adapter in sorted(Path('/sys/class/infiniband').iterdir()):
        device = (adapter / 'device').resolve()
        numa = device / 'numa_node'
        physical_device = device / 'physfn'
        adapters.append(dict(name=adapter.name, pci_path=str(device),
            physical_device_path=str(physical_device.resolve()) if physical_device.exists() else str(device),
            numa_node=int(numa.read_text()) if numa.is_file() else None,
            ports=[port.name for port in sorted((adapter / 'ports').iterdir())]))
    result.update(devices=devices, adapters=adapters,
        software=dict(torch=torch.__version__, cuda=torch.version.cuda,
                      nccl=torch.cuda.nccl.version()),
        environment={key: value for key, value in os.environ.items()
                     if key.startswith(('CUDA_', 'NCCL_'))},
        topology_source='actual_hardware_queries_and_sysfs',
        effective_bandwidth_source='requires_actual_nccl_benchmark',
        gpu_direct_rdma='requires_observed_runtime_transport')
    result['gpu_nic_affinity'] = [dict(gpu_uuid=gpu['uuid'], adapter=adapter['name'],
        same_numa_node=gpu['numa_node'] >= 0 and gpu['numa_node'] == adapter['numa_node'],
        source='actual_sysfs_numa_relationship') for gpu in devices for adapter in adapters]
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(output)


if __name__ == "__main__":
    main()
