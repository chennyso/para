import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    allowed = {'chenny-g5-8gpu-ib-20261001': 'g5', 'chenny-g6-8gpu-ib-20261010': 'g6'}
    if args.pod not in allowed:
        raise ValueError('恢复范围仅包含本次ParaMorph实验使用的两个Pod')
    command = ['kubectl', '--context', 'prd', '-n', 'default']
    pod = json.loads(subprocess.run(command + ['get', 'pod', args.pod, '-o', 'json'],
        check=True, capture_output=True, text=True).stdout)
    if (pod['status']['phase'] != 'Failed' or pod['spec']['nodeName'] != allowed[args.pod]
            or pod['metadata'].get('ownerReferences')):
        raise ValueError('只允许恢复已经终止的独立训练Pod')
    annotations = {key: value for key, value in pod['metadata'].get('annotations', {}).items()
        if key not in ('k8s.v1.cni.cncf.io/network-status', 'kubectl.kubernetes.io/last-applied-configuration')
        and not key.startswith('cni.projectcalico.org/')}
    manifest = dict(apiVersion='v1', kind='Pod', metadata=dict(name=args.pod,
        namespace='default', labels=pod['metadata'].get('labels', {}), annotations=annotations),
        spec=pod['spec'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run(command + ['delete', 'pod', args.pod, '--wait=true'], check=True)
    subprocess.run(command + ['create', '-f', str(args.output)], check=True)
    subprocess.run(command + ['wait', '--for=condition=Ready', f'pod/{args.pod}', '--timeout=180s'], check=True)


if __name__ == '__main__':
    main()
