import unittest

from predictor.physical_paths import communication_resources


class PhysicalPathTests(unittest.TestCase):
    def topology(self):
        return dict(devices=[
            dict(physical_rank=0, node_index=0, numa_node=0,
                pci_path='/sys/devices/pci0000:10/0000:10:01.0/0000:11:00.0'),
            dict(physical_rank=1, node_index=0, numa_node=1,
                pci_path='/sys/devices/pci0000:20/0000:20:01.0/0000:21:00.0'),
            dict(physical_rank=2, node_index=1, numa_node=0,
                pci_path='/sys/devices/pci0000:30/0000:30:01.0/0000:31:00.0')],
            paths=[
                dict(node_index=0, name='mlx5_9', numa_node=1, ports=['1'],
                    pci_path='/sys/devices/pci0000:40/0000:40:01.0/0000:41:00.0'),
                dict(node_index=1, name='mlx5_9', numa_node=0, ports=['1'],
                    pci_path='/sys/devices/pci0000:50/0000:50:01.0/0000:51:00.0')],
            environment_per_node={'0': {'NCCL_P2P_DISABLE': '1', 'NCCL_IB_HCA': 'mlx5_9'},
                                  '1': {'NCCL_P2P_DISABLE': '1', 'NCCL_IB_HCA': 'mlx5_9'}})

    def test_intra_node_pcie_and_host_staging_resources(self):
        result = communication_resources(self.topology(), [0, 1])
        self.assertIn('host_memory:0:numa:0', result['resource_ids'])
        self.assertIn('host_memory:0:numa:1', result['resource_ids'])
        self.assertTrue(any(value.startswith('pcie:0:') for value in result['resource_ids']))

    def test_cross_node_includes_gpu_pcie_and_selected_ib_ports(self):
        result = communication_resources(self.topology(), [0, 2])
        self.assertIn('ib_port:0:mlx5_9:1', result['resource_ids'])
        self.assertIn('ib_port:1:mlx5_9:1', result['resource_ids'])
        self.assertTrue(any(value.startswith('pcie:0:') for value in result['resource_ids']))
        self.assertTrue(any(value.startswith('pcie:1:') for value in result['resource_ids']))
        self.assertEqual(result['missing_topology'], [
            'node:0:runtime_gdr_or_host_staging', 'node:1:runtime_gdr_or_host_staging'])


if __name__ == '__main__':
    unittest.main()
