from dataclasses import dataclass

from megatron.core.frontier.boundary.payload import derive_payload


@dataclass(frozen=True)
class PipelineCut:
    cut_id: str
    producer_regions: tuple[str, ...]
    consumer_regions: tuple[str, ...]
    source_contracts: tuple
    destination_contracts: tuple

    def __post_init__(self):
        if not self.producer_regions or not self.consumer_regions or not self.source_contracts:
            raise ValueError('pipeline cuts require producer, consumer and payload contracts')
        if self.source_contracts != self.destination_contracts:
            raise ValueError('semantic cuts require an explicit lowering for contract conversion')
        derive_payload(self)


def enumerate_cuts(graph):
    regions = {region.region_id: region for region in graph.regions}
    cuts = []
    for source, destination in graph.edges:
        producer, consumer = regions[source], regions[destination]
        if not producer.transportable_output:
            continue
        if producer.output_contracts != consumer.input_contracts:
            continue
        cuts.append(PipelineCut(f'{source}->{destination}', (source,), (destination,),
                                producer.output_contracts, consumer.input_contracts))
    return tuple(cuts)
