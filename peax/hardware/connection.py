from typing import Dict
from . import processor as prc


class Connection:
    """class to describe the connections between processors, either on the same MPSoC or across a network"""

    def __init__(
        self, start: prc.Processor, end: prc.Processor, bandwidth: float
    ) -> None:
        self.start_point = start
        self.end_point = end
        self.bandwidth = bandwidth
        pass

    def calculate_latency(self, IFM_size: int) -> float:
        """calculates the transmission latency for the given IFM size (in byte) for this connection

        Args:
            IFM_size (int): size of tensor in byte

        Returns:
            float: estimated best-case latency
        """
        size = float(IFM_size)
        latency = size / self.bandwidth

        return latency
    
    def toDict(self) -> Dict:
        data = {}
        data["start"] = self.start_point.name
        data["end"] = self.end_point.name
        data["bandwidth"] = self.bandwidth

        return data
