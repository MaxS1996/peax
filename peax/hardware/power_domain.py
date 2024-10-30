from typing import List, Set
from . import processor as proc


class PowerDomain:
    """describes a power domain of the target platform"""

    def __init__(
        self, processors: Set[proc.Processor], startup_penalty: float = 1.0
    ) -> None:
        self.processors = processors
        self.startup_penalty = startup_penalty

        pass

    def __contains__(self, item: proc.Processor) -> bool:
        """check if processor is part of this power domain

        Args:
            item (proc.Processor): the processor that might be part of this power domain

        Returns:
            bool: True, if processor belongs to this power domain, False otherwise
        """
        return item in self.processors
