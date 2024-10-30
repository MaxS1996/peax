from typing import List, Set, Dict

from tensorflow import keras

from ..components import resource as res


class Processor:
    """Representation of a target processor.
    Does not equal a typical CPU, as it contains memory information as well and can also represent GPUs and ASIPs.
    The user can specify the tpyical MAC throughput, the memory size and add layers that are not supported as well as a dictionary of functions to describe unsupported or supported layer configurations
    """

    name: str
    """descriptive name of the CPU, used for printing summaries and error messages"""

    _MACs: int
    """number of MACs that this processor can perform per second"""

    _MEM: int
    "size of the usuable memory in byte"

    _unsupported_layers: Set[keras.layers.Layer]
    """Keras layers that cannot be executed by this processor"""

    _ruleset: Dict[keras.layers.Layer, callable]
    """A dictionary, Keras layers as keys, functions as values.
    The functions decide based on the Keras layer object, if it can be executed by the processor.
    This allows to evaluate the decision based on the hyperparameter configuration of the specific instance"""

    estimation_functions: Dict[keras.layers.Layer, callable]
    """A Dict of functions that estimate the required MAC for each layer type, if executed by this processor.
    This was added to account for the different execution strategies of different processor types."""

    def __init__(
        self,
        name: str,
        compute_macs: int = 5 * (10**9),
        memory_size: int = 4294967296,
        limitations: Set[keras.layers.Layer] = None,
        ruleset: Dict[keras.layers.Layer, callable] = None,
        estimation_functions: Dict[keras.layers.Layer, callable] = None,
    ) -> None:
        self.name = name
        self._MACs = compute_macs
        self._MEM = memory_size
        if limitations is None:
            self._unsupported_layers = []
        else:
            self._unsupported_layers = limitations

        if ruleset is None:
            self._ruleset = dict()
        else:
            self._ruleset = ruleset

        if estimation_functions is not None:
            self.estimation_functions = estimation_functions
        else:
            self.estimation_functions = res._default_mac_estimators

        pass

    def check(
        self,
        layer: keras.layers.Layer,
        estimation_functions: Dict[keras.layers.Layer, callable] = None,
        dtypes: List[str] = None,
    ):
        """Checks, if a layer can be executed by the processor, estimates how long it would take, and estiamtes the IFM and OFM memory footprint of the layer for different datatypes

        Args:
            layer (keras.layers.Layer): The layer that, whose compatibility with the processor should be checked
            estimation_functions (Dict[keras.layers.Layer, callable], optional): optional dictionary of special functions to estimate the layers MAC footprint, can be used for specific ASIPs or Libraries. Defaults to None.
            dtypes (List[str], optional): Dtypes that should be evaluated. Defaults to None.

        Returns:
            supported (bool) : if the layer can be executed based on the unsupported layers, and ruleset
            delay (float) : estimated processing time
            weight_alloc (float) : how much memory is allocated by the layer weights
            TODO: IFM alloc
        """
        """check if layer can be executed by processor and estimate how long it will take and if input and output fit into memory"""

        if dtypes is None or len(dtypes) == 0:
            dtypes = ["float32"]

        supported = True
        weight_alloc = dict()

        if estimation_functions is None:
            estimation_functions = self.estimation_functions

        # check support

        if type(layer) in self._unsupported_layers:
            supported = False

        if type(layer) in self._ruleset.keys():
            supported = self._ruleset[type(layer)](layer)

        if supported == False:
            for dtype in dtypes:
                weight_alloc[dtype] = float("inf")
            return (supported, float("inf"), weight_alloc, float("inf"))

        # estimate processing speed
        layer_cost = res.get_layer_macs(
            layer=layer, estimation_functions=estimation_functions
        )
        delay = layer_cost / self._MACs

        # estimate memory footprint
        weight_size = res.get_layer_weight_size(layer=layer, datatypes=dtypes)

        for dtype in dtypes:
            weight_alloc[dtype] = weight_size[dtype] / self._MEM

        # TODO: IFM allocation

        return (supported, delay, weight_alloc, layer_cost)

    def __str__(self) -> str:
        return f"Processor {self.name} with {self._MACs} MAC/s and {self._MEM} bytes memory"
    
    def toDict(self) -> Dict:
        data = {}
        data["name"] = self.name
        data["MAC/sec"] = self._MACs
        data["MEM"] = self._MEM

        return data
