import tensorflow as tf
import numpy as np

import pytest


from peax.hardware import processor as pr
from peax.hardware import connection as con


class TestConnection:
    @pytest.fixture
    def connection(self):
        start = pr.Processor("start")
        end = pr.Processor("end")
        bandwidth = 2.0
        return con.Connection(start, end, bandwidth)

    def test_connection_properties(self, connection):
        assert connection.start_point.name == "start"
        assert connection.end_point.name == "end"
        assert connection.bandwidth == 2.0

    def test_connection_calculate_latency(self, connection):
        assert connection.calculate_latency(1024) == 512.0
        assert connection.calculate_latency(2048) == 1024.0
        assert connection.calculate_latency(4096) == 2048.0
