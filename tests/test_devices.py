import unittest

from opvec.modeling.devices import model_load_device_kwargs, parse_max_memory


class DevicesTest(unittest.TestCase):
    def test_parse_max_memory_accepts_gpu_indices_and_cpu(self):
        self.assertEqual(parse_max_memory(["0=70GiB", "1=68GiB", "cpu=120GiB"]), {0: "70GiB", 1: "68GiB", "cpu": "120GiB"})

    def test_model_load_device_kwargs_only_sets_sharding_when_requested(self):
        self.assertEqual(model_load_device_kwargs(device_map=None, max_memory=["0=70GiB"]), {})
        self.assertEqual(
            model_load_device_kwargs(device_map="auto", max_memory=["0=70GiB"]),
            {"device_map": "auto", "max_memory": {0: "70GiB"}},
        )


if __name__ == "__main__":
    unittest.main()
