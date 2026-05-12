import os
import unittest
from unittest import mock

from scripts.frameworks import opvec_verl_external_lib as hook


class VerlExternalLibHookTest(unittest.TestCase):
    def test_parse_max_gated_modules(self):
        self.assertIsNone(hook._parse_max_gated_modules(None))
        self.assertIsNone(hook._parse_max_gated_modules(""))
        self.assertIsNone(hook._parse_max_gated_modules("0"))
        self.assertEqual(hook._parse_max_gated_modules("2"), 2)

    def test_env_truthy(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(hook._env_truthy("MISSING", default=False))
            self.assertTrue(hook._env_truthy("MISSING", default=True))
        with mock.patch.dict(os.environ, {"FLAG": "yes"}, clear=True):
            self.assertTrue(hook._env_truthy("FLAG", default=False))
        with mock.patch.dict(os.environ, {"FLAG": "0"}, clear=True):
            self.assertFalse(hook._env_truthy("FLAG", default=True))

    def test_normalize_gate_parameterization(self):
        self.assertEqual(hook._normalize_gate_parameterization("global_param"), "global-parameter")
        self.assertEqual(hook._normalize_gate_parameterization("param"), "parameter")
        self.assertEqual(hook._normalize_gate_parameterization("global"), "global")


if __name__ == "__main__":
    unittest.main()
