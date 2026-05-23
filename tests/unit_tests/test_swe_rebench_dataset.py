# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.hf_datasets.swe_rebench import filter_and_deserialize


class TestSweRebenchDeserialize(unittest.TestCase):
    def test_filters_fields_and_parses_tool_arguments(self):
        row = {
            "trajectory_id": "t1",
            "trajectory": [
                {"role": "system", "content": "sys", "extra": 1},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "out",
                    "name": "bash",
                    "tool_call_id": "c1",
                    "noise": 2,
                },
            ],
        }
        out = filter_and_deserialize(row)
        self.assertEqual(out["trajectory_id"], "t1")
        traj = out["trajectory"]
        self.assertEqual(list(traj[0].keys()), ["role", "content"])
        self.assertEqual(traj[1]["tool_calls"][0]["function"]["arguments"], {"cmd": "ls"})
        self.assertEqual(
            list(traj[2].keys()),
            ["role", "content", "name", "tool_call_id"],
        )

    def test_unknown_role_raises(self):
        row = {
            "trajectory": [{"role": "narrator", "content": "x"}],
        }
        with self.assertRaises(ValueError):
            filter_and_deserialize(row)


if __name__ == "__main__":
    unittest.main()
