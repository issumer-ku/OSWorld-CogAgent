import unittest

from osworld_cogagent.runner import _parser


class RunnerCliTests(unittest.TestCase):
    def test_provider_neutral_arguments_are_available(self):
        args = _parser().parse_args(["--osworld-root", "/tmp/OSWorld"])
        self.assertIsNone(args.base_url)
        self.assertEqual(args.model, "cogagent-9b-20241220")
        self.assertEqual(args.provider_name, "docker")
        self.assertEqual(args.num_envs, 1)

    def test_runpod_or_lab_url_is_not_hardcoded(self):
        args = _parser().parse_args(
            [
                "--osworld-root",
                "/tmp/OSWorld",
                "--base_url",
                "http://lab.example:8000/v1",
            ]
        )
        self.assertEqual(args.base_url, "http://lab.example:8000/v1")


if __name__ == "__main__":
    unittest.main()
