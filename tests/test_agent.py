import io
import unittest
from unittest.mock import patch

from PIL import Image

from osworld_cogagent import CogAgent


def screenshot_bytes() -> bytes:
    image = Image.new("RGB", (1000, 500), color=(240, 240, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class StubCogAgent(CogAgent):
    def __init__(self, responses, **kwargs):
        super().__init__(**kwargs)
        self.responses = list(responses)
        self.calls = []

    def _call(self, messages, *, temperature=None):
        self.calls.append((messages, temperature))
        if not self.responses:
            raise AssertionError("Unexpected model call")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def response(operation: str, action: str = "Perform the next operation") -> str:
    return (
        "Status: Inspecting the visible desktop.\n"
        f"Action: {action}\n"
        f"Grounded Operation: {operation}"
    )


class CogAgentTests(unittest.TestCase):
    def setUp(self):
        self.obs = {"screenshot": screenshot_bytes()}

    def test_native_prompt_and_history(self):
        agent = StubCogAgent(
            [
                response("CLICK(box=[[100,100,300,300]], element_info='Button')"),
                response("KEY_PRESS(key='Return')"),
            ],
            completion_verification=False,
        )
        _, first = agent.predict("Open the item", self.obs)
        _, second = agent.predict("Open the item", self.obs)
        self.assertEqual(first, ["pyautogui.click(200, 100)"])
        self.assertEqual(second, ['pyautogui.press("enter")'])
        prompt = agent.calls[1][0][0]["content"][0]["text"]
        self.assertIn("Task: Open the item", prompt)
        self.assertIn("CLICK(box=[[100,100,300,300]]", prompt)
        self.assertIn("(Platform: WIN)", prompt)
        self.assertTrue(
            agent.calls[1][0][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )

    def test_invalid_format_is_reprompted(self):
        agent = StubCogAgent(
            ["No structured action", response("CLICK(box=[[0,0,100,100]])")],
            completion_verification=False,
        )
        with patch.dict("os.environ", {"OSWORLD_COGAGENT_FORMAT_RETRIES": "1"}):
            _, code = agent.predict("Click it", self.obs)
        self.assertEqual(code, ["pyautogui.click(50, 25)"])
        self.assertIn("could not be executed", agent.calls[1][0][0]["content"][0]["text"])

    def test_format_failures_are_bounded_across_observations(self):
        agent = StubCogAgent(
            ["bad one", "bad two"],
            completion_verification=False,
            format_failure_limit=2,
        )
        with patch.dict("os.environ", {"OSWORLD_COGAGENT_FORMAT_RETRIES": "0"}):
            _, first = agent.predict("Click it", self.obs)
            _, second = agent.predict("Click it", self.obs)
        self.assertEqual(first, ["WAIT"])
        self.assertTrue(second[0].startswith("MODEL_ERROR:"))

    def test_end_requires_completion_audit(self):
        agent = StubCogAgent(
            [
                response("END", "The task is complete"),
                response("CLICK(box=[[800,800,900,900]])", "One checkbox remains"),
            ],
            completion_verification=True,
        )
        _, code = agent.predict("Select every checkbox", self.obs)
        self.assertEqual(code, ["pyautogui.click(850, 425)"])
        self.assertIn("COMPLETION AUDIT", agent.calls[1][0][0]["content"][0]["text"])
        self.assertEqual(agent.calls[1][1], 0.0)

    def test_repeated_action_gets_corrective_retry(self):
        repeated = response("CLICK(box=[[100,100,300,300]])")
        agent = StubCogAgent(
            [repeated, repeated, repeated, repeated, response("KEY_PRESS(key='Return')")],
            completion_verification=False,
            loop_action_limit=3,
        )
        for _ in range(3):
            agent.predict("Continue", self.obs)
        _, code = agent.predict("Continue", self.obs)
        self.assertEqual(code, ['pyautogui.press("enter")'])
        self.assertIn("screen remained unchanged", agent.calls[4][0][0]["content"][0]["text"])

    def test_reset_clears_task_state(self):
        agent = StubCogAgent([], completion_verification=False)
        agent.history_steps.append("CLICK(box=[[0,0,1,1]])")
        agent.consecutive_format_failures = 2
        agent.reset()
        self.assertEqual(agent.history_steps, [])
        self.assertEqual(agent.consecutive_format_failures, 0)


if __name__ == "__main__":
    unittest.main()
