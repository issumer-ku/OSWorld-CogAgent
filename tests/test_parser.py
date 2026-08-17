import unittest

from osworld_cogagent.parser import parse_cogagent_response


class CogAgentActionParserTest(unittest.TestCase):
    def parse(self, operation: str):
        response = f"Status: ready\nAction: execute test\nGrounded Operation: {operation}"
        return parse_cogagent_response(response, 1920, 1080)

    def test_click_uses_normalized_box_center(self):
        parsed = self.parse("CLICK(box=[[100,200,300,400]], element_info='Save')")
        self.assertEqual(parsed.pyautogui_code, ["pyautogui.click(384, 324)"])

    def test_click_accepts_zero_padded_coordinate_literals(self):
        parsed = self.parse("CLICK(box=[[143,057,170,082]], element_info='Colors')")
        self.assertEqual(parsed.pyautogui_code, ["pyautogui.click(300, 75)"])

    def test_type_accepts_zero_padded_coordinates_without_changing_text(self):
        parsed = self.parse(
            "TYPE(box=[[062,083,931,111]], text='007', element_info='search')"
        )
        self.assertEqual(parsed.pyautogui_code[0], "pyautogui.click(953, 105)")
        self.assertEqual(
            parsed.pyautogui_code[-1],
            'pyautogui.write("007", interval=0.01)',
        )

    def test_multiline_type_preserves_text(self):
        parsed = self.parse(
            "TYPE(\nbox=[[400,400,600,500]],\ntext='hello, world',\n"
            "element_info='Editor'\n)"
        )
        self.assertEqual(parsed.pyautogui_code[0], "pyautogui.click(960, 486)")
        self.assertEqual(
            parsed.pyautogui_code[-1],
            'pyautogui.write("hello, world", interval=0.01)',
        )

    def test_scroll_and_key_press(self):
        self.assertEqual(
            self.parse("SCROLL_DOWN(box=[[0,0,999,999]], step_count=5)").pyautogui_code,
            ["pyautogui.scroll(-5)"],
        )
        self.assertEqual(
            self.parse("KEY_PRESS(key='Return')").pyautogui_code,
            ['pyautogui.press("enter")'],
        )

    def test_gesture_converts_nested_key_actions(self):
        parsed = self.parse(
            "GESTURE(actions=[KEY_DOWN(key='Lcontrol'), "
            "KEY_PRESS(key='A'), KEY_UP(key='Lcontrol')])"
        )
        self.assertEqual(
            parsed.pyautogui_code,
            [
                'pyautogui.keyDown("ctrl")',
                'pyautogui.press("a")',
                'pyautogui.keyUp("ctrl")',
            ],
        )

    def test_launch_url_and_end(self):
        parsed = self.parse("LAUNCH(app='None', url='https://example.com')")
        self.assertEqual(parsed.pyautogui_code[0], 'pyautogui.hotkey("ctrl", "l")')
        self.assertEqual(self.parse("END").pyautogui_code, ["DONE"])

    def test_rejects_code_execution_and_unsupported_operations(self):
        with self.assertRaises(ValueError):
            self.parse("__import__('os').system('id')")
        with self.assertRaises(ValueError):
            self.parse("QUOTE_TEXT(box=[[0,0,999,999]], output='x')")

    def test_malformed_operation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.parse("CLICK(box=[[100,200,300]])")


if __name__ == "__main__":
    unittest.main()
