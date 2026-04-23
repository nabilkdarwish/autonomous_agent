"""
Unit tests — run with: python -m pytest tests/ -v
"""

import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.builtin_tools import (
    CalculatorTool, FileReaderTool, FileWriterTool,
    CodeExecutorTool, MemoryStoreTool
)
from memory.memory_manager import ShortTermMemory, LongTermMemory, MemoryManager
from core.config import AgentConfig
from core.models import MessageRole, TaskStatus, Plan, Step


# ─── Calculator Tests ────────────────────────────────────────────────────────

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = CalculatorTool()

    def test_basic_addition(self):
        result = self.calc.run(expression="2 + 2")
        self.assertTrue(result.success)
        self.assertIn("4", result.output)

    def test_power(self):
        result = self.calc.run(expression="2 ** 10")
        self.assertTrue(result.success)
        self.assertIn("1024", result.output)

    def test_sqrt(self):
        result = self.calc.run(expression="sqrt(144)")
        self.assertTrue(result.success)
        self.assertIn("12", result.output)

    def test_unsafe_expression(self):
        result = self.calc.run(expression="__import__('os')")
        self.assertFalse(result.success)

    def test_invalid_expression(self):
        result = self.calc.run(expression="abc + xyz")
        self.assertFalse(result.success)


# ─── File Tools Tests ────────────────────────────────────────────────────────

class TestFileTools(unittest.TestCase):
    def setUp(self):
        self.reader = FileReaderTool()
        self.writer = FileWriterTool()
        self.tmp = tempfile.mkdtemp()

    def test_write_and_read(self):
        path = os.path.join(self.tmp, "test.txt")
        write_result = self.writer.run(path=path, content="Hello, Agent!")
        self.assertTrue(write_result.success)

        read_result = self.reader.run(path=path)
        self.assertTrue(read_result.success)
        self.assertIn("Hello, Agent!", read_result.output)

    def test_read_nonexistent(self):
        result = self.reader.run(path="/nonexistent/path/file.txt")
        self.assertFalse(result.success)

    def test_append_mode(self):
        path = os.path.join(self.tmp, "append.txt")
        self.writer.run(path=path, content="Line 1\n")
        self.writer.run(path=path, content="Line 2\n", mode="a")
        read = self.reader.run(path=path)
        self.assertIn("Line 1", read.output)
        self.assertIn("Line 2", read.output)


# ─── Code Executor Tests ─────────────────────────────────────────────────────

class TestCodeExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = CodeExecutorTool()

    def test_simple_print(self):
        result = self.executor.run(code="print('hello world')")
        self.assertTrue(result.success)
        self.assertIn("hello world", result.output)

    def test_math_computation(self):
        result = self.executor.run(code="print(sum(range(100)))")
        self.assertTrue(result.success)
        self.assertIn("4950", result.output)

    def test_blocked_import(self):
        result = self.executor.run(code="import os; print(os.getcwd())")
        self.assertFalse(result.success)

    def test_syntax_error(self):
        result = self.executor.run(code="def foo(: pass")
        self.assertFalse(result.success)


# ─── Memory Tests ────────────────────────────────────────────────────────────

class TestShortTermMemory(unittest.TestCase):
    def test_add_and_retrieve(self):
        stm = ShortTermMemory(max_messages=5)
        stm.add(MessageRole.USER, "Hello")
        stm.add(MessageRole.ASSISTANT, "Hi there")
        msgs = stm.get_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].content, "Hello")

    def test_overflow_trimming(self):
        stm = ShortTermMemory(max_messages=3)
        for i in range(5):
            stm.add(MessageRole.USER, f"Message {i}")
        msgs = stm.get_messages()
        self.assertEqual(len(msgs), 3)

    def test_clear(self):
        stm = ShortTermMemory()
        stm.add(MessageRole.USER, "Test")
        stm.clear()
        self.assertEqual(len(stm), 0)


class TestLongTermMemory(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = AgentConfig(vector_store_path=self.tmp_dir)
        self.ltm = LongTermMemory(self.config)

    def test_store_and_retrieve(self):
        self.ltm.store("Python is a great programming language")
        self.ltm.store("Machine learning requires lots of data")
        results = self.ltm.search("Python programming")
        self.assertGreater(len(results), 0)

    def test_empty_search(self):
        results = self.ltm.search("some query on empty memory")
        self.assertEqual(results, [])

    def test_persistence(self):
        self.ltm.store("Remember this fact")
        # Re-init with same path
        ltm2 = LongTermMemory(self.config)
        self.assertGreater(len(ltm2), 0)


# ─── Plan / Step Model Tests ─────────────────────────────────────────────────

class TestModels(unittest.TestCase):
    def test_plan_creation(self):
        plan = Plan(goal="Test goal")
        plan.add_step("Step one", tool_name="calculator", tool_args={"expression": "1+1"})
        plan.add_step("Step two")
        self.assertEqual(len(plan.steps), 2)
        self.assertFalse(plan.is_complete())

    def test_step_lifecycle(self):
        step = Step(step_id=1, description="Test step")
        self.assertEqual(step.status, TaskStatus.PENDING)
        step.mark_started()
        self.assertEqual(step.status, TaskStatus.IN_PROGRESS)
        step.mark_completed("done")
        self.assertEqual(step.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(step.duration())

    def test_plan_completion(self):
        plan = Plan(goal="goal")
        s1 = plan.add_step("Step 1")
        s2 = plan.add_step("Step 2")
        s1.mark_started(); s1.mark_completed("ok")
        s2.mark_started(); s2.mark_completed("ok")
        self.assertTrue(plan.is_complete())
        self.assertFalse(plan.has_failures())


if __name__ == "__main__":
    unittest.main(verbosity=2)
