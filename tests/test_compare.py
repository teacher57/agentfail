import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from agentfail.cli import main
from agentfail.compare import compare, gate
from agentfail.loader import load_tau2
from agentfail.stats import bootstrap_ci, paired_permutation_p, pass_hat_k

from builders import run_file


class StatsTest(unittest.TestCase):
    def test_pass_hat_k(self):
        succ = {"a": [True, True], "b": [True, False], "c": [False, False]}
        self.assertAlmostEqual(pass_hat_k(succ, 1), 0.5)
        self.assertAlmostEqual(pass_hat_k(succ, 2), 1 / 3)
        self.assertIsNone(pass_hat_k(succ, 3))

    def test_permutation_exact(self):
        self.assertEqual(paired_permutation_p([0, 0, 0]), 1.0)
        self.assertAlmostEqual(paired_permutation_p([1] * 5), 2 / 32)
        self.assertAlmostEqual(paired_permutation_p([1, -1]), 1.0)

    def test_permutation_monte_carlo_is_deterministic(self):
        d = [1] * 30 + [-1] * 5
        p = paired_permutation_p(d)
        self.assertLess(p, 0.001)
        self.assertEqual(p, paired_permutation_p(d))

    def test_bootstrap_ci(self):
        self.assertEqual(bootstrap_ci([0.5] * 10), (0.5, 0.5))
        lo, hi = bootstrap_ci([0, 1] * 20)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.base = run_file(d / "base.json", n_tasks=40, n_pass=30, trials=2)
        self.worse = run_file(d / "worse.json", n_tasks=40, n_pass=15, trials=2)
        self.one_trial = run_file(d / "one.json", n_tasks=40, n_pass=30, trials=1)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main(list(map(str, args)))
        return code, out.getvalue()

    def test_self_compare(self):
        cmp = compare(load_tau2(self.base), load_tau2(self.base))
        self.assertEqual(cmp.pass_rate.delta, 0)
        self.assertEqual(cmp.pass_rate.p, 1.0)
        self.assertEqual(gate(cmp, 0.0, 0.05, {}), [])

    def test_regression_detected(self):
        cmp = compare(load_tau2(self.base), load_tau2(self.worse))
        self.assertAlmostEqual(cmp.pass_rate.delta, -15 / 40)
        self.assertLess(cmp.pass_rate.p, 0.001)
        self.assertEqual(len(cmp.flips["broken"]), 15)
        self.assertEqual([(k, round(b, 3), round(c, 3)) for k, b, c in cmp.pass_k],
                         [(1, 0.75, 0.375), (2, 0.75, 0.375)])
        wrong = next(d for d in cmp.categories if d.name == "wrong_args")
        self.assertAlmostEqual(wrong.delta, 15 / 40)

    def test_mixed_trial_counts(self):
        cmp = compare(load_tau2(self.base), load_tau2(self.one_trial))
        self.assertEqual(cmp.pass_rate.delta, 0)
        self.assertIsNone(cmp.pass_k[1][2])  # pass^2 undefined for a 1-trial run

    def test_cli_exit_codes(self):
        self.assertEqual(self.run_cli("compare", self.base, self.base)[0], 0)
        code, out = self.run_cli("compare", self.base, self.worse)
        self.assertEqual(code, 1)
        self.assertIn("GATE: FAIL", out)
        # within tolerance → pass
        self.assertEqual(self.run_cli("compare", self.base, self.worse, "--max-drop", "0.5")[0], 0)
        # category limit
        code, out = self.run_cli("compare", self.base, self.worse, "--max-drop", "1",
                                 "--max-increase", "wrong_args=0.1")
        self.assertEqual(code, 1)
        self.assertIn("wrong_args rose", out)

    def test_cli_json(self):
        out = Path(self.tmp.name) / "cmp.json"
        self.run_cli("compare", self.base, self.worse, "--json", out)
        data = json.loads(out.read_text())
        self.assertFalse(data["gate"]["passed"])
        self.assertEqual(data["shared_tasks"], 40)

    def test_cli_rejects_unknown_category(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
            main(["compare", str(self.base), str(self.base), "--max-increase", "bogus=1"])
        self.assertEqual(e.exception.code, 2)

    def test_cli_classify(self):
        code, out = self.run_cli("classify", self.worse, "--examples", "1")
        self.assertEqual(code, 0)
        self.assertIn("pass=30 (37.5%)", out)
        self.assertIn("[wrong_args]", out)


if __name__ == "__main__":
    unittest.main()
