import json
import tempfile
import unittest
from pathlib import Path

from agentfail.classify import classify
from agentfail.loader import load_tau2

from builders import (CANCEL, auth, bad_messages, call, flat, good_messages, results, say, sim,
                      task, user)


def load(sims, actions=(CANCEL,), script=None):
    """Write a one-task results file and return its loaded sims."""
    tasks = [task(s["task_id"], actions, script) for s in sims]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.json"
        p.write_text(json.dumps(results(tasks, sims)))
        return load_tau2(p).sims


def classify_one(messages, actions=(CANCEL,), script=None, **kw):
    return classify(load([sim(0, messages, **kw)], actions, script)[0])


FAIL = dict(reward=0.0, db_match=False)


class LoaderTest(unittest.TestCase):
    def test_joins_tool_results_and_types(self):
        s = load([sim(0, flat(auth(), *call("get_order_details", error=True, order_id="#W9")))])[0]
        calls = s.calls()
        self.assertEqual([c.name for c in calls], ["find_user_id_by_email", "get_order_details"])
        self.assertEqual([c.error for c in calls], [False, True])
        self.assertEqual(s.tool_type("cancel_pending_order"), "write")
        self.assertEqual(s.tool_type("exchange_delivered_order_items"), "write")  # prefix fallback
        self.assertEqual(s.expected[0].name, "cancel_pending_order")

    def test_string_arguments_are_parsed(self):
        msgs = flat(auth())
        msgs[1]["tool_calls"][0]["arguments"] = '{"email": "ann@example.com"}'
        s = load([sim(0, msgs)])[0]
        self.assertEqual(s.calls()[0].args, {"email": "ann@example.com"})


class ClassifyTest(unittest.TestCase):
    def test_pass_has_no_primary_or_findings(self):
        r = classify_one(good_messages())
        self.assertIsNone(r.primary)
        self.assertEqual(r.labels, set())

    def test_wrong_args(self):
        r = classify_one(bad_messages(), **FAIL)
        self.assertEqual(r.primary, "wrong_args")
        self.assertIn("order_id", next(f.detail for f in r.findings if f.label == "wrong_args"))

    def test_missing_action(self):
        r = classify_one(flat(auth(), user("cancel W1"), say("Sorry, I can't.")), **FAIL)
        self.assertEqual(r.primary, "missing_action")

    def test_extra_write(self):
        msgs = flat(good_messages(), user("yes also update address to 1 Elm St"),
                    call("modify_user_address", user_id="ann_1", address1="1 Elm St"))
        r = classify_one(msgs, **FAIL)
        self.assertEqual(r.primary, "extra_write")

    def test_errored_extra_write_is_not_extra(self):
        msgs = flat(good_messages(), user("yes cancel W2 too"),
                    call("cancel_pending_order", error=True, order_id="#W2", reason="no longer needed"))
        self.assertNotIn("extra_write", classify_one(msgs, **FAIL).labels)

    def test_expected_write_that_errors_still_matches(self):
        msgs = flat(auth(), user("cancel W1"), user("yes"),
                    call("cancel_pending_order", error=True, order_id="#W1", reason="no longer needed"))
        self.assertNotIn("missing_action", classify_one(msgs).labels)

    def test_list_args_compared_as_multiset(self):
        action = ("cancel_pending_order", {"order_id": "#W1", "item_ids": ["a1", "b2"]})
        msgs = flat(auth(), user("W1 items a1 b2"), user("yes"),
                    call("cancel_pending_order", order_id="#W1", item_ids=["b2", "a1"]))
        self.assertEqual(classify_one(msgs, actions=[action]).labels, set())

    def test_compare_args_limits_keys(self):
        tasks = [{"id": "0", "evaluation_criteria": {"actions": [
            {"name": "cancel_pending_order", "arguments": {"order_id": "#W1", "reason": "x"},
             "compare_args": ["order_id"]}]}}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            p.write_text(json.dumps(results(tasks, [sim(0, good_messages())])))
            self.assertNotIn("wrong_args", classify(load_tau2(p).sims[0]).labels)

    def test_timeout_wins(self):
        r = classify_one(bad_messages(), termination="timeout", reward=0.0, db_match=None)
        self.assertEqual(r.primary, "timeout")

    def test_fabricated_arg_in_write_is_primary(self):
        msgs = flat(auth(), user("cancel my order"), user("yes"),
                    call("cancel_pending_order", order_id="#W1", reason="no longer needed"))
        r = classify_one(msgs, **FAIL)
        self.assertEqual(r.primary, "fabricated_args")

    def test_fabricated_arg_in_read_is_evidence_only(self):
        msgs = flat(call("find_user_id_by_email", error=True, email="user@example.com"),
                    bad_messages())
        r = classify_one(msgs, **FAIL)
        self.assertIn("fabricated_args", r.labels)
        self.assertEqual(r.primary, "wrong_args")

    def test_premature_action_before_auth(self):
        msgs = flat(user("cancel W1"), user("yes"),
                    call("cancel_pending_order", order_id="#W1", reason="no longer needed"))
        r = classify_one(msgs, **FAIL)
        self.assertEqual(r.primary, "premature_action")

    def test_unconfirmed_write_is_secondary(self):
        msgs = flat(auth(), user("cancel W1"),
                    call("cancel_pending_order", order_id="#W1", reason="no longer needed"))
        r = classify_one(msgs)
        self.assertIn("unconfirmed_write", r.labels)
        self.assertIsNone(r.primary)

    def test_db_mismatch_only(self):
        self.assertEqual(classify_one(good_messages(), **FAIL).primary, "db_mismatch_only")

    def test_unneeded_transfer(self):
        msgs = flat(auth(), user("cancel W1"), *call("transfer_to_human_agents", summary="cancel"))
        r = classify_one(msgs, **FAIL)
        self.assertEqual(r.primary, "unneeded_transfer")
        self.assertEqual(r.fault, "agent")

    def test_expected_transfer_is_fine(self):
        msgs = flat(auth(), user("I want a human"), *call("transfer_to_human_agents", summary="x"))
        r = classify_one(msgs, actions=[("transfer_to_human_agents", {"summary": "x"})])
        self.assertNotIn("unneeded_transfer", r.labels)

    def test_transfer_after_all_writes_is_not_primary(self):
        msgs = flat(good_messages(), *call("transfer_to_human_agents", summary="x"))
        r = classify_one(msgs, reward=0.0, db_match=True, nl=["Agent should say $5"])
        self.assertIn("unneeded_transfer", r.labels)
        self.assertEqual(r.primary, "nl_assertion_fail")

    def test_user_invented_info_blocking_auth_is_user_fault(self):
        script = "You are Ann Lee. Your email is ann@example.com."
        msgs = flat(user("I'm Ann Lee, zip 94109"),
                    call("find_user_id_by_name_zip", error=True, result="Error: User not found",
                         first_name="Ann", last_name="Lee", zip="94109"),
                    say("I couldn't find you."))
        r = classify_one(msgs, script=script, **FAIL)
        self.assertEqual(r.primary, "user_invented_info")
        self.assertEqual(r.fault, "user")
        self.assertIn("94109", r.findings[0].detail)

    def test_user_info_from_script_is_not_invented(self):
        script = "You are Ann. Your email is ann@example.com, order #W1."
        self.assertNotIn("user_invented_info", classify_one(good_messages(), script=script).labels)

    def test_user_invented_info_agent_recovered_is_evidence_only(self):
        script = "You are Ann. Your email is ann@example.com."
        msgs = flat(user("zip is 94109"),
                    call("find_user_id_by_name_zip", error=True, first_name="Ann", last_name="X",
                         zip="94109"),
                    bad_messages())  # agent later authenticates by email, then cancels the wrong order
        r = classify_one(msgs, script=script, **FAIL)
        self.assertIn("user_invented_info", r.labels)
        self.assertEqual(r.primary, "wrong_args")

    def test_nl_assertion_fail(self):
        r = classify_one(good_messages(), reward=0.0, db_match=True, nl=["Agent should say $5"])
        self.assertEqual(r.primary, "nl_assertion_fail")


if __name__ == "__main__":
    unittest.main()
