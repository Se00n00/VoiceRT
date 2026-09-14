"""SessionStore unit tests: no GPU, no weights, no server."""
import threading
import unittest

from engine.session import SessionStore, new_session_id


class TestSessionStore(unittest.TestCase):
    def test_history_autocreates_empty(self):
        st = SessionStore()
        self.assertEqual(st.history("abc"), [])

    def test_remember_turn_order(self):
        st = SessionStore()
        st.remember_turn("s", "hi", "hello")
        self.assertEqual(st.history("s"), [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])

    def test_max_turns_prunes_oldest(self):
        st = SessionStore(max_turns=4)
        for i in range(5):
            st.remember_turn("s", f"u{i}", f"a{i}")
        hist = st.history("s")
        self.assertEqual(len(hist), 4)
        self.assertEqual(hist[0]["content"], "u3")

    def test_reset_and_drop(self):
        st = SessionStore()
        st.remember_turn("s", "u", "a")
        st.reset("s")
        self.assertEqual(st.history("s"), [])
        st.remember_turn("s", "u", "a")
        self.assertTrue(st.drop("s"))
        self.assertFalse(st.drop("s"))

    def test_ttl_sweeps(self):
        import time

        st = SessionStore(max_age_s=0.05)
        st.remember_turn("old", "u", "a")
        time.sleep(0.08)
        st.remember_turn("new", "u", "a")  # access triggers sweep
        self.assertEqual(st.stats()["sessions"], 1)

    def test_max_sessions_evicts_lru(self):
        st = SessionStore(max_sessions=2)
        st.remember_turn("a", "u", "a")
        st.remember_turn("b", "u", "a")
        st.remember_turn("c", "u", "a")
        self.assertEqual(st.stats()["sessions"], 2)
        self.assertEqual(st.history("a"), [])  # evicted

    def test_thread_safe(self):
        st = SessionStore(max_turns=10000)
        ths = [threading.Thread(target=lambda i=i: st.remember_turn(
            "s", f"u{i}", f"a{i}")) for i in range(20)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        self.assertEqual(len(st.history("s")), 40)

    def test_ids_unique(self):
        self.assertNotEqual(new_session_id(), new_session_id())


if __name__ == "__main__":
    unittest.main()
