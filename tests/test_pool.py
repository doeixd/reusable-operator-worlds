import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from row.pool import BatchFailed, PoolBudget, run_pool

GIB = 1024**3
MIB = 1024**2


def square(job):
    return job * job


def fail_on_three(job):
    if job == 3:
        raise ValueError("cell 3 broke")
    return job


class PoolBudgetTests(unittest.TestCase):
    def test_cap_is_memory_bound_then_core_bound(self):
        budget = PoolBudget(measured_rss_bytes=364 * MIB, reserve_bytes=4 * GIB, cores=16)
        # 9.8 GiB free: (9.8 - 4) / 0.355 = 16.3 -> core bound at 14.
        self.assertEqual(budget.cap(int(9.8 * GIB)), 14)
        # 4.7 GiB free: (4.7 - 4) / 0.355 = 1.97 -> memory bound at 1.
        self.assertEqual(budget.cap(int(4.7 * GIB)), 1)
        # Below reserve: nothing may start.
        self.assertEqual(budget.cap(3 * GIB), 0)

    def test_hard_cap_and_measured_rss_required(self):
        budget = PoolBudget(measured_rss_bytes=364 * MIB, cores=16, hard_cap=3)
        self.assertEqual(budget.cap(20 * GIB), 3)
        with self.assertRaises(ValueError):
            PoolBudget(measured_rss_bytes=0).cap(20 * GIB)

    def test_dispatch_is_checked_against_live_free_memory(self):
        budget = PoolBudget(measured_rss_bytes=1 * GIB, reserve_bytes=4 * GIB, cores=16)
        # Free memory falls while the batch runs: the third dispatch must wait.
        readings = iter([9 * GIB, 9 * GIB, 9 * GIB, 6 * GIB, 6 * GIB, 6 * GIB, 9 * GIB] + [9 * GIB] * 50)
        seen = []

        def probe():
            value = next(readings)
            seen.append(value)
            return value

        # Test dispatch arithmetic with in-process workers. Actual torch process
        # concurrency is covered by the real scientific equivalence/memory gate;
        # this synthetic free-memory probe must not launch four real interpreters
        # on a host whose actual commitment may be much lower than the fixture.
        with patch("row.pool.ProcessPoolExecutor", ThreadPoolExecutor):
            results = run_pool(square, [1, 2, 3, 4], budget, free_probe=probe,
                               poll_seconds=0.01, sleep=lambda s: None, log=lambda m: None)
        self.assertEqual(results, [1, 4, 9, 16])
        # The probe was consulted more than once per job: dispatch-time checks.
        self.assertGreater(len(seen), 4)

    def test_failed_cell_fails_the_batch(self):
        budget = PoolBudget(measured_rss_bytes=1 * MIB, reserve_bytes=0, cores=4)
        with self.assertRaises(BatchFailed) as caught:
            run_pool(fail_on_three, [1, 2, 3, 4, 5], budget, free_probe=lambda: 8 * GIB,
                     poll_seconds=0.01, log=lambda m: None)
        self.assertIn("cell 3 broke", str(caught.exception))

    def test_no_room_refuses_to_start(self):
        budget = PoolBudget(measured_rss_bytes=1 * GIB, reserve_bytes=4 * GIB, cores=16)
        with self.assertRaises(BatchFailed):
            run_pool(square, [1], budget, free_probe=lambda: 4 * GIB, log=lambda m: None)

    def test_success_is_delivered_before_later_failure(self):
        received = []
        budget = PoolBudget(measured_rss_bytes=MIB, reserve_bytes=0, hard_cap=1)
        with self.assertRaises(BatchFailed):
            run_pool(fail_on_three, [1, 3, 5], budget, free_probe=lambda: 8 * GIB,
                     on_result=lambda job, result: received.append((job, result)),
                     poll_seconds=0.01, log=lambda m: None)
        self.assertEqual(received, [(1, 1)])

    def test_empty_resume_has_no_workers_and_no_memory_requirement(self):
        self.assertEqual(run_pool(square, [], PoolBudget(MIB), free_probe=lambda: 0), [])

    def test_callback_failure_stops_dispatch(self):
        received = []
        def persist(job, result):
            received.append(job)
            raise OSError("disk full")
        with self.assertRaisesRegex(BatchFailed, "disk full"):
            run_pool(square, [1, 2], PoolBudget(MIB, reserve_bytes=0, hard_cap=1),
                     free_probe=lambda: 8 * GIB, on_result=persist, log=lambda m: None)
        self.assertEqual(received, [1])


if __name__ == "__main__":
    unittest.main()
