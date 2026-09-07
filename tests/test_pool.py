import unittest

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


if __name__ == "__main__":
    unittest.main()
