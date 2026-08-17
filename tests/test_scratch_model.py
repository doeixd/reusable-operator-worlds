import unittest

import numpy as np

from row.models import ScratchResidualMLP


class ScratchModelTests(unittest.TestCase):
    def test_training_reduces_loss_on_fixed_batch(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(32, 4))
        y = np.tanh(0.5 * x)
        model = ScratchResidualMLP(4, 16, seed=3, learning_rate=0.01)
        before = np.mean(np.square(model.predict(x) - y))
        for _ in range(200):
            model.update(x, y)
        after = np.mean(np.square(model.predict(x) - y))
        self.assertLess(after, before * 0.25)


if __name__ == "__main__":
    unittest.main()
