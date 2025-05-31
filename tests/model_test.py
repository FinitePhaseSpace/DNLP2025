import unittest
import tempfile
import os

from dnlp2025.model import *


class MyTestCase(unittest.TestCase):

    def test_save_and_load_model(self):
        model = AIAYNModel()
        model.eval()

        # Generate dummy input and output
        input_tensor = torch.randn(1, 10)
        output_before = model(input_tensor)

        # Save the model
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            torch.save(model.state_dict(), tmp.name)
            path = tmp.name

        # Load into a new instance
        loaded_model = AIAYNModel()
        loaded_model.load_state_dict(torch.load(path))
        loaded_model.eval()

        output_after = loaded_model(input_tensor)

        # Clean up
        os.remove(path)

        # Compare outputs
        self.assertTrue(torch.allclose(output_before, output_after, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
