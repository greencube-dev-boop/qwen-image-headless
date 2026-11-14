import json
import os
import unittest

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "Qwen_Image_2.1_GGUF_ComfyUI.ipynb")

class TestNotebookStructure(unittest.TestCase):
    def test_notebook_json_validity(self):
        self.assertTrue(os.path.isfile(NOTEBOOK_PATH))
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        self.assertIn("cells", nb)
        cells = nb["cells"]
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 4)

        # Check for model discovery in code
        all_code = "".join("".join(c["source"]) for c in code_cells)
        self.assertIn("build_model_catalog", all_code)
        self.assertIn("format_model_catalog_table", all_code)
        self.assertIn("resolve_model_choice", all_code)
        self.assertIn("sync_selected_model", all_code)
        self.assertIn("SELECTED_DIFFUSION_FILE", all_code)
        self.assertIn("load_input_images_table", all_code)
        self.assertIn("GEN_CONFIG", all_code)

if __name__ == "__main__":
    unittest.main()
