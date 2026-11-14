import os
import tempfile
import unittest
from pathlib import Path

# We will test the model catalog discovery, formatting, and selection logic
from download_models import (
    build_model_catalog,
    format_model_catalog_table,
    resolve_model_choice,
    KNOWN_REPOSITORIES
)

class TestModelCatalog(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.drive_root = os.path.join(self.temp_dir.name, "drive_root")
        self.comfy_dir = os.path.join(self.temp_dir.name, "ComfyUI")
        
        # Create mock drive models folder
        self.drive_diff_dir = os.path.join(self.drive_root, "models", "Diffusion Models")
        os.makedirs(self.drive_diff_dir, exist_ok=True)
        
        # Create a mock cached model in Drive
        self.mock_cached_model = os.path.join(self.drive_diff_dir, "my_custom_qwen_q4.gguf")
        with open(self.mock_cached_model, "wb") as f:
            f.write(b"MOCK_GGUF_DATA" * 1000)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_catalog_discovery_with_drive_cache(self):
        catalog = build_model_catalog(
            drive_root=self.drive_root,
            comfy_dir=self.comfy_dir,
            fetch_online=False  # Test local/offline discovery first
        )
        self.assertGreater(len(catalog), 0)
        
        # Check that the Drive cached model is discovered and marked
        cached_entries = [m for m in catalog if m["filename"] == "my_custom_qwen_q4.gguf"]
        self.assertEqual(len(cached_entries), 1)
        self.assertEqual(cached_entries[0]["source"], "Google Drive Cache")
        self.assertEqual(cached_entries[0]["status"], "Ready (Drive)")

    def test_catalog_includes_known_community_developers(self):
        catalog = build_model_catalog(
            drive_root=self.drive_root,
            comfy_dir=self.comfy_dir,
            fetch_online=False
        )
        developers = {m["developer"] for m in catalog}
        # Check that major developers are represented in curated catalog
        self.assertTrue(any("KasugaiSakura" in d or "abenzerps" in d for d in developers))
        self.assertTrue(any("Unsloth" in d for d in developers))
        self.assertTrue(any("AtomicChat" in d for d in developers))
        self.assertTrue(any("Comfy-Org" in d for d in developers))

    def test_resolve_choice_by_id(self):
        catalog = build_model_catalog(fetch_online=False)
        first_entry = catalog[0]
        resolved = resolve_model_choice("1", catalog)
        self.assertEqual(resolved["id"], 1)
        self.assertEqual(resolved["filename"], first_entry["filename"])

    def test_resolve_choice_by_filename(self):
        catalog = build_model_catalog(fetch_online=False)
        target = "qwen-image-2.1-UC-Q4_K_M.gguf"
        resolved = resolve_model_choice(target, catalog)
        self.assertEqual(resolved["filename"], target)

    def test_resolve_choice_by_quant_keyword(self):
        catalog = build_model_catalog(fetch_online=False)
        resolved = resolve_model_choice("Q8_0", catalog)
        self.assertIn("Q8_0", resolved["filename"])

    def test_resolve_custom_repo_entry(self):
        catalog = build_model_catalog(fetch_online=False)
        custom_spec = "custom-dev/qwen-custom-gguf:my-custom-model.gguf"
        resolved = resolve_model_choice(custom_spec, catalog)
        self.assertEqual(resolved["filename"], "my-custom-model.gguf")
        self.assertEqual(resolved["repo"], "custom-dev/qwen-custom-gguf")
        self.assertEqual(resolved["source"], "Hugging Face (Custom)")

    def test_format_catalog_table(self):
        catalog = build_model_catalog(fetch_online=False)
        text_table = format_model_catalog_table(catalog, as_html=False)
        self.assertIn("ID", text_table)
        self.assertIn("Model Filename", text_table)
        self.assertIn("Developer", text_table)
        
        html_table = format_model_catalog_table(catalog, as_html=True)
        self.assertIn("<table", html_table)
        self.assertIn("KasugaiSakura", html_table)

if __name__ == "__main__":
    unittest.main()
