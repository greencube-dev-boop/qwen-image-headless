import os
import re
import tempfile
import unittest
from pathlib import Path
from PIL import Image

def load_input_images_table(input_dir):
    input_path = Path(input_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in valid_exts and not p.name.startswith(".")],
                   key=lambda p: p.name.lower())
    
    registry = {}
    if not files:
        return registry, "No reference images found"
    
    rows = []
    for i, file_path in enumerate(files, start=1):
        tag = f"<image{i}>"
        registry[tag] = str(file_path)
        try:
            with Image.open(file_path) as im:
                w, h = im.size
        except Exception:
            w, h = "?", "?"
        sz_mb = file_path.stat().st_size / (1024 * 1024)
        sz_str = f"{sz_mb:.2f} MB" if sz_mb >= 1.0 else f"{sz_mb * 1024:.0f} KB"
        rows.append(f"{tag}: {file_path.name} ({w}x{h}, {sz_str})")
        
    table_str = "\n".join(rows)
    return registry, table_str


def parse_prompt_references(prompt, image_reference, input_images):
    referenced_tags = re.findall(r"<image\d+>", prompt)
    unique_tags = list(dict.fromkeys(referenced_tags))
    effective_prompt = prompt
    images_to_send = []

    if image_reference:
        if not unique_tags:
            return effective_prompt, images_to_send, unique_tags, "warning_no_tags"
        
        missing = [t for t in unique_tags if t not in input_images]
        if missing:
            raise ValueError(f"Referenced tags {missing} not found in input table. Available: {list(input_images.keys())}")
        
        images_to_send = [input_images[t] for t in unique_tags]
        tag_map = {}
        for idx, orig_tag in enumerate(unique_tags, start=1):
            target_tag = f"<image{idx}>"
            tag_map[orig_tag] = target_tag
            
        effective_prompt = re.sub(r"<image\d+>", lambda m: tag_map.get(m.group(0), m.group(0)), effective_prompt)
                
    return effective_prompt, images_to_send, unique_tags, "ok"


class TestStep6And7(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.input_dir = self.test_dir.name
        
        # Create sample test images
        img1 = Image.new("RGB", (256, 256), color="red")
        img1.save(os.path.join(self.input_dir, "ref_a.png"))
        
        img2 = Image.new("RGB", (512, 512), color="blue")
        img2.save(os.path.join(self.input_dir, "ref_b.jpg"))

    def tearDown(self):
        self.test_dir.cleanup()

    def test_step6_table_indexing(self):
        registry, table = load_input_images_table(self.input_dir)
        self.assertEqual(len(registry), 2)
        self.assertIn("<image1>", registry)
        self.assertIn("<image2>", registry)
        self.assertTrue(registry["<image1>"].endswith("ref_a.png"))
        self.assertTrue(registry["<image2>"].endswith("ref_b.jpg"))
        self.assertIn("256x256", table)
        self.assertIn("512x512", table)

    def test_step7_text_to_image_no_refs(self):
        prompt = "A majestic lion on a cliff"
        registry, _ = load_input_images_table(self.input_dir)
        eff_prompt, imgs, tags, status = parse_prompt_references(prompt, image_reference=False, input_images=registry)
        self.assertEqual(eff_prompt, prompt)
        self.assertEqual(len(imgs), 0)
        self.assertEqual(len(tags), 0)

    def test_step7_image_reference_mapping(self):
        prompt = "Apply style from <image2> to <image1>"
        registry, _ = load_input_images_table(self.input_dir)
        eff_prompt, imgs, tags, status = parse_prompt_references(prompt, image_reference=True, input_images=registry)
        self.assertEqual(tags, ["<image2>", "<image1>"])
        # Slot remapping: <image2> is first mentioned -> mapped to slot <image1>, <image1> mapped to <image2>
        self.assertEqual(len(imgs), 2)
        self.assertEqual(imgs[0], registry["<image2>"])
        self.assertEqual(imgs[1], registry["<image1>"])
        self.assertEqual(eff_prompt, "Apply style from <image1> to <image2>")

    def test_step7_missing_tag_error(self):
        prompt = "Modify <image5>"
        registry, _ = load_input_images_table(self.input_dir)
        with self.assertRaises(ValueError):
            parse_prompt_references(prompt, image_reference=True, input_images=registry)


if __name__ == "__main__":
    unittest.main()
