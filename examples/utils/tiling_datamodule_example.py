"""
Examples and runnable demonstrations for TilingDataModuleWrapper.

This file combines usage examples and a runnable end-to-end inference demo.

Contents:
  Usage Examples (illustrative, require real data paths):
    1. Basic usage
    2. With model patch size
    3. Inference-only tiling
    4. Inference with prediction stitching
    5. Lightning CLI YAML configuration
    6. Performance comparison
    7. Cache management

  Runnable Demo (self-contained with mock data):
    8. End-to-end inference with tile stitching

Run the demos directly:
    python examples/utils/tiling_datamodule_example.py
"""

import tempfile

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from terratorch.datamodules import GenericNonGeoSegmentationDataModule, TilingDataModuleWrapper


# ---------------------------------------------------------------------------
# Shared mock classes (used by the runnable demo)
# ---------------------------------------------------------------------------

class MockDataset(Dataset):
    """Synthetic dataset with gradient images for demonstration."""

    def __init__(self, num_samples=3, image_size=(512, 512)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        h, w = self.image_size
        y_grid = torch.linspace(0, 1, h).unsqueeze(1).expand(h, w)
        x_grid = torch.linspace(0, 1, w).unsqueeze(0).expand(h, w)
        image = torch.stack([y_grid, x_grid, (y_grid + x_grid) / 2])
        return {
            "image": image,
            "mask": torch.randint(0, 5, (h, w)),
            "filename": f"image_{idx}.tif",
            "original_size": (h, w),
        }


class MockDataModule(LightningDataModule):
    """Minimal datamodule backed by MockDataset."""

    def __init__(self, batch_size=4, image_size=(512, 512)):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.predict_dataset = None

    def setup(self, stage=None):
        if stage in ("predict", None):
            self.predict_dataset = MockDataset(num_samples=2, image_size=self.image_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)


class MockSegmentationModel(torch.nn.Module):
    """Tiny segmentation model for demonstration."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


# Mock dataset for documentation examples
class SimpleDummyDataset(Dataset):
    """Simple dataset with large images for testing tiling."""

    def __init__(self, num_samples=100, image_size=(1024, 1024), num_classes=5):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        h, w = self.image_size
        image = torch.rand(3, h, w)
        mask = torch.randint(0, self.num_classes, (h, w))
        return {
            "image": image,
            "mask": mask,
            "filename": f"image_{idx}.tif",
        }


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Step 1: Create your base datamodule
    print("\n1. Creating base datamodule...")
    base_dm = GenericNonGeoSegmentationDataModule(
        batch_size=4,
        num_workers=0,
        train_data_root="./data/train",  # Your data path
        val_data_root="./data/val",
        test_data_root="./data/test",
        num_classes=10,
    )
    
    # Step 2: Wrap with tiling
    print("2. Wrapping with tiling...")
    tiled_dm = TilingDataModuleWrapper(
        base_datamodule=base_dm,
        tile_size=(512, 512),     # Tile to 512x512 patches
        overlap=64,                # 64 pixel overlap
        cache_dir="./tile_cache",  # Cache location
        apply_to_splits=["train", "val"],  # Which splits to tile
    )
    
    print("3. DataModule ready! Now use it in training:")
    print("   trainer.fit(model, tiled_dm)")
    print()


def example_with_model_patch_size():
    """Example with model patch size consideration."""
    print("=" * 60)
    print("Example 2: With Model Patch Size")
    print("=" * 60)
    
    # Your model expects inputs divisible by patch_size=16
    print("\n1. Model requires patch_size=16 (e.g., ViT-based)")
    
    base_dm = GenericNonGeoSegmentationDataModule(
        batch_size=4,
        num_workers=0,
        train_data_root="./data/train",
        num_classes=10,
    )
    
    # Configure tiling to respect patch size
    print("2. Configuring tiling with patch_size...")
    tiled_dm = TilingDataModuleWrapper(
        base_datamodule=base_dm,
        tile_size=(512, 512),       # Already divisible by 16
        overlap=64,
        patch_size=16,              # Model patch size
        padding="symmetric",        # Match model's expected padding
        cache_dir="./tile_cache",
    )
    
    print("3. Tiles will be compatible with model's patch size")
    print()


def example_inference_only():
    """Example for inference-time tiling."""
    print("=" * 60)
    print("Example 3: Inference-Only Tiling")
    print("=" * 60)
    
    base_dm = GenericNonGeoSegmentationDataModule(
        batch_size=1,
        num_workers=0,
        test_data_root="./data/test",
        num_classes=10,
    )
    
    # Only tile test split (for inference)
    print("\n1. Tiling only test split for inference...")
    tiled_dm = TilingDataModuleWrapper(
        base_datamodule=base_dm,
        tile_size=(512, 512),
        overlap=128,                    # More overlap for smoother predictions
        apply_to_splits=["test"],       # Only test split
        cache_dir="./tile_cache_test",
        keep_incomplete_tiles=True,     # Keep edge tiles
    )
    
    print("2. Use for inference:")
    print("   trainer.test(model, tiled_dm)")
    print()


def example_inference_with_stitching():
    """Illustrative example of inference with prediction stitching (requires real data).

    For a self-contained runnable version, see example_inference_stitching_runnable().
    """
    print("=" * 60)
    print("Example 3b: Inference with Prediction Stitching (Illustrative)")
    print("=" * 60)

    print("""
    # 1. Setup tiled datamodule for inference
    tiled_dm = TilingDataModuleWrapper(
        base_datamodule=your_datamodule,  # any LightningDataModule
        tile_size=(256, 256),
        overlap=64,
        apply_to_splits=["test"],
        keep_incomplete_tiles=True,       # Needed for full image coverage
        cache_dir="./tile_cache_inference",
    )
    tiled_dm.setup("test")

    # 2. Run inference and collect tile predictions
    predictions, coords = [], []
    for batch in tiled_dm.test_dataloader():
        with torch.no_grad():
            preds = model(batch["image"])  # [B, C, H, W]
        predictions.append(preds)
        coords.extend(batch["tile_coords"])  # [(y1, x1, y2, x2), ...]

    all_preds = torch.cat(predictions, dim=0)  # [N_tiles, C, H, W]

    # 3. Stitch tiles back into the original image size
    stitched = TilingDataModuleWrapper.stitch_predictions(
        tile_predictions=all_preds,
        tile_coords=coords,
        original_size=(1024, 1024),  # H, W of original image
        overlap=64,
        use_blending=True,           # Smooth cosine blending at boundaries
    )
    # stitched.shape => [C, 1024, 1024]

    # 4. Inspect the blend mask used for stitching
    blend_mask = TilingDataModuleWrapper.get_blend_mask(tile_size=256, overlap=64)
    # blend_mask[128, 128]  => 1.0  (center, full weight)
    # blend_mask[0, 128]    => ~0.0 (edge, zero weight)
    """)
    print("  (See Example 8 for a self-contained runnable version)")
    print()


def example_configuration_file():
    """Example YAML configuration."""
    print("=" * 60)
    print("Example 4: Lightning CLI Configuration")
    print("=" * 60)
    
    config = """
# config.yaml
trainer:
  max_epochs: 10
  accelerator: gpu
  devices: 1

model:
  class_path: terratorch.tasks.SemanticSegmentationTask
  init_args:
    model_args:
      encoder: prithvi_vit_100
      decoder: FCNDecoder
    num_classes: 10

data:
  class_path: terratorch.datamodules.TilingDataModuleWrapper
  init_args:
    # Base datamodule to wrap
    base_datamodule:
      class_path: terratorch.datamodules.GenericNonGeoSegmentationDataModule
      init_args:
        batch_size: 8
        num_workers: 4
        train_data_root: ./data/train
        val_data_root: ./data/val
        num_classes: 10
    
    # Tiling configuration
    tile_size: [512, 512]
    overlap: 64
    cache_dir: ./tile_cache
    apply_to_splits: [train, val]
    rebuild_cache: false
    keep_incomplete_tiles: true
    
    # Optional: for model compatibility
    patch_size: 16
    padding: symmetric
"""
    
    print("\nSave this as config.yaml, then run:")
    print("  python -m terratorch fit --config config.yaml")
    print("\nConfig:")
    print(config)


def example_comparing_performance():
    """Compare training with/without tiling."""
    print("=" * 60)
    print("Example 5: Performance Comparison")
    print("=" * 60)
    
    print("\nScenario: Training on 1024x1024 images")
    print()
    
    print("WITHOUT Tiling:")
    print("  ❌ Model pads each image to divisible size in forward()")
    print("  ❌ Padding computed every epoch")
    print("  ❌ Large images may exceed GPU memory")
    print("  ❌ Inconsistent padding across model versions")
    print()
    
    print("WITH Tiling:")
    print("  ✅ Images tiled once to 512x512, cached to disk")
    print("  ✅ No padding needed in model forward()")
    print("  ✅ Smaller tiles fit in GPU memory")
    print("  ✅ More samples per epoch (each tile is a sample)")
    print("  ✅ Consistent behavior across model versions")
    print()
    
    print("First Epoch:")
    print("  Without tiling: 100 images × 100ms = 10 seconds")
    print("  With tiling:    400 tiles × 100ms (tiling) + 5ms (cache write) = 42 seconds")
    print()
    
    print("Second Epoch:")
    print("  Without tiling: 100 images × 100ms = 10 seconds")
    print("  With tiling:    400 tiles × 5ms (cache read) = 2 seconds  ⚡")
    print()
    
    print("After 10 epochs:")
    print("  Without tiling: 100 seconds")
    print("  With tiling:    60 seconds (42 + 9×2)  💰")
    print()


def example_cache_management():
    """Example of cache management."""
    print("=" * 60)
    print("Example 6: Cache Management")
    print("=" * 60)
    
    print("\n1. Check cache size:")
    print("   $ du -sh ./tile_cache")
    print()
    
    print("2. Clear cache:")
    print("   $ rm -rf ./tile_cache")
    print()
    
    print("3. Rebuild cache (force):")
    tiled_dm_code = """
    tiled_dm = TilingDataModuleWrapper(
        base_datamodule=base_dm,
        tile_size=(512, 512),
        cache_dir="./tile_cache",
        rebuild_cache=True,  # Force rebuild
    )
    """
    print(tiled_dm_code)
    print()
    
    print("4. Cache structure:")
    print("""
    tile_cache/
    ├── train/
    │   ├── tile_index_abc123.json
    │   ├── tile_0_0_0_512_512.pt
    │   └── ...
    └── val/
        └── ...
    """)


def example_inference_stitching_runnable():
    """Runnable end-to-end demo: inference with tile stitching (uses mock data)."""
    print("=" * 70)
    print("Example 8: End-to-End Inference Demo (Runnable)")
    print("=" * 70)

    tile_size = (256, 256)
    overlap = 64
    num_classes = 5
    image_size = (512, 512)

    print(f"\nConfiguration:")
    print(f"  - Image size:      {image_size}")
    print(f"  - Tile size:       {tile_size}")
    print(f"  - Overlap:         {overlap}px")
    print(f"  - Num classes:     {num_classes}")
    print()

    base_dm = MockDataModule(batch_size=4, image_size=image_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        tiled_dm = TilingDataModuleWrapper(
            base_datamodule=base_dm,
            tile_size=tile_size,
            overlap=overlap,
            apply_to_splits=["predict"],
            keep_incomplete_tiles=True,
            cache_dir=tmpdir,
        )
        tiled_dm.setup("predict")
        dataloader = tiled_dm.predict_dataloader()

        model = MockSegmentationModel(num_classes=num_classes)
        model.eval()

        # --- Collect tile predictions for the first image ---
        all_predictions, all_coords = [], []
        first_image_done = False
        total_tiles = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"]
            coords = batch["tile_coords"]
            base_indices = batch["base_idx"]

            print(f"  Batch {batch_idx + 1}: {images.shape[0]} tiles, shape {tuple(images.shape)}")

            with torch.no_grad():
                preds = model(images)

            for i in range(images.shape[0]):
                bidx = base_indices[i].item() if not isinstance(base_indices, list) else base_indices[i]
                if bidx == 0:
                    all_predictions.append(preds[i:i + 1])
                    all_coords.append(coords[i])
                    total_tiles += 1
                else:
                    first_image_done = True
                    break
            if first_image_done:
                break

        all_predictions = torch.cat(all_predictions, dim=0)
        print(f"\n  Collected {all_predictions.shape[0]} tiles from first image")

        # --- Stitch back into full image ---
        stitched = TilingDataModuleWrapper.stitch_predictions(
            tile_predictions=all_predictions,
            tile_coords=all_coords,
            original_size=image_size,
            overlap=overlap,
            use_blending=True,
        )
        assert stitched.shape == (num_classes, image_size[0], image_size[1])
        assert torch.all(torch.isfinite(stitched))
        print(f"  Stitched output shape: {tuple(stitched.shape)}  ✓")

        # --- Inspect blend mask ---
        blend_mask = TilingDataModuleWrapper.get_blend_mask(tile_size=tile_size[0], overlap=overlap)
        print(f"\n  Blend mask shape:  {tuple(blend_mask.shape)}")
        print(f"  Center value:      {blend_mask[128, 128]:.4f}  (should be 1.0)")
        print(f"  Edge corner:       {blend_mask[0, 0]:.4f}  (ramp start)")
        print(f"  Edge midpoint:     {blend_mask[0, 128]:.4f}  (ramp in one direction)")
        print()

    print("✓ All checks passed!")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "TilingDataModuleWrapper — Examples & Runnable Demo" + " " * 8 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Illustrative usage examples (require real data paths)
    example_basic_usage()
    example_with_model_patch_size()
    example_inference_only()
    example_inference_with_stitching()
    example_configuration_file()
    example_comparing_performance()
    example_cache_management()

    # Self-contained runnable demo
    example_inference_stitching_runnable()

    print("=" * 70)
    print("For more details, see:")
    print("  docs/tiling_datamodule_wrapper.md")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
