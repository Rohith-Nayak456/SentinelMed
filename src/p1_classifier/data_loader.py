
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ManifestImageDataset(Dataset):
    """Load images and labels using P2's dataset manifest."""

    VALID_SPLITS = {"train", "val", "test"}
    VALID_LABELS = {0, 1}

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str,
        image_size: int | None = None,
        num_channels: int = 1,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split: {split}")

        if image_size is not None and image_size < 1:
            raise ValueError("image_size must be positive or None.")

        if num_channels not in (1, 3):
            raise ValueError("num_channels must be 1 or 3.")

        self.data_root = Path(data_root)
        manifest_path = Path(manifest_path)

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"Dataset root not found: {self.data_root}"
            )

        self.transform = transforms.Compose([
            *(
                [transforms.Resize((image_size, image_size))]
                if image_size is not None
                else []
            ),
            transforms.Grayscale(num_output_channels=num_channels),
            transforms.ToTensor(),
        ])

        with manifest_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)

            required_columns = {
                "image_id", "label", "split",
                "degradation", "image_path",
            }
            missing = required_columns - set(reader.fieldnames or [])

            if missing:
                raise ValueError(
                    f"Manifest is missing columns: {sorted(missing)}"
                )

            self.rows = [
                row for row in reader
                if row["split"] == split
                and row["degradation"] == "original"
            ]

        if not self.rows:
            raise ValueError(
                f"No original images found for split: {split}"
            )

        self.samples = []

        for row in self.rows:
            try:
                label = int(row["label"])
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid label for {row['image_id']}"
                )

            if label not in self.VALID_LABELS:
                raise ValueError(
                    f"Unexpected label {label} for {row['image_id']}"
                )

            image_path = self.data_root / row["image_path"]

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image missing for {row['image_id']}: {image_path}"
                )

            self.samples.append((image_path, label, row["image_id"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, image_id = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("L")
            image_tensor = self.transform(image)

        return image_tensor, label


def create_image_datasets(
    manifest_path: str,
    data_root: str,
    image_size: int | None = None,
    num_channels: int = 1,
):
    """Return training and validation datasets; test stays separate."""
    train_dataset = ManifestImageDataset(
        manifest_path, data_root, "train", image_size, num_channels
    )
    val_dataset = ManifestImageDataset(
        manifest_path, data_root, "val", image_size, num_channels
    )
    return train_dataset, val_dataset


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size: int = 32,
):
    """Create shuffled training and non-shuffled validation loaders."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader
