from pathlib import Path

image_dir = Path("train/images")
label_dir = Path("train/labels")

images = {p.stem for p in image_dir.iterdir() if p.is_file()}
labels = {p.stem for p in label_dir.iterdir() if p.is_file()}

missing_labels = images - labels
orphan_labels = labels - images

print(f"Images: {len(images)}")
print(f"Labels: {len(labels)}")

print("\nImages without labels:")
for name in sorted(missing_labels):
    print(name)

print("\nLabels without images:")
for name in sorted(orphan_labels):
    print(name)


