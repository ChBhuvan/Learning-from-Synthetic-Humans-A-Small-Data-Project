from surreal_dataset import SurrealSegmentationDataset

dataset_path = "/Users/vaibhavkejriwal/desktop/dataset_surreal/dataset/SURREAL/data"

ds = SurrealSegmentationDataset(dataset_path, split="train")

print("Dataset length:", len(ds))

# Fetch one sample
img, mask = ds[0]

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Unique labels:", mask.unique())

