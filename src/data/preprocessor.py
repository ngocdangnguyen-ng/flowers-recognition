import numpy as np
from PIL import Image


class SimpleImagePreprocessor:
	"""
	Preprocesses flower images: resize, normalize, split. Matches notebook logic.
	"""
	def __init__(self, img_size=(224, 224), random_seed=42):
		self.img_size = img_size
		self.random_seed = random_seed

	def preprocess_image(self, image_path):
		try:
			image = Image.open(image_path)
			if image.mode != 'RGB':
				image = image.convert('RGB')
			image = image.resize(self.img_size, Image.Resampling.LANCZOS)
			image_array = np.array(image).astype(np.float32) / 255.0
			return image_array, image
		except Exception as e:
			print(f"Error processing image {image_path}: {e}")
			return None, None

	def preprocess_pil_image(self, pil_image):
		try:
			if pil_image.mode != 'RGB':
				pil_image = pil_image.convert('RGB')
			pil_image = pil_image.resize(self.img_size, Image.Resampling.LANCZOS)
			image_array = np.array(pil_image).astype(np.float32) / 255.0
			return image_array, pil_image
		except Exception as e:
			print(f"Error processing PIL image: {e}")
			return None, None

	def split_dataset(self, image_paths, train_split=0.7, val_split=0.15, test_split=0.15):
		"""Split image paths into train/val/test sets."""
		np.random.seed(self.random_seed)
		n = len(image_paths)
		idx = np.random.permutation(n)
		train_end = int(train_split * n)
		val_end = train_end + int(val_split * n)
		train_idx = idx[:train_end]
		val_idx = idx[train_end:val_end]
		test_idx = idx[val_end:]
		train = [image_paths[i] for i in train_idx]
		val = [image_paths[i] for i in val_idx]
		test = [image_paths[i] for i in test_idx]
		return train, val, test
		indices = np.arange(len(image_paths))
		np.random.shuffle(indices)
		n_train = int(train_split * len(indices))
		n_val = int(val_split * len(indices))
		train_idx = indices[:n_train]
		val_idx = indices[n_train:n_train+n_val]
		test_idx = indices[n_train+n_val:]
		train = [image_paths[i] for i in train_idx]
		val = [image_paths[i] for i in val_idx]
		test = [image_paths[i] for i in test_idx]
		return train, val, test
