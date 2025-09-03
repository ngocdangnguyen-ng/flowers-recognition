import os
import numpy as np
from PIL import Image

class FlowerDataLoader:
	"""
	Loads image paths and class labels, computes statistics, and visualizes samples for EDA and preprocessing.
	"""
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.class_names = self._get_class_names()
		self.image_paths = self._get_image_paths()

	def _get_class_names(self):
		"""Return sorted list of class names (subfolders)."""
		return sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])

	def _get_image_paths(self):
		"""Return list of dicts: {'path': ..., 'class': ...}"""
		image_paths = []
		for class_name in self.class_names:
			class_dir = os.path.join(self.data_dir, class_name)
			for fname in os.listdir(class_dir):
				if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
					image_paths.append({
						'path': os.path.join(class_dir, fname),
						'class': class_name
					})
		return image_paths

	def get_class_distribution(self):
		"""Return dict: class_name -> count."""
		from collections import Counter
		return dict(Counter([img['class'] for img in self.image_paths]))

	def get_image_size_stats(self):
		"""Return min/max/mean width/height for all images."""
		sizes = []
		for img in self.image_paths:
			try:
				with Image.open(img['path']) as im:
					sizes.append(im.size)
			except Exception as e:
				print(f"Error processing {img['path']}: {e}")
				continue
		if sizes:
			widths, heights = zip(*sizes)
			return {
				'min_width': min(widths),
				'max_width': max(widths),
				'min_height': min(heights),
				'max_height': max(heights),
				'mean_width': float(np.mean(widths)),
				'mean_height': float(np.mean(heights))
			}
		return {}

	def get_sample_images(self, n_per_class=3):
		"""Return up to n_per_class sample images per class."""
		samples = []
		for class_name in self.class_names:
			class_imgs = [img for img in self.image_paths if img['class'] == class_name]
			samples.extend(class_imgs[:n_per_class])
		return samples

	def visualize_class_distribution(self):
		"""Bar plot of class distribution."""
		import matplotlib.pyplot as plt
		dist = self.get_class_distribution()
		plt.figure(figsize=(8, 4))
		plt.bar(dist.keys(), dist.values(), color='skyblue')
		plt.title('Class Distribution')
		plt.xlabel('Class')
		plt.ylabel('Count')
		plt.show()

	def visualize_sample_images(self, n_per_class=3):
		"""Show sample images per class."""
		import matplotlib.pyplot as plt
		samples = self.get_sample_images(n_per_class)
		n = len(samples)
		plt.figure(figsize=(3*n, 3))
		for i, sample in enumerate(samples):
			img = Image.open(sample['path'])
			plt.subplot(1, n, i+1)
			plt.imshow(img)
			plt.title(sample['class'])
			plt.axis('off')
		plt.tight_layout()
		plt.show()

	def summary(self):
		print(f"Classes: {self.class_names}")
		print(f"Total images: {len(self.image_paths)}")
		print(f"Class distribution: {self.get_class_distribution()}")
		print(f"Image size stats: {self.get_image_size_stats()}")
