import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

labels = ['attached', 'transitioning', 'stalled']

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Dataset Sample Check', fontsize=14, fontweight='bold')

for row, label in enumerate(labels):
    folder = f'dataset_pinn/{label}'
    images = os.listdir(folder)
    samples = random.sample(images, 3)

    for col, fname in enumerate(samples):
        img = mpimg.imread(os.path.join(folder, fname))
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'{label}', fontsize=9)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('dataset_check.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Done")