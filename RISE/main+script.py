# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras import backend as K
from keras.preprocessing import image

# Define the Model class
class Model():
    def __init__(self):
        K.set_learning_phase(0)
        self.model = ResNet50()
        self.input_size = (224, 224)

    def run_on_batch(self, x):
        return self.model.predict(x)

# Load and preprocess image
def load_img(path):
    img = image.load_img(path, target_size=model.input_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

# Generate masks for RISE
def generate_masks(N, s, p1):
    cell_size = np.ceil(np.array(model.input_size) / s)
    up_size = (s + 1) * cell_size
    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')
    masks = np.empty((N, *model.input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + model.input_size[0], y:y + model.input_size[1]]
    masks = masks.reshape(-1, *model.input_size, 1)
    return masks

# Explain the model's prediction using RISE
def explain(model, inp, masks):
    preds = []
    masked = inp * masks
    for i in tqdm(range(0, N, batch_size), desc='Explaining'):
        preds.append(model.run_on_batch(masked[i:min(i+batch_size, N)]))
    preds = np.concatenate(preds)
    sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *model.input_size)
    sal = sal / N / p1
    return sal

# Define the class_name function
def class_name(idx):
    return decode_predictions(np.eye(1, 1000, idx))[0][0][1]


# Create an instance of the Model class
model = Model()

# Load an image and preprocess it
image_path = 'data/black_bear/ILSVRC2012_val_00000592.JPEG'
img, x = load_img(image_path)

# Set the parameters for RISE
N = 2000
s = 8
p1 = 0.5
num_masks_to_display = 4

# Generate masks for RISE
masks = generate_masks(N, s, p1)

# Display some of the generated masks
fig, axs = plt.subplots(1, num_masks_to_display, figsize=(15, 3))
for i in range(num_masks_to_display):
    ax = axs[i]
    im = ax.imshow(masks[i, :, :, 0], cmap='jet')
    ax.set_title(f'Mask {i+1}')
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
plt.tight_layout()
plt.show()

# Explain the model's prediction using RISE
sal = explain(model, x, masks)

# Overlay the saliency map on the input image
class_idx = 243  # Replace with the index of the class you want to explain
plt.title('Explanation for `{}`'.format(class_name(class_idx)))
plt.axis('off')
plt.imshow(img)
plt.imshow(sal[class_idx], cmap='jet', alpha=0.5)
plt.show()


get_ipython().run_line_magic('run', '/path/to/RISE_resnet50.py')

