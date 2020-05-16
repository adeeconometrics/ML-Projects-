import numpy as np
import matplotlib.pyplot as plt

# initializing arrays
np.empty([1, 1])
np.ones([2, 3])  # is this the same as np.ones((2,3))?
np.zeros((2, 2))


# plotting 2D colormap
np.linspace(0, 50, 30)
image = np.random.randn(30, 30)
plt.imshow(image, cmap=plt.cm.hot)
plt.colorbar()

# simple visualizations

array_cosine = np.cos()
