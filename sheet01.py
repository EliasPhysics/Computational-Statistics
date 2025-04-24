import numpy as np

class KSigmaClipper:
    def __init__(self, k=3, max_iter=5, use_median=False):
        """
        Initialize the clipper.
        :param k: The sigma threshold for clipping
        :param max_iter: Maximum number of iterations
        :param use_median: Use median instead of mean for central tendency
        """
        self.k = k
        self.max_iter = max_iter
        self.use_median = use_median

    def clip(self, data):
        data = np.asarray(data)
        mask = np.ones(data.shape, dtype=bool)

        for _ in range(self.max_iter):
            working_data = data[mask]
            center = np.median(working_data) if self.use_median else np.mean(working_data)
            std = np.std(working_data)

            new_mask = np.abs(data - center) <= self.k * std
            if np.array_equal(mask, new_mask):
                break
            mask = new_mask

        self.clipped_data = data[mask]
        return self.clipped_data

    def clipped_mean(self):
        return np.mean(self.clipped_data)

    def robust_rms(self):
        return np.sqrt(np.mean(np.square(self.clipped_data)))

