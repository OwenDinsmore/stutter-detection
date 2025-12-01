import numpy as np

def time_mask(mfcc, mask_prob=0.5, max_mask_length=10):
    if np.random.rand() < mask_prob:
        augmented = mfcc.copy()
        mask_length = np.random.randint(1, max_mask_length)
        mask_start = np.random.randint(0, mfcc.shape[0] - mask_length)
        augmented[mask_start:mask_start + mask_length, :] = 0
        return augmented
    return mfcc

def frequency_mask(mfcc, mask_prob=0.5, max_mask_freqs=4):
    if np.random.rand() < mask_prob:
        augmented = mfcc.copy()
        mask_freqs = np.random.randint(1, max_mask_freqs)
        mask_start = np.random.randint(0, mfcc.shape[1] - mask_freqs)
        augmented[:, mask_start:mask_start + mask_freqs] = 0
        return augmented
    return mfcc

def add_noise(mfcc, noise_prob=0.5, noise_std=0.01):
    if np.random.rand() < noise_prob:
        noise = np.random.normal(0, noise_std, mfcc.shape)
        return mfcc + noise
    return mfcc

def time_shift(mfcc, shift_prob=0.5, max_shift=10):
    if np.random.rand() < shift_prob:
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(mfcc, shift, axis=0)
    return mfcc

def augment_mfcc(mfcc):
    augmented = mfcc.copy()
    augmented = time_mask(augmented)
    augmented = frequency_mask(augmented)
    augmented = add_noise(augmented)
    augmented = time_shift(augmented)
    return augmented
