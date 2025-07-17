import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = 128

NUM_TRAIN = 2000
NUM_TEST = 500

def ensure_clean_dir(path):
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

def setup_dirs():
    for split in [TRAIN_DIR, TEST_DIR]:
        for cls in ["human", "vehicle"]:
            ensure_clean_dir(os.path.join(split, cls))
    print("\n All train/test folders created and cleared.")

def generate_human_spectrogram(idx, out_folder, seed=None):
    np.random.seed(seed)
    t = np.linspace(0, 1, IMG_SIZE)

    freq = np.random.uniform(3, 9)           
    amp = np.random.uniform(10, 35)        
    phase = np.random.uniform(0, 2*np.pi)
    noise_level = np.random.uniform(0.1, 0.6) 
    signal = amp * np.sin(2 * np.pi * freq * t + phase)
    noise = np.random.normal(0, noise_level, IMG_SIZE)
    f = signal + noise

    plt.figure(figsize=(2, 2), dpi=IMG_SIZE//2)
    plt.plot(t, f, color='black', linewidth=2)
    plt.axis('off')
    plt.savefig(os.path.join(out_folder, f"human_{idx}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_vehicle_spectrogram(idx, out_folder, seed=None):
    np.random.seed(seed)
    t = np.linspace(0, 1, IMG_SIZE)

    slope = np.random.uniform(8, 35)        
    intercept = np.random.uniform(-3, 3)    
    noise_level = np.random.uniform(0.05, 0.4)

    signal = slope * t + intercept
    noise = np.random.normal(0, noise_level, IMG_SIZE)
    f = signal + noise

    plt.figure(figsize=(2, 2), dpi=IMG_SIZE//2)
    plt.plot(t, f, color='black', linewidth=2)
    plt.axis('off')
    plt.savefig(os.path.join(out_folder, f"vehicle_{idx}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    setup_dirs()

    print(f"\n Generating {NUM_TRAIN} human TRAIN images...")
    for i in range(NUM_TRAIN):
        generate_human_spectrogram(i, os.path.join(TRAIN_DIR, "human"), seed=100 + i)

    print(f"\n Generating {NUM_TRAIN} vehicle TRAIN images...")
    for i in range(NUM_TRAIN):
        generate_vehicle_spectrogram(i, os.path.join(TRAIN_DIR, "vehicle"), seed=200 + i)

    print(f"\n Generating {NUM_TEST} human TEST images...")
    for i in range(NUM_TEST):
        generate_human_spectrogram(i, os.path.join(TEST_DIR, "human"), seed=300 + i)

    print(f"\n Generating {NUM_TEST} vehicle TEST images...")
    for i in range(NUM_TEST):
        generate_vehicle_spectrogram(i, os.path.join(TEST_DIR, "vehicle"), seed=400 + i)

    print("\n All synthetic spectrograms generated successfully!")

if __name__ == "__main__":
    main()
