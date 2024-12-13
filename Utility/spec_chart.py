import numpy as np
import matplotlib.pyplot as plt


def visualize_spectrogram(file_path):
    """
    Visualize a saved spectrogram file.

    Args:
        file_path (str): Path to the saved .npy spectrogram file.
    """
    # Load the spectrogram
    spectrogram = np.load(file_path)

    # Display the spectrogram
    plt.figure(figsize=(10, 5))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Amplitude (log scale)')
    plt.title('Spectrogram')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.show()


# Example usage
visualize_spectrogram('C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/Spectograms/Ses01F_impro01/Ses01F_impro01_F000_neu/Ses01F_impro01_F000_neu_0.npy')
