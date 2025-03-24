import numpy as np
import matplotlib.pyplot as plt


def visualize_and_save_spectrogram(file_path, save_path=None):
    """
    Visualize and optionally save a spectrogram file.

    Args:
        file_path (str): Path to the saved .npy spectrogram file.
        save_path (str, optional): Path to save the spectrogram image (e.g., 'output.png').
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

    # Save the spectrogram to a file if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Spectrogram saved to {save_path}")

    plt.show()


# Example usage
spectrogram_path = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/Spectograms_classes/ang/Ses01F_impro01_F012_0.png'
output_image_path = 'C:/Users/oskar/PycharmProjects/pracaDyplomowa/data/IEMOCAP_data/Spectograms/example.png'

visualize_and_save_spectrogram(spectrogram_path, output_image_path)
