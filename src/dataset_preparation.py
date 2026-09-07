import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_quiver, plot_pressure


def prepare_dataset(dataset_dir, out_file, dimensions=(64, 64), with_coords=False):
    height, width = dimensions

    files = os.listdir(dataset_dir)
    files = [f for f in files if f.endswith('.dat')]
    files.sort()

    coords_file = pd.read_csv(f'{dataset_dir}/coordinates.csv')
    coords_array = coords_file.drop(columns=['cellI']).to_numpy()

    inputs = []
    outputs = []
    for f in files:
        print(f)
        df = pd.read_csv(f'{dataset_dir}/{f}').drop(columns=['cellI'])
        if with_coords:
            df['x'] = coords_file['x']
            df['y'] = coords_file['y']
            df['z'] = coords_file['z']
        df = df.to_numpy().T.reshape(-1, height, width)

        inputs.append(df[4:])
        outputs.append(df[0:4])

    # Verify shapes
    print(f"Input shape: {np.array(inputs).shape}")
    print(f"Output shape: {np.array(outputs).shape}")

    # Visualize a sample
    sample_input = inputs[0]
    sample_output = outputs[0]

    plt.imshow(sample_input[0], cmap='gray', origin='lower')

    plot_quiver(sample_output, coords_array)
    plot_pressure(sample_output, coords_array)

    # Create a test split and put it aside
    test_size = int(0.1 * len(inputs))
    test_inputs = np.array(inputs[-test_size:])
    test_outputs = np.array(outputs[-test_size:])

    # Save
    inputs = np.array(inputs[:-test_size])
    outputs = np.array(outputs[:-test_size])
    np.savez(out_file, inputs=inputs, outputs=outputs)
    np.savez(out_file.replace('.npz', '_test.npz'), inputs=test_inputs, outputs=test_outputs)

    print(f"Training set saved to {out_file} with {inputs.shape[0]} samples.")
    print(f"Test set saved to {out_file.replace('.npz', '_test.npz')} with {test_inputs.shape[0]} samples.")


def attach_coordinates_to_inputs(
        source_npz_path: str,
        coords_csv_path: str,
        output_npz_path: str,
        spatial_size: int = 64
) -> None:
    """
    Loads a dataset containing 'inputs' and 'outputs', appends spatial coordinate
    channels (x, y, z) to the 'inputs' array only, and saves the result.

    The 'outputs' array is preserved without modification. The coordinates are
    reconstructed to match the specific transformation logic:
    DataFrame -> to_numpy() -> Transpose -> Reshape.

    Args:
        source_npz_path: Path to the existing .npz file. Must contain keys
                         'inputs' and 'outputs'.
        coords_csv_path: Path to the CSV file containing 'x', 'y', 'z' columns.
        output_npz_path: Destination path for the new .npz file.
        spatial_size: The height/width of the grid (default 64).

    Raises:
        KeyError: If 'inputs' or 'outputs' keys are missing from the source file.
        ValueError: If spatial dimensions mismatch between CSV and target grid.
    """

    # Load the coordinates with explicit column selection to guarantee channel order
    coords_df = pd.read_csv(coords_csv_path)[['x', 'y', 'z']]

    # Verify that the CSV length aligns with the target spatial resolution
    expected_pixels = spatial_size * spatial_size
    if len(coords_df) != expected_pixels:
        raise ValueError(
            f"Coordinate CSV length ({len(coords_df)}) does not match target "
            f"grid dimensions ({spatial_size}x{spatial_size}={expected_pixels})."
        )

    # Reconstruct the coordinate grid using the original dataset generation logic:
    # 1. Convert to (Pixels, 3) -> 2. Transpose to (3, Pixels) -> 3. Reshape to (3, H, W)
    coords_grid = coords_df.to_numpy().T.reshape(-1, spatial_size, spatial_size)

    # Open the source archive
    with np.load(source_npz_path) as data:
        # Validate existence of required keys
        if 'inputs' not in data or 'outputs' not in data:
            raise KeyError(
                f"Source file must contain 'inputs' and 'outputs'. "
                f"Found: {list(data.keys())}"
            )

        inputs_array = data['inputs']
        outputs_array = data['outputs']

    # Extract dimensions from the inputs array
    # Expected shape: (Batch_Size, Channels, Height, Width)
    n_samples, n_channels, height, width = inputs_array.shape

    # Ensure the input image size matches the coordinate grid size
    if height != spatial_size or width != spatial_size:
        raise ValueError(
            f"Input spatial dims ({height}x{width}) do not match "
            f"requested spatial_size ({spatial_size})."
        )

    print(f"Original Inputs Shape: {inputs_array.shape}")

    # Prepare coordinates for broadcasting:
    # 1. Expand dims to (1, 3, H, W)
    # 2. Tile along the batch axis to match (N, 3, H, W)
    coords_expanded = np.expand_dims(coords_grid, axis=0)
    coords_tiled = np.tile(coords_expanded, (n_samples, 1, 1, 1))

    # Concatenate along the channel axis (axis 1)
    # New inputs shape: (N, n_channels + 3, H, W)
    new_inputs = np.concatenate([inputs_array, coords_tiled], axis=1)

    print(f"New Inputs Shape:      {new_inputs.shape}")
    print(f"Outputs Shape:         {outputs_array.shape} (Unchanged)")

    # Save both arrays using the original keys 'inputs' and 'outputs'
    np.savez_compressed(
        output_npz_path,
        inputs=new_inputs,
        outputs=outputs_array
    )
    print(f"Successfully saved to: {output_npz_path}")


if __name__ == "__main__":
    # dim = 256
    # trait = ""
    # dataset_dir = f"../working/mixer_{dim}{trait}"
    # out_file = f"../data/mixer_{dim}{trait}.npz"
    # prepare_dataset(dataset_dir, out_file, dimensions=(dim, dim), with_coords=False)
    # print("Dataset preparation complete.")

    attach_coordinates_to_inputs(
        source_npz_path="./data/mixer_64.npz",
        coords_csv_path="./data/coordinates_64.csv",
        output_npz_path="./data/mixer_64_with_coords.npz",
        spatial_size=64,
    )
