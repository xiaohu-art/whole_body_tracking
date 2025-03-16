import os
import numpy as np
import torch


class RealTrajLoader:
    """
    Helper class to load and sample motion data from multiple NumPy (.npz) files in a folder.
    """

    def __init__(self, folder_path: str, device: torch.device | str) -> None:
        """
        Load and append motion data from all npz files in the folder, record trajectory counts,
        and store the number of frames for each trajectory.

        Args:
            folder_path: Path to the folder containing npz files.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the folder path is invalid or no npz files are found.
        """
        # Verify the folder exists
        assert os.path.isdir(folder_path), f"Invalid folder path: {folder_path}"

        # List all npz files in the folder
        file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npz")]
        assert len(file_list) > 0, f"No npz files found in folder: {folder_path}"

        q_list, v_list, a_list = [], [], []
        self.frames_per_traj = []  # List to store the number of frames in each trajectory
        joint_names = None

        # Load each file, accumulate data, and record the frame count for each trajectory
        for file in sorted(file_list):  # Sorting for consistency (optional)
            data = np.load(file)
            num_frames_in_file = data["q"].shape[0]
            self.frames_per_traj.append(num_frames_in_file)

            # For the first file, store joint_names; ensure consistency for the others.
            if joint_names is None:
                joint_names = data["joint_names"].tolist()
            else:
                if joint_names != data["joint_names"].tolist():
                    raise ValueError(f"Mismatch in joint names in file {file}.")

            q_list.append(data["q"])
            v_list.append(data["v"])
            a_list.append(data["a"])

        # Concatenate arrays along the frame dimension (axis=0)
        q_concat = np.concatenate(q_list, axis=0)
        v_concat = np.concatenate(v_list, axis=0)
        a_concat = np.concatenate(a_list, axis=0)

        self.device = device
        self._joint_names = joint_names
        self._q = torch.tensor(q_concat, dtype=torch.float32, device=self.device)
        self._v = torch.tensor(v_concat, dtype=torch.float32, device=self.device)
        self._a = torch.tensor(a_concat, dtype=torch.float32, device=self.device)
        self.num_frames = self._q.shape[0]
        self.num_trajectories = len(file_list)

        # Store frames per trajectory as a torch tensor for efficient indexing.
        self._frames_per_traj = torch.tensor(self.frames_per_traj, dtype=torch.int64, device=self.device)

        # Compute cumulative frame offsets for each trajectory.
        traj_offsets = [0]
        for frames in self.frames_per_traj:
            traj_offsets.append(traj_offsets[-1] + frames)
        self._traj_offsets = torch.tensor(traj_offsets, dtype=torch.int64, device=self.device)

        print(f"Real trajectory loaded from folder '{folder_path}':")
        print(f"  - Number of trajectories: {self.num_trajectories}")
        print(f"  - Total frames: {self.num_frames}")
        print(f"  - Frames per trajectory: {self.frames_per_traj}")

    @property
    def joint_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._joint_names

    @property
    def num_joints(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._joint_names)

    def sample_frame(self, traj_ids: torch.Tensor, frame_offsets: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the position, velocity, and acceleration for the given trajectory IDs and local frame offsets.
        Internally, this method computes the corresponding global indices.

        Args:
            traj_ids: Tensor of shape (N,) containing trajectory indices.
            frame_offsets: Tensor of shape (N,) containing the local frame indices within each trajectory.

        Returns:
            A tuple (q, v, a) for the corresponding frames.
        """
        global_indexes = self._traj_offsets[traj_ids] + frame_offsets
        return self._q[global_indexes], self._v[global_indexes], self._a[global_indexes]

    def sample_indexes(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample global frame indexes uniformly from the entire concatenated motion data,
        then determine the corresponding trajectory IDs and local frame offsets.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            A tuple (traj_ids, frame_offsets), where each is a tensor of shape (num_samples,).
        """
        # Sample global frame indexes uniformly
        global_indexes = torch.randint(0, self.num_frames, (num_samples,), device=self.device)
        # Compute trajectory IDs using torch.searchsorted:
        traj_ids = torch.searchsorted(self._traj_offsets, global_indexes, right=True) - 1
        # Compute local frame offsets for each global index
        frame_offsets = global_indexes - self._traj_offsets[traj_ids]
        return traj_ids, frame_offsets

    def get_end_mask(self, traj_ids: torch.Tensor, frame_offsets: torch.Tensor) -> torch.Tensor:
        """
        Provides a boolean mask indicating which environments (or commands) have reached the end
        of their assigned trajectory and need to be resampled.

        Args:
            traj_ids: Tensor of shape (N,) containing trajectory indices for each environment.
            frame_offsets: Tensor of shape (N,) containing the current local frame index
                           (relative to the corresponding trajectory).

        Returns:
            A boolean tensor of shape (N,). Each entry is True if the corresponding frame offset
            has reached (or exceeded) the number of frames in that trajectory and thus needs resampling.
        """
        max_frames = self._frames_per_traj[traj_ids]
        # Reverse the mask: True indicates that the frame_offset is NOT valid and needs resampling.
        mask = frame_offsets >= max_frames
        return mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder containing npz trajectory files")
    args = parser.parse_args()

    traj = RealTrajLoader(args.folder, "cpu")

    # Example usage of sample_indexes and sample_frame:
    traj_ids, frame_offsets = traj.sample_indexes(50)
    print("- Sampled trajectory IDs:", traj_ids)
    print("- Sampled frame offsets:", frame_offsets)
    # print the probability of each trajectory being sampled
    print("- Probability of sampling each trajectory:")
    print(torch.bincount(traj_ids) / len(traj_ids))
