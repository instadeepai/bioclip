import os
from typing import NamedTuple, Tuple

import numpy as np

from bio_clip.data import residue_constants


class ProteinStructureSample(NamedTuple):
    chain_id: str
    nb_residues: int
    aatype: np.ndarray  # of type bool with shape (nb_residues, 21),
    # One-hot representation of the input amino acid sequence (20 amino acids + unknown)
    # with the residues indexed according to residue_contants.RESTYPES
    atom37_positions: np.ndarray  # of type float32 with shape (nb_residues, 37, 3),
    # atom37 representations of the 3D structure of the protein
    atom37_gt_exists: np.ndarray  # of type bool with shape (nb_residues, 37), Mask
    # denoting whether the corresponding atom's position was specified in the
    # pdb databank entry.
    atom37_atom_exists: np.ndarray  # of type bool with shape (nb_residues, 37), Mask
    # denoting whether the corresponding atom exists for each residue in the atom37
    # representation.
    resolution: float  # experimental resolution of the 3D structure as specified in the
    # pdb databank entry if available, otherwise 0.
    pdb_cluster_size: int  # size of the cluster in PDB this sample belongs to, 1 if not
    # available

    @classmethod
    def from_file(cls, filepath: str) -> "ProteinStructureSample":
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"{filepath} does not exist")

        with open(filepath, "rb") as file:
            dict_representation = np.load(file, allow_pickle=True)[()]

        return cls(**dict_representation)

    def to_file(self, filepath: str) -> None:
        assert os.path.exists(os.path.dirname(filepath))
        dict_representation = self._asdict()
        np.save(
            filepath,
            dict_representation,
        )

    def get_missing_backbone_coords_mask(self) -> np.ndarray:
        return ~(
            self.atom37_gt_exists[:, residue_constants.CA_INDEX]
            & self.atom37_gt_exists[:, residue_constants.N_INDEX]
            & self.atom37_gt_exists[:, residue_constants.C_INDEX]
        )

    def get_local_reference_frames(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ca_coords = self.atom37_positions[:, residue_constants.CA_INDEX]
        n_coords = self.atom37_positions[:, residue_constants.N_INDEX]
        c_coords = self.atom37_positions[:, residue_constants.C_INDEX]

        any_missing_coords = self.get_missing_backbone_coords_mask()

        x_axis = n_coords - ca_coords
        x_axis_norm = np.where(any_missing_coords, 1.0, np.linalg.norm(x_axis, axis=-1))
        assert np.all(x_axis_norm > 1e-3)
        x_axis /= np.expand_dims(x_axis_norm, axis=-1)

        z_axis = np.cross(x_axis, c_coords - ca_coords)
        z_axis_norm = np.where(any_missing_coords, 1.0, np.linalg.norm(z_axis, axis=-1))
        assert np.all(z_axis_norm > 1e-3)
        z_axis /= np.expand_dims(z_axis_norm, axis=-1)

        y_axis = np.cross(z_axis, x_axis)
        assert x_axis.shape == y_axis.shape == z_axis.shape
        return (x_axis, y_axis, z_axis)


def onehot_to_sequence(one_hot_encoding: np.ndarray) -> str:
    """
    Maps a one-hot encoding to a sequence of amino acids

    Args:
        one_hot_encoding: np.array of type np.bool with shape (*, 21)

    Returns:
      The amino acid sequence
    """
    assert len(one_hot_encoding.shape) == 2
    assert one_hot_encoding.shape[1] == residue_constants.NUM_RESTYPES, str(
        one_hot_encoding.shape
    )
    assert np.all(np.sum(one_hot_encoding.astype(np.uint16), axis=1) == 1)
    residue_id_encoding = np.where(one_hot_encoding)[1]
    return "".join(
        [residue_constants.RESTYPES[residue_id] for residue_id in residue_id_encoding]
    )
