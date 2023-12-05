import numpy as np

# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
# There are 21 residue types.
RESTYPES = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",  # unknown type
]
NUM_RESTYPES = len(RESTYPES)  # 21
RESTYPES_ORDER = {restype: i for i, restype in enumerate(RESTYPES)}
RESTYPE_1TO3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
RESTYPE_3TO1 = {v: k for k, v in RESTYPE_1TO3.items()}

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
ATOM_TYPES = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
ATOM_ORDER = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)  # := 37.

CA_INDEX = ATOM_TYPES.index("CA")
N_INDEX = ATOM_TYPES.index("N")
C_INDEX = ATOM_TYPES.index("C")
CB_INDEX = ATOM_TYPES.index("CB")


def create_pesto_element_tokeniser_from_atom37():
    # construct q from atom37_mask and tokens...
    # This will be a vector which maps from atom37 index to element token
    ATOM_TYPES2ELEM = {
        "N": "N",
        "CA": "C",
        "C": "C",
        "CB": "C",
        "O": "O",
        "CG": "C",
        "CG1": "C",
        "CG2": "C",
        "OG": "O",
        "OG1": "O",
        "SG": "S",
        "CD": "C",
        "CD1": "C",
        "CD2": "C",
        "ND1": "N",
        "ND2": "N",
        "OD1": "O",
        "OD2": "O",
        "SD": "S",
        "CE": "C",
        "CE1": "C",
        "CE2": "C",
        "CE3": "C",
        "NE": "N",
        "NE1": "N",
        "NE2": "N",
        "OE1": "O",
        "OE2": "O",
        "CH2": "C",
        "NH1": "N",
        "NH2": "N",
        "OH": "O",
        "CZ": "C",
        "CZ2": "C",
        "CZ3": "C",
        "NZ": "N",
        "OXT": "O",
    }
    # find the element from the PDB atom type name
    # find the PeSTo index for that element

    # standard elements (sorted by aboundance) (32)
    pesto_std_elements = [
        "C",
        "O",
        "N",
        "S",
        "P",
        "Se",
        "Mg",
        "Cl",
        "Zn",
        "Fe",
        "Ca",
        "Na",
        "F",
        "Mn",
        "I",
        "K",
        "Br",
        "Cu",
        "Cd",
        "Ni",
        "Co",
        "Sr",
        "Hg",
        "W",
        "As",
        "B",
        "Mo",
        "Ba",
        "Pt",
    ]
    pesto_std_elements2ix = {e: i for i, e in enumerate(pesto_std_elements)}

    map_matrix = np.zeros(NUM_ATOM_TYPES, dtype=np.int64)
    for i, atom_type in enumerate(ATOM_TYPES):
        map_matrix[i] = pesto_std_elements2ix[ATOM_TYPES2ELEM[atom_type]]
    return map_matrix, len(pesto_std_elements)


(
    ATOM37_IX_TO_PESTO_ELEM_IX,
    PESTO_TOTAL_1HOT_SIZE,
) = create_pesto_element_tokeniser_from_atom37()
