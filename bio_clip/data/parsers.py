import re
from typing import List, NamedTuple, Sequence, Tuple

from bio_clip.data.residue_constants import RESTYPES

SEQUENCE_REGEX_RAW = f"[{''.join(RESTYPES)}]+$"
SEQUENCE_REGEX = re.compile(SEQUENCE_REGEX_RAW)


class SequenceWithID(NamedTuple):
    id: str  # noqa: A003
    sequence: str


def parse_fasta(fasta_string: str) -> Tuple[bool, str, Sequence[SequenceWithID]]:
    """Parses a FASTA-formted string.

    Arguments:
      fasta_string (str): The string contents of a FASTA file.

    Returns:
      A 3-uple:
        - (bool) whether parsing the fasta string was a success
        - (str) an error message if parsing the string failed
        - (Sequence[SequenceWithID): list of SequenceWithID
            parsed from the string.
    """
    id_sequence_pairs: List[SequenceWithID] = []
    sequence_id = ""
    sequence = ""
    for line_nb, line in enumerate(fasta_string.splitlines() + [">end"]):
        if line.endswith("\n"):
            line = line[:-1]

        if line.startswith(">"):
            if len(line) == 1:
                return (
                    False,
                    f"Line #{line_nb} is incorrectly formatted, expect it to have "
                    "at least two characters",
                    id_sequence_pairs,
                )

            if line_nb > 0:
                if sequence == "":
                    return (
                        False,
                        f"Line #{line_nb - 1} is incorrectly formatted, expect a "
                        "sequence between two consecutive sequence headers",
                        id_sequence_pairs,
                    )

                id_sequence_pairs.append(
                    SequenceWithID(id=sequence_id, sequence=sequence)
                )
                sequence = ""

            sequence_id = line[1:].split(" ")[0 if line[1] != " " else 1]

        else:
            if sequence_id == "":
                return (
                    False,
                    f"The first line of a fasta file should start with the character >"
                    f"{sequence}",
                    id_sequence_pairs,
                )

            sequence += line
            if SEQUENCE_REGEX.match(line) is None:
                return (
                    False,
                    f"Sequence found line #{line_nb} does not pass the sequence regex: "
                    f"{sequence}",
                    id_sequence_pairs,
                )
    return (True, "", id_sequence_pairs)
