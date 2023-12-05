import os
import tarfile

from tqdm import tqdm

base = "extracted"
tar_filename = "all_biounits_1June2023.tar.gz"
with tarfile.open(tar_filename, "r:gz") as file:
    paths = file.getnames()
    data = {}
    for path in tqdm(paths):
        if "." in path:
            print(path)
            data[path] = file.extractfile(path).read()


# We have to remove the tar file as the TPU doesn't have enough disk space for both
os.system(f"rm {tar_filename}")
for path in tqdm(data):
    os.makedirs(os.path.split(os.path.join(base, path))[0], exist_ok=True)
    with open(os.path.join(base, path), "wb") as f:
        f.write(data[path])
