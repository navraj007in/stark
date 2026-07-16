"""Gate 7 (G7-01) — fetch and verify the frozen Tiny YOLOv2 workload.

Downloads the pinned Tiny YOLOv2 ONNX model (MIT, ONNX Model Zoo) and its
bundled ONNX-authored test data (input_0.pb / output_0.pb) into the gitignored
scratch dir, verifies each against a pinned SHA-256, and extracts the test
tensors. The model and test tensors are NOT committed (fetch-by-checksum, as in
Gate 5); only this script and the recorded checksums are tracked.

Run with `--print-hashes` to compute and print digests without verifying (used
once to establish the pins). Normal runs verify and exit non-zero on any
mismatch.

Usage:
    fetch-artifacts.py [--print-hashes]
"""

import hashlib
import os
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.abspath(os.path.join(HERE, "..", "..", "tmp", "gate7"))

# Canonical source: the onnx/models Git-LFS media endpoint (authoritative).
BASE = (
    "https://media.githubusercontent.com/media/onnx/models/main/"
    "validated/vision/object_detection_segmentation/tiny-yolov2/model"
)
MODEL_URL = f"{BASE}/tinyyolov2-8.onnx"
TESTDATA_URL = f"{BASE}/tinyyolov2-8.tar.gz"
# Reference image for real detections (PyTorch Hub sample, BSD-3-Clause repo).
IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"

# Pinned SHA-256 digests (lowercase hex). Established via --print-hashes.
EXPECTED = {
    "tinyyolov2-8.onnx": "583fb7fdc948435ceac9fa82efc7708701efe8382a859a3dd46526b155f5f2ae",
    "test_data_set_0/input_0.pb": "7df6dfefe116d2024a2678a3b015b98256ab8e15205b6b67e5291c8e7889cb5c",
    "test_data_set_0/output_0.pb": "59bd363150b98190fe3957e98905a645feec39b16c7f543e8e96d9e8e595c727",
    "dog.jpg": "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    if os.path.exists(path):
        return
    print(f"downloading {url}\n        -> {path}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "stark-gate7/1.0"})
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def extract_testdata(tar_path):
    """Extract the two test tensors from the model tarball into DEST."""
    wanted = ("input_0.pb", "output_0.pb")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            base = os.path.basename(member.name)
            if base in wanted and "test_data_set_0" in member.name:
                member.name = os.path.join("test_data_set_0", base)
                tar.extract(member, DEST)


def main():
    print_mode = "--print-hashes" in sys.argv
    os.makedirs(DEST, exist_ok=True)

    model_path = os.path.join(DEST, "tinyyolov2-8.onnx")
    tar_path = os.path.join(DEST, "tinyyolov2-8.tar.gz")
    image_path = os.path.join(DEST, "dog.jpg")
    download(MODEL_URL, model_path)
    download(TESTDATA_URL, tar_path)
    download(IMAGE_URL, image_path)
    extract_testdata(tar_path)

    files = {
        "tinyyolov2-8.onnx": model_path,
        "test_data_set_0/input_0.pb": os.path.join(DEST, "test_data_set_0", "input_0.pb"),
        "test_data_set_0/output_0.pb": os.path.join(DEST, "test_data_set_0", "output_0.pb"),
        "dog.jpg": image_path,
    }

    digests = {name: sha256(p) for name, p in files.items()}

    if print_mode:
        for name, d in digests.items():
            print(f'    "{name}": "{d}",')
        return

    failures = []
    for name, got in digests.items():
        want = EXPECTED[name]
        status = "ok" if got == want else "MISMATCH"
        print(f"{status:8} {name}  {got}")
        if got != want:
            failures.append(name)
    if failures:
        print(f"\nERROR: checksum mismatch for {failures}", file=sys.stderr)
        sys.exit(1)
    print(f"\nall artifacts verified in {DEST}")


if __name__ == "__main__":
    main()
