"""Read single members out of a huge remote zip with HTTP range requests (no full download).

ASL Citizen ships as one 42.8 GB zip; the split CSVs that define the released models' label order are a few hundred
KB inside it. zipfile only needs seek/read, so a range-reading file object lets it parse the central directory and
inflate just the members asked for.

Run: uv run python -m training.remote_zip --list
     uv run python -m training.remote_zip ASL_Citizen/splits/train.csv ...   (writes under data/asl_citizen/)
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

ASL_CITIZEN_ZIP = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "asl_citizen"
BLOCK = 1 << 20  # fetch at least 1 MB per request: zipfile issues many small reads


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url, self.pos, self.cache = url, 0, {}
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges") != "bytes":
                raise OSError(f"{url} does not support range requests")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            lo, hi = i * BLOCK, min((i + 1) * BLOCK, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.cache[i] = r.read()
        return self.cache[i]

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        out = bytearray()
        while self.pos < end:
            b = self._block(self.pos // BLOCK)
            off = self.pos % BLOCK
            take = b[off:off + end - self.pos]
            out += take
            self.pos += len(take)
        return bytes(out)

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="*")
    ap.add_argument("--list", action="store_true", help="list non-video members")
    ap.add_argument("--url", default=ASL_CITIZEN_ZIP)
    a = ap.parse_args()
    z = zipfile.ZipFile(HttpRangeFile(a.url))
    if a.list:
        for i in z.infolist():
            if not i.filename.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
                print(f"{i.file_size:>12}  {i.filename}")
        return
    for m in a.members:
        dst = OUT / Path(m).relative_to(Path(m).parts[0])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(m))
        print(f"wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
