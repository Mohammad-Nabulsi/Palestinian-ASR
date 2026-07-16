import sys
import time
import nbformat
from nbclient import NotebookClient

nb_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else nb_path

nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, kernel_name="asr_data_prep", timeout=None)

start = time.time()
print(f"Executing {nb_path} ...", flush=True)
client.execute()
print(f"Done in {time.time() - start:.1f}s", flush=True)

nbformat.write(nb, out_path)
print(f"Wrote {out_path}", flush=True)
