"""Default restore-only gcsfs retry patch for every 8x8x16 overlay (2026-10-23, ledger-manager; proof: ledger #643).
The generators that write 8x8x16 overlays (mk_x16.py, mk_h123.py, mk_a0dc.py) call apply(<overlay dir>) last.
It copies gcsfs_retry/sitecustomize.py into the overlay's launcher ConfigMap and adds PYTHONPATH=/launcher:/deps/src,
GCSFS_RETRIES=20, GCSFS_JITTER_S=4 and GCSFS_PIN=2026.2.0 to its env ConfigMap. Each host then logs
"GCSFS-RETRY patched", or exits 3 before any restore if gcsfs is not 2026.2.0. Watchers check every host with
gcsfs_retry/check_patched.py <jobset> --expect 256 (exit 2: a host is missing the line; exit 1: a request waited
over 360 s of the ~519 s patience). Numerics, data order and batch hashes are unchanged; the rendered hash is not."""
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
ENV = ("  # Restore-only gcsfs retry patch (gcsfs_patch.py; ledger #643): /launcher first so sitecustomize.py loads.\n"
       "  - PYTHONPATH=/launcher:/deps/src\n  - GCSFS_RETRIES=20\n  - GCSFS_JITTER_S=4\n  - GCSFS_PIN=2026.2.0\n")


def apply(dst):
    dst = pathlib.Path(dst)
    assert "cloud.google.com/gke-tpu-topology: 8x8x16" in (dst / "jobset.yaml").read_text(), f"{dst}: not an 8x8x16 overlay"
    shutil.copyfile(HERE / "gcsfs_retry" / "sitecustomize.py", dst / "sitecustomize.py")
    k = dst / "kustomization.yaml"
    t = k.read_text()
    if "  - sitecustomize.py" in t:
        return
    lines = t.split("\n")
    i = [n for n, l in enumerate(lines) if l.startswith("  - train_sft.sh")]
    assert len(i) == 1, f"{dst}: launcher ConfigMap file line"
    lines.insert(i[0] + 1, "  - sitecustomize.py # restore-only gcsfs retry patch (gcsfs_patch.py)")
    t = "\n".join(lines)
    assert t.count("  literals:\n") == 1 and "PYTHONPATH=" not in t and "GCSFS_" not in t, f"{dst}: env ConfigMap"
    k.write_text(t.replace("  literals:\n", "  literals:\n" + ENV, 1))
