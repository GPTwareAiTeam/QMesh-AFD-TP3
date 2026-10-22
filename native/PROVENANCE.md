# Provenance of the vendored AFD transport sources

This directory makes the AFD-TP3 native transport self-contained so a fresh
checkout can build it without a sibling working tree.

## Origin

- Project: QMesh native verbs/AFD transport (same project as this repository)
- Private working tree: branch `qmesh-hybrid-hotpath`
- Commit: `b88ec90fd2465d647dd250f7282c7a3716584c26`
  (`afd: vectorize mapped combine and large-row pack publication`)
- Public remote: none. The origin repository has no configured git remote, so
  these files are vendored rather than referenced as a fetchable dependency.

The four `qmesh_afd_seam*` files are byte-for-byte copies; their hashes match
the originals. `qmesh_afd_edge.cpp` and `qmesh_afd_peer.cpp` were copied and
then public-sanitized: host-specific RDMA device names and private management
IPs were replaced by generic/loopback examples, and both programs now require
an explicit device and management IP. Executable code is otherwise unchanged;
the launchers already pass explicit values. Hashes below are of the files in
this directory.

| File | SHA-256 (this directory) | Origin SHA-256 |
| --- | --- | --- |
| `qmesh_afd_peer.cpp` | `d94720c578cdc6ed7249e98b4e7602f5ae98d6ff0a9d932029c22158ba0dcfb5` | `be3a319638178cdd8220730dd3cdbcc80ef49f54f1ae8341e8ec49e14c0d0281` |
| `qmesh_afd_edge.cpp` | `2d1633dbfb7de8583504828c20349d06c8e0b6f714f9ac98487eaa70f30e144b` | `fdabc56806735ac23c92ddcae308be0558edb92c96eeb26251b92a213b079031` |
| `qmesh_afd_seam.cu` | `240992132ca352ae5f4f042c16e24702c776926ca6578fbb129fc39baff2a4ca` | same |
| `qmesh_afd_seam_host.inc` | `6289ec5cd33ba2e3cda28e9f7e304e2bcf1dd936ebf796a831b6618aa74798b8` | same |
| `qmesh_afd_seam_api.inc` | `488ceac0719aa6a8009e4d96c2a9c14f5e6981a8b1d95df995b7e5420ce75d2f` | same |
| `qmesh_afd_seam_py.inc` | `08f94c3931210cd289b6185381e48c9447297175a1a551f9f1bddcd6469e6eec` | same |

Include closure: `qmesh_afd_edge.cpp` is standalone. `qmesh_afd_peer.cpp`
textually includes `qmesh_afd_seam.cu`, `qmesh_afd_seam_host.inc`,
`qmesh_afd_seam_api.inc`, and `qmesh_afd_seam_py.inc`; together they are the
whole extension. No other project source file is required.

## License

The origin tree at that commit contains no `LICENSE`, `COPYING`, or `NOTICE`
file, and the six sources carry no copyright or SPDX header. They are custom,
first-party QMesh work. The repository owner publishes them under the root MIT
license text (`../LICENSE`), whose copyright line is
`Copyright (c) 2026 QMesh contributors`. That collective line covers the QMesh
distribution; it does not name individual authors. The origin tree supplies no
separate license text or grant.

The build links system libraries (`libibverbs` from rdma-core and the CUDA
runtime). Their licenses come from the installed packages; no third-party source
or binary is redistributed here.

## Retired `wire/qmesh_afd_edge.cpp`

The old snapshot (wire row capacity 8, per-layer slots) was removed after every
script moved to `native/` and no active consumer remained; it is recoverable
from git history. This directory holds the source used by `build.sh` (row
capacity 4096, one shared in/out slot pair).
