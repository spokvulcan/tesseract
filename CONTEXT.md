# Context

Single source of truth for Tesseract's domain language and the *why* behind the
system. Keep this narrative current; record discrete decisions as ADRs in
`docs/adr/`.

> Stub — populate as the domain model stabilizes. See `ARCHITECTURE.md` for
> structure and `docs/adr/` for decisions.

## Domain

_TODO: the ubiquitous language — Agent, Session, Prefix Cache, TriAttention,
Package, Tool — defined in one place._

## Language

### Prefix cache snapshot lifecycle

**Snapshot State**:
The value attached to each radix-tree node encoding which tier(s) hold its KV-cache
snapshot and the snapshot's write phase. A six-case enum (`empty`, `ramOnly`,
`pendingWrite`, `pendingDropped`, `committed`, `ssdOnly`) that owns *both* the RAM
body and the **Snapshot Ref**; transitions return a **State Effect**.
_Avoid_: storage-ref lifecycle, slot, residency, the "five-state table".

**Snapshot Ref**:
The immutable on-disk identity of a snapshot — `snapshotID`, `partitionDigest`,
`tokenOffset`, `checkpointType`, `bytesOnDisk`. Carried as the payload of the
ref-bearing **Snapshot State** cases. Knows *what and where on disk*, never the
write phase (that is the enum case).
_Avoid_: SnapshotStorageRef, storage ref, descriptor.

**State Effect**:
The topology-only outcome a **Snapshot State** transition reports to its caller:
`settled`, `becameEmpty`, or `ignored(reason)`. `becameEmpty` carries **no payload**
and is the only signal that triggers the tree's self-heal (which then removes the node
only if topology allows). The dropped snapshot ID is **not** on `becameEmpty` — it
rides on `DropBodyResult` instead, because the common body drops (states 2/4) are
`settled`, not `becameEmpty`. `ignored(reason)` is propagated (not precondition'd) only
on the two forgiving SSD-writer callback edges (`commit`, `dropRef`).
_Avoid_: transition result, mutation outcome.

**dropRef**:
The forgiving SSD-writer callback edge for a pending **Snapshot Ref**. It is distinct
from committed-ref cleanup after a hydration failure; pending refs are dropped by the
writer callback, committed refs are cleared only after a failed SSD hydration.
_Avoid_: clear ref, remove ref.

**Committed Ref Cleanup**:
The strict cleanup edge after a failed SSD hydration of a committed **Snapshot Ref**.
It applies only to committed-ref states; pending refs are handled by `dropRef`, not
by hydration cleanup.
_Avoid_: generic ref clear, storage-ref cleanup.

**Explicit Ref Discard**:
The strict cleanup edge after the SSD backing has already been explicitly deleted or
cancelled, such as during leaf supersession. It may discard any ref-bearing
**Snapshot State** because the caller has already removed the backing snapshot.
_Avoid_: hydration cleanup, generic ref clear.

**canEvictNode**:
The load-bearing invariant query on **Snapshot State**: true iff the node holds no
live **Snapshot Ref**, i.e. removing the node structure cannot orphan an
SSD-resident snapshot. Distinct from `hasResidentBody` (the RAM-budget concept) —
a node may be node-removable yet still hold a useful RAM body.
_Avoid_: canRemove, isOrphanable.

**hasResidentBody**:
The RAM-budget query on **Snapshot State**: true iff the node holds a RAM body
that can be dropped to free memory. Distinct from `canEvictNode` — body eviction
may leave a node in place when a **Snapshot Ref** still pins an SSD-resident
snapshot.
_Avoid_: body-removable, resident snapshot.

> **Flagged ambiguity — "State".** `SnapshotState` is the prefix-cache lifecycle
> enum. It is unrelated to `HybridCacheSnapshot.LayerState` (MLX layer tensors) and
> to `@Observable` view/app state. When unqualified "state" is ambiguous, say
> "snapshot state" or "layer state".

**Example dialogue:**

> **Dev:** When the SSD writer's drop callback fires, who removes the node?
> **Expert:** Nobody removes it directly. The callback routes through the tree,
> the tree applies `dropRef`, and the *state* decides. If the drop leaves a RAM
> body, the state settles to `ramOnly` — node stays. If there's nothing left, the
> transition returns `becameEmpty` and the tree self-heals: it removes the node.
> **Dev:** So the eviction loop also checks `becameEmpty`?
> **Expert:** No — that loop is proactive, it picks LRU victims with
> `hasResidentBody`, then drops the RAM body. `becameEmpty` is reactive cleanup
> after a drop, and `canEvictNode` is the structural-removal invariant. Different
> predicates: a `ramOnly` node has both `hasResidentBody` and `canEvictNode`, while
> a committed node with RAM has `hasResidentBody` but not `canEvictNode`.

## Why

_TODO: the constraints that shape the system (fully offline, on-device MLX,
privacy-first, Apple Silicon) and the trade-offs they force._
