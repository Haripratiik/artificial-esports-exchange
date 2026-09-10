r"""Crash recovery: journal the exogenous inputs, replay them, rebuild the world.

Everything this exchange knows lives in memory. Accounts, API keys, positions,
working orders and the whole price series die with the process, which is the
last thing stopping anyone from leaving an algorithm running against it
overnight.

The fix here is not a database. It is the LMAX recipe, and this codebase makes
it unusually cheap. Martin Fowler, writing about exactly this situation: "It
would be possible to journal the output events too... In practice, however,
this isn't worthwhile. The business logic is deterministic and very fast, so
there's no gain from storing the results."
(https://martinfowler.com/articles/lmax.html). Aeron Cluster states the same
recipe in three parts: a snapshot, an ordered log of every input message, and
deterministic application logic.

**Why input journalling fits this exchange specifically.** ``sim/kernel.py``
seeds each agent from ``_stable_seed(seed, "agent", agent_id)``, and its draws
are independent of join order. Agent behaviour therefore *regenerates* on
replay from the seed alone. The only things the seed cannot regenerate are the
events that came from outside the simulation: a person taking a seat, an order
submitted, an order cancelled. Those, and only those, go in the journal.

The size argument is the one VMware made for its fault-tolerant hypervisor:
logging inputs is far smaller than logging state. One aggressive order that
sweeps 200 resting price levels is **one** journal record, not 200 fills, not
201 book updates, and not a new snapshot of two hundred accounts.

**The seed belongs in the header.** Replay reconstructs agent behaviour by
re-running the generators, so a journal that does not record the seed and the
market configuration it was built from replays into a different market and says
nothing about it. Pass them in ``metadata``. This module cannot check that for
you because it deliberately knows nothing about the venue.

---

**Record format.** Every record is a self-delimiting frame, so a tail torn by a
dying process is detectable rather than silently reinterpreted as the next
record::

    +--------+--------+----------+---------------+------+-----+---------+
    | length | crc32  | sequence | timestamp_ns  | klen | kind| payload |
    |  u32   |  u32   |   u64    |     i64       |  u8  | ascii| json   |
    +--------+--------+----------+---------------+------+-----+---------+
    \___ frame header, 8 bytes __/\____ body, `length` bytes ___________/

``crc32`` covers the length prefix as well as the body, so a corrupted length
is caught by the checksum rather than by wandering off into the next record.
Sequence numbers are monotonic and gapless; timestamps are the *simulated*
nanosecond at which the input was injected, and never decrease. The payload is
canonical JSON: sorted keys, no insignificant whitespace, ASCII escaped.

**No floats, anywhere.** ``_check_persistable`` rejects ``float``, ``Decimal``,
``tuple``, ``set`` and non-string dict keys before anything is written. Floats
because a persisted 0.1 is not 0.1 and this project's conservation check
returns an exact integer zero. ``Decimal`` and ``tuple`` because JSON would
quietly hand them back as ``str`` and ``list``: that is bug class 1 in
``CONTRIBUTING.md`` (a value crossing a boundary in a form its label does not claim,
one consumer compensating, every other consumer wrong), and a journal is the
widest boundary in the system. Money is integer minor units, time is integer
nanoseconds, prices are integer ticks.

---

**Recovery semantics**, following Redis, whose ``aof-load-truncated`` defaults
to on:

* A final record whose framing is incomplete at end of file is a **torn tail**.
  The process died between ``write`` and the next ``write``; the bytes that did
  land are dropped with a loud warning and everything before them is kept.
* A record whose bytes are all present but whose checksum fails is
  **corruption** and raises. So is a sequence gap and a timestamp that goes
  backwards. Mid-file damage is not recoverable by guessing, and guessing is
  how you get a plausible market that never happened.
* A **damaged length prefix is corruption, not a tear**, and the two look
  identical from the outside: both are a frame asking for more bytes than the
  file can supply. Getting it backwards is the worst outcome in this module,
  because a torn tail is dropped silently-by-design and the caller is told the
  load succeeded. Two things separate them. A length no legitimate record could
  have is refused before the read. A length the file cannot satisfy is checked
  against the stored checksum at the length the file *can* satisfy: if that
  validates, every byte of the record is present and it is the framing that is
  wrong. See ``_Reader._reject_damaged_length``.
* An ``engine_version`` that differs from the one in the header **refuses to
  replay**. This is the sharpest risk in the whole design and it is silent: if
  the number or order of RNG draws changes, or the PRNG is swapped, every
  earlier journal still replays cleanly into a different but entirely plausible
  state, and nobody notices. There is no checksum that can catch that, because
  every byte on disk is still correct. Only the version can.

Snapshots carry ``(state, last_applied_sequence, engine_version)``. The
sequence number is not optional decoration: Raft's snapshot metadata is
``lastIncludedIndex`` plus the state, and Flink's is "a pointer into each of the
data sources plus a copy of the state". Without it there is no way to know
where to resume the journal, and the snapshot is unloadable. Snapshots are
taken at a sequence boundary between events, and written to a temporary file
that is fsynced and then ``os.replace``d, so a half-written snapshot can never
be loaded. There is deliberately no fork-and-copy-on-write: Redis publishes
measured fork costs of 62ms for 6.9GB on a physical Xeon and 1460ms on an older
EC2 instance, and a stall of that size here is thousands of orders.

This module knows nothing about the venue, the market or the kernel. It moves
bytes and calls callbacks, which is what lets it be tested without building a
market.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import struct
import zlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, BinaryIO, Final, Literal

__all__ = [
    "FORMAT_VERSION",
    "JOURNAL_MAGIC",
    "KINDS",
    "KIND_CANCEL_ALL",
    "KIND_KEY_ISSUED",
    "KIND_ORDER_CANCELLED",
    "KIND_ORDER_SUBMITTED",
    "KIND_SEAT_JOINED",
    "SNAPSHOT_MAGIC",
    "Journal",
    "JournalCorruption",
    "JournalError",
    "JournalFormatError",
    "JournalTornTail",
    "JournalVersionMismatch",
    "Record",
    "Recovery",
    "ReplayResult",
    "Snapshot",
    "encode_record",
    "header_of",
    "load_snapshot",
    "read_records",
    "recover",
    "repair",
    "replay",
    "write_snapshot",
]

LOG = logging.getLogger(__name__)

# Eight bytes so the type of a file is decidable from its first read, and
# version-tagged so a future frame layout is a different magic rather than a
# subtly misparsed old one.
JOURNAL_MAGIC: Final = b"ARENAJN\x01"
SNAPSHOT_MAGIC: Final = b"ARENASN\x01"

FORMAT_VERSION: Final = 1

# The exogenous inputs. Everything else in this exchange is a consequence of
# these plus the seed, and consequences are not journalled.
KIND_SEAT_JOINED: Final = "seat_joined"
KIND_KEY_ISSUED: Final = "key_issued"
KIND_ORDER_SUBMITTED: Final = "order_submitted"
KIND_ORDER_CANCELLED: Final = "order_cancelled"
KIND_CANCEL_ALL: Final = "cancel_all"

#: The kinds this exchange produces today. ``append`` does not validate against
#: this set, only against the *shape* of a kind name, so whoever wires the
#: module in can add one without editing this file. The replay callback is what
#: must refuse a kind it does not understand: a callback that silently ignores
#: an unknown kind diverges from the recorded run and reports success.
KINDS: Final = frozenset(
    {
        KIND_SEAT_JOINED,
        KIND_KEY_ISSUED,
        KIND_ORDER_SUBMITTED,
        KIND_ORDER_CANCELLED,
        KIND_CANCEL_ALL,
    }
)

_KIND_RE: Final = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")

_FRAME_HEADER: Final = struct.Struct(">II")  # body length, crc32 over prefix+body
_RECORD_PREFIX: Final = struct.Struct(">QqB")  # sequence, timestamp_ns, kind length

# A corrupted length prefix must not be allowed to ask for a gigabyte-sized
# read. Nothing legitimate here is close: the largest real record is an order
# submission, which is a few hundred bytes.
_MAX_BODY_BYTES: Final = 8 * 1024 * 1024

_UINT64_MAX: Final = (1 << 64) - 1
_INT64_MAX: Final = (1 << 63) - 1

Durability = Literal["none", "flush", "fsync"]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class JournalError(Exception):
    """Base for every failure this module raises."""


class JournalFormatError(JournalError):
    """The bytes are not a journal, or its header is unreadable."""


class JournalCorruption(JournalError):
    """Damage that cannot be recovered by dropping a tail.

    A failed checksum on a record whose bytes are all present, a gap in the
    sequence, or a timestamp that moves backwards. Redis draws the line in the
    same place: a truncated tail is loaded anyway, and mid-file corruption is
    reported rather than repaired.
    """


class JournalTornTail(JournalError):
    """The final record is incomplete and the caller asked not to tolerate it."""


class JournalVersionMismatch(JournalError):
    """The journal was written by a different engine version.

    Raised rather than warned about, because the failure it prevents is
    invisible. Deterministic replay only reproduces the recorded run if the
    logic is unchanged; against changed logic it produces a *different*
    market that looks entirely normal, with the right accounts, plausible
    prices and no error anywhere.
    """


# ---------------------------------------------------------------------------
# values that may be persisted
# ---------------------------------------------------------------------------


def _check_persistable(value: Any, path: str = "payload") -> None:
    """Reject anything that would not come back out exactly as it went in.

    Two separate failures are being prevented here.

    *Floats.* This project's conservation check returns an exact integer zero
    and prices are integer ticks precisely so that no rounding decision is ever
    left to a binary fraction. A float in a journal record would reintroduce
    that error at the one place where it is permanent.

    *Silent type changes.* ``json`` turns a ``Decimal`` into whatever a custom
    encoder says, a ``tuple`` into a ``list``, and an integer dict key into a
    string. Every one of those round-trips without complaint and comes back a
    different type, which is bug class 1 in ``CONTRIBUTING.md`` verbatim: a value
    crossing a boundary in a form its label does not claim. Refusing at write
    time costs the caller one conversion and buys exact equality on replay.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        raise TypeError(
            f"{path}: no floats in a persisted record (got {value!r}). "
            "Money is integer minor units, time is integer nanoseconds, "
            "prices are integer ticks."
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_persistable(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path}: dict keys must be str, got {type(key).__name__} "
                    f"({key!r}); json would return it as a str and the "
                    "round-trip would not compare equal"
                )
            _check_persistable(item, f"{path}.{key}")
        return
    if isinstance(value, tuple):
        raise TypeError(
            f"{path}: use a list, not a tuple; a tuple round-trips as a list "
            "and the recovered value would not compare equal to the original"
        )
    if isinstance(value, Decimal):
        raise TypeError(
            f"{path}: convert a Decimal to integer minor units or integer "
            f"ticks before journalling it (got {value!r})"
        )
    raise TypeError(f"{path}: {type(value).__name__} is not persistable")


def _canonical_json(value: Any) -> bytes:
    """Serialize so equal content always yields equal bytes.

    Deliberately *not* ``arena.determinism.canonical_json``, even though the
    options match. That helper serves research aggregation and carries a
    ``default=`` hook that turns a ``Decimal`` into a string; this is a
    persisted wire format, and it must not be able to start accepting new types
    or change its escaping because somebody improved a research path. The four
    options below are the format.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Record:
    """One exogenous input, as it was recorded.

    ``timestamp_ns`` is *simulated* time at the moment of injection, not wall
    clock. Point-in-time recovery asks questions in the market's own clock
    ("replay to just before the halt"), and the market's clock is the kernel's.
    """

    sequence: int
    timestamp_ns: int
    kind: str
    payload: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "kind": self.kind,
            "payload": self.payload,
        }


def encode_record(sequence: int, timestamp_ns: int, kind: str, payload: Any) -> bytes:
    """Encode one record as a self-delimiting frame.

    Exposed so a test, a repair tool or a fuzzer can build frames without
    opening a file, and so the encoding has exactly one implementation.
    """
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise TypeError(f"sequence must be an int, got {type(sequence).__name__}")
    if sequence < 1 or sequence > _UINT64_MAX:
        raise ValueError(f"sequence out of range: {sequence}")
    if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
        raise TypeError(f"timestamp_ns must be an int, got {type(timestamp_ns).__name__}")
    if timestamp_ns < 0 or timestamp_ns > _INT64_MAX:
        raise ValueError(f"timestamp_ns out of range: {timestamp_ns}")
    if not isinstance(kind, str) or not _KIND_RE.match(kind):
        raise ValueError(
            f"kind must be lowercase ascii, 1 to 32 chars, got {kind!r}. "
            f"The kinds this exchange records today are {sorted(KINDS)}."
        )

    _check_persistable(payload)
    kind_bytes = kind.encode("ascii")
    body = _RECORD_PREFIX.pack(sequence, timestamp_ns, len(kind_bytes))
    body += kind_bytes + _canonical_json(payload)
    if len(body) > _MAX_BODY_BYTES:
        raise ValueError(f"record body of {len(body)} bytes exceeds {_MAX_BODY_BYTES}")

    length = struct.pack(">I", len(body))
    # The checksum covers the length prefix too. Without that, a length flipped
    # from 40 to 41 would be caught only by whatever the next record's frame
    # header happened to decode to, which is a much later and much stranger
    # error than "record 812 failed its checksum".
    crc = zlib.crc32(length + body) & 0xFFFFFFFF
    return length + struct.pack(">I", crc) + body


def _decode_body(body: bytes, offset: int, label: str) -> Record:
    sequence, timestamp_ns, kind_len = _RECORD_PREFIX.unpack_from(body, 0)
    start = _RECORD_PREFIX.size
    if start + kind_len > len(body):
        raise JournalCorruption(
            f"{label}: record at offset {offset} claims a {kind_len} byte kind "
            f"but its body is only {len(body)} bytes"
        )
    try:
        kind = body[start : start + kind_len].decode("ascii")
        payload = json.loads(body[start + kind_len :].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as bad:
        # The checksum already passed, so these bytes are the bytes that were
        # written. Reaching here means they were written by something that does
        # not agree with this decoder, which is a format problem, not damage.
        raise JournalFormatError(
            f"{label}: record at offset {offset} passed its checksum but does "
            f"not decode: {bad}"
        ) from bad
    return Record(sequence=sequence, timestamp_ns=timestamp_ns, kind=kind, payload=payload)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _open_source(source: Any) -> tuple[BinaryIO, bool, str]:
    """Normalize a path, a bytes blob or an open stream into a stream.

    Returns ``(stream, we_opened_it, label)``. Accepting bytes is what lets the
    whole recovery path be tested without touching a filesystem.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(source)), True, "<bytes>"
    if hasattr(source, "read"):
        return source, False, getattr(source, "name", "<stream>")
    path = os.fspath(source)
    return open(path, "rb"), True, path


class _Reader:
    """Streaming frame reader. Distinguishes a torn tail from real damage."""

    def __init__(self, stream: BinaryIO, label: str) -> None:
        self._stream = stream
        self.label = label
        self.offset = 0
        self.intact_offset = 0
        self.torn_bytes = 0
        self.torn_offset: int | None = None
        self.header = self._read_header()
        self.intact_offset = self.offset

    # -- header ------------------------------------------------------------

    def _read(self, count: int) -> bytes:
        chunk = self._stream.read(count)
        self.offset += len(chunk)
        return chunk

    def _read_header(self) -> dict[str, Any]:
        magic = self._read(len(JOURNAL_MAGIC))
        if not magic:
            raise JournalFormatError(
                f"{self.label}: empty file. The process died before the header "
                "was written, so there is nothing to replay."
            )
        if magic != JOURNAL_MAGIC:
            raise JournalFormatError(
                f"{self.label}: not a journal (magic {magic!r}, "
                f"expected {JOURNAL_MAGIC!r})"
            )
        frame = self._read(_FRAME_HEADER.size)
        if len(frame) < _FRAME_HEADER.size:
            raise JournalFormatError(f"{self.label}: header frame is truncated")
        length, crc = _FRAME_HEADER.unpack(frame)
        if length > _MAX_BODY_BYTES:
            raise JournalFormatError(f"{self.label}: header claims {length} bytes")
        body = self._read(length)
        if len(body) < length:
            # No leniency here. A torn *header* leaves nothing behind it that
            # could be replayed, so there is no "everything before it" to save.
            raise JournalFormatError(f"{self.label}: header is truncated")
        if zlib.crc32(struct.pack(">I", length) + body) & 0xFFFFFFFF != crc:
            raise JournalCorruption(f"{self.label}: header checksum failed")
        try:
            header = json.loads(body.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as bad:
            raise JournalFormatError(f"{self.label}: header is not readable: {bad}") from bad
        if not isinstance(header, dict):
            raise JournalFormatError(f"{self.label}: header is not an object")
        if header.get("format") != FORMAT_VERSION:
            raise JournalFormatError(
                f"{self.label}: frame format {header.get('format')!r}, "
                f"this build reads {FORMAT_VERSION}"
            )
        for required in ("engine_version", "first_sequence"):
            if required not in header:
                raise JournalFormatError(f"{self.label}: header is missing {required!r}")
        return header

    # -- records -----------------------------------------------------------

    def records(self) -> Iterator[tuple[bytes, Record]]:
        """Yield ``(frame_bytes, record)`` until the journal ends or tears.

        The frame bytes come back with the record so a caller can hash exactly
        what it consumed, or copy a prefix of the journal, without re-encoding
        and hoping the encoder agrees with itself.
        """
        expected = int(self.header["first_sequence"])
        previous_timestamp = -1
        while True:
            start = self.offset
            frame = self._read(_FRAME_HEADER.size)
            if not frame:
                return  # clean end of file
            if len(frame) < _FRAME_HEADER.size:
                self._tear(start, len(frame))
                return
            length, crc = _FRAME_HEADER.unpack(frame)
            # Checked before the read, not after, and it decides the diagnosis.
            # An interrupted write leaves a *correct* length prefix followed by
            # too few bytes, because the length is written in the same call as
            # the body it describes. A length no legitimate record could have
            # is therefore damage, not truncation, and must not be reported as
            # a droppable tail, nor be allowed to ask for a gigabyte read.
            if length > _MAX_BODY_BYTES:
                raise JournalCorruption(
                    f"{self.label}: record at offset {start} claims {length} bytes, "
                    f"over the {_MAX_BODY_BYTES} byte limit; the length prefix is "
                    "damaged and the file cannot be read past this point"
                )
            body = self._read(length)
            if len(body) < length:
                # Short read. Either the record the process was in the middle of
                # writing when it died (expected, dropped with a warning), or
                # a length prefix damaged upward, which is corruption and must
                # not be dropped. The `> _MAX_BODY_BYTES` gate above only catches
                # the absurd end of that; a length flipped from 41 to 42 lands
                # here, and the naive reading calls it a torn tail. Measured on
                # the 8-record journal in tests/test_journal.py, bumping the
                # final record's 41-byte length to 42, to 1041 and to 4194345:
                # all three warned "truncated final record, dropping 49 trailing
                # bytes", returned the first 7 records, and reported success.
                #
                # The discriminator is the checksum, which covers the length
                # prefix. If the prefix was damaged, the record's real bytes are
                # all still on disk and the stored crc still describes them, so
                # re-checking against the length the file can actually supply
                # validates. A genuine tear is missing bytes that were never
                # written, so no length validates and it stays a tear.
                self._reject_damaged_length(start, length, crc, body)
                self._tear(start, len(frame) + len(body))
                return
            if zlib.crc32(struct.pack(">I", length) + body) & 0xFFFFFFFF != crc:
                raise JournalCorruption(
                    f"{self.label}: checksum failed for the record at offset "
                    f"{start}, whose {length} bytes are all present. This is "
                    "mid-file damage, not a torn tail, and replaying past it "
                    "would rebuild a market that never happened."
                )
            if length < _RECORD_PREFIX.size:
                raise JournalCorruption(
                    f"{self.label}: record at offset {start} is too short to "
                    f"hold a record prefix ({length} bytes)"
                )
            record = _decode_body(body, start, self.label)
            if record.sequence != expected:
                raise JournalCorruption(
                    f"{self.label}: sequence {record.sequence} at offset {start}, "
                    f"expected {expected}. A gap means an input was lost, and "
                    "replaying the rest would silently skip it."
                )
            if record.timestamp_ns < previous_timestamp:
                raise JournalCorruption(
                    f"{self.label}: record {record.sequence} is stamped "
                    f"{record.timestamp_ns}ns, behind the previous record's "
                    f"{previous_timestamp}ns. Simulated time never moves "
                    "backwards, and point-in-time recovery would be meaningless."
                )
            expected = record.sequence + 1
            previous_timestamp = record.timestamp_ns
            self.intact_offset = self.offset
            yield frame + body, record

    def _reject_damaged_length(self, start: int, length: int, crc: int, body: bytes) -> None:
        """Raise if a short read is a damaged length prefix rather than a tear.

        Called only when the frame claimed ``length`` bytes and the file could
        supply fewer. One extra ``crc32`` over bytes already in hand.

        A false positive needs a torn tail whose surviving bytes checksum to the
        stored crc at their own length, which is one chance in 2^32, and it fails
        toward refusing to load rather than toward loading damage silently.

        What this deliberately does not do is search for a valid record at every
        shorter length. That would catch a mid-file length damaged into swallowing
        the records behind it, but the checksum covers the length prefix, so each
        candidate needs its own pass over the body: O(n^2), which at the 8MB cap
        is 32TB of hashing to diagnose one bad frame. The single candidate below
        is the one that matters, because it is the case that reaches the end of
        the file and gets dropped as expected wear.
        """
        if len(body) < _RECORD_PREFIX.size:
            # Too short to be a record at all, so there is nothing to validate
            # against and a tear is the only reading left.
            return
        if zlib.crc32(struct.pack(">I", len(body)) + body) & 0xFFFFFFFF != crc:
            return
        raise JournalCorruption(
            f"{self.label}: record at offset {start} claims {length} bytes but the "
            f"file holds {len(body)}, and those {len(body)} bytes match the stored "
            "checksum. The length prefix is damaged, not the file truncated: the "
            "record is whole and its framing is wrong. Dropping it as a torn tail "
            "would discard an input that was fully written and report success."
        )

    def _tear(self, start: int, size: int) -> None:
        self.torn_offset = start
        self.torn_bytes = size


def header_of(source: Any) -> dict[str, Any]:
    """Read a journal's header without replaying it.

    Cheap enough to call before deciding whether recovery is even possible:
    the ``engine_version`` and the ``metadata`` that names the seed are both
    here, in the first few hundred bytes.
    """
    stream, owned, label = _open_source(source)
    try:
        return _Reader(stream, label).header
    finally:
        if owned:
            stream.close()


def read_records(
    source: Any,
    *,
    engine_version: str | None = None,
    tolerate_torn_tail: bool = True,
) -> Iterator[Record]:
    """Iterate a journal's records. Verifies framing, checksums and ordering.

    Provided for tools and tests. Recovery should go through :func:`replay`,
    which reports what it dropped instead of leaving the caller to notice.
    """
    stream, owned, label = _open_source(source)
    try:
        reader = _Reader(stream, label)
        if engine_version is not None:
            _check_version(reader.header["engine_version"], engine_version, label)
        for _frame, record in reader.records():
            yield record
        if reader.torn_offset is not None:
            message = _torn_message(label, reader)
            if not tolerate_torn_tail:
                raise JournalTornTail(message)
            LOG.warning("%s", message)
    finally:
        if owned:
            stream.close()


def _torn_message(label: str, reader: _Reader) -> str:
    return (
        f"{label}: truncated final record at offset {reader.torn_offset}; "
        f"dropping {reader.torn_bytes} trailing bytes. Everything up to offset "
        f"{reader.intact_offset} is intact. This is the expected shape of a "
        "process that died mid-write."
    )


def _check_version(found: str, expected: str, label: str) -> None:
    if found != expected:
        raise JournalVersionMismatch(
            f"{label}: journal was written by engine version {found!r}, this "
            f"build is {expected!r}. Refusing to replay. Deterministic replay "
            "regenerates agent behaviour by re-running the generators, so "
            "against changed logic it would not fail. It would rebuild a "
            "different market that looks completely plausible, with the right "
            "accounts and sensible prices, and nothing would report an error. "
            "Recover with a build of the original version, or start a new "
            "journal from a snapshot."
        )


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


class Journal:
    """Append-only writer for exogenous inputs.

    Not thread safe and deliberately not made so: the whole design assumes one
    ordered input stream, and a lock here would hide the fact that two threads
    appending is already a determinism bug upstream.
    """

    def __init__(
        self,
        stream: BinaryIO,
        *,
        engine_version: str,
        header: Mapping[str, Any],
        last_sequence: int,
        last_timestamp_ns: int,
        durability: Durability = "flush",
        path: str | None = None,
        owns_stream: bool = True,
    ) -> None:
        if durability not in ("none", "flush", "fsync"):
            raise ValueError(f"unknown durability {durability!r}")
        self._stream = stream
        self._owns_stream = owns_stream
        self.path = path
        self.engine_version = engine_version
        self.header = dict(header)
        self.durability: Durability = durability
        self._sequence = last_sequence
        self._last_timestamp_ns = last_timestamp_ns
        self._records = 0
        self._bytes = 0
        self._closed = False

    # -- construction ------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        *,
        engine_version: str,
        metadata: Mapping[str, Any] | None = None,
        created_ns: int = 0,
        first_sequence: int = 1,
        durability: Durability = "flush",
    ) -> Journal:
        """Start a new journal. Refuses to overwrite an existing file.

        ``metadata`` is where the kernel seed and the market build
        configuration go. Replay regenerates every agent's draws from that
        seed, so a journal without it records inputs to a market nobody can
        reconstruct. This module cannot check it, because it knows nothing
        about the venue.

        ``created_ns`` defaults to 0 rather than the wall clock on purpose. A
        wall-clock stamp in the header would make two journals of the same
        events differ in their first hundred bytes, and byte-identity of the
        encoding is the cheapest determinism check available.
        """
        header = _build_header(engine_version, metadata, created_ns, first_sequence)
        target = os.fspath(path)
        # "xb", not "wb": a journal is the only copy of what happened, and
        # opening the exchange twice against one file should fail loudly rather
        # than truncate the record of the first session.
        stream = open(target, "xb")
        stream.write(JOURNAL_MAGIC + _encode_frame(_canonical_json(header)))
        journal = cls(
            stream,
            engine_version=engine_version,
            header=header,
            last_sequence=first_sequence - 1,
            last_timestamp_ns=-1,
            durability=durability,
            path=target,
        )
        journal._sync()
        return journal

    @classmethod
    def in_memory(
        cls,
        *,
        engine_version: str,
        metadata: Mapping[str, Any] | None = None,
        created_ns: int = 0,
        first_sequence: int = 1,
    ) -> Journal:
        """A journal over a ``BytesIO``. For tests and for encoding fixtures."""
        header = _build_header(engine_version, metadata, created_ns, first_sequence)
        stream = io.BytesIO()
        stream.write(JOURNAL_MAGIC + _encode_frame(_canonical_json(header)))
        return cls(
            stream,
            engine_version=engine_version,
            header=header,
            last_sequence=first_sequence - 1,
            last_timestamp_ns=-1,
            durability="none",
            path=None,
        )

    @classmethod
    def open_for_append(
        cls,
        path: str | os.PathLike[str],
        *,
        engine_version: str,
        durability: Durability = "flush",
        repair_torn_tail: bool = True,
    ) -> Journal:
        """Reopen an existing journal to continue writing.

        Scans the whole file first, which is the only way to learn the last
        sequence number and the only way to be sure the tail is intact.
        Appending after a torn record would leave the garbage bytes wedged in
        the middle of the file, where the next replay classifies them as
        unrecoverable corruption rather than a droppable tail. So a torn tail
        is truncated here by default, the way Redis truncates a short AOF at
        load rather than refusing to start.

        The engine version is checked on the way in as well. A journal is only
        replayable if every record in it came from the same logic, so appending
        a new build's records to an old build's file would produce a file that
        can never be replayed at all.
        """
        target = os.fspath(path)
        with open(target, "rb") as probe:
            reader = _Reader(probe, target)
            _check_version(reader.header["engine_version"], engine_version, target)
            last_sequence = int(reader.header["first_sequence"]) - 1
            last_timestamp = -1
            for _frame, record in reader.records():
                last_sequence = record.sequence
                last_timestamp = record.timestamp_ns
            header = reader.header
            torn = reader.torn_offset is not None
            intact = reader.intact_offset
            if torn:
                message = _torn_message(target, reader)
                if not repair_torn_tail:
                    raise JournalTornTail(message)
                LOG.warning("%s", message)

        if torn:
            with open(target, "r+b") as fix:
                fix.truncate(intact)
                fix.flush()
                os.fsync(fix.fileno())

        stream = open(target, "r+b")
        stream.seek(0, os.SEEK_END)
        return cls(
            stream,
            engine_version=engine_version,
            header=header,
            last_sequence=last_sequence,
            last_timestamp_ns=last_timestamp,
            durability=durability,
            path=target,
        )

    # -- state -------------------------------------------------------------

    @property
    def last_sequence(self) -> int:
        """Sequence of the last record written. The snapshot boundary."""
        return self._sequence

    @property
    def last_timestamp_ns(self) -> int:
        return max(0, self._last_timestamp_ns)

    @property
    def records_written(self) -> int:
        return self._records

    @property
    def bytes_written(self) -> int:
        return self._bytes

    # -- appending ---------------------------------------------------------

    def append(self, kind: str, timestamp_ns: int, payload: Any) -> int:
        """Record one exogenous input. Returns its sequence number.

        Timestamps must not go backwards. The kernel refuses to process an
        event scheduled before its clock; a journal that accepted one would
        make ``until_timestamp_ns`` recovery arbitrary, because there would be
        no single point in the file where the market's clock passed a given
        nanosecond.
        """
        if self._closed:
            raise JournalError("journal is closed")
        if timestamp_ns < self._last_timestamp_ns:
            raise ValueError(
                f"timestamp {timestamp_ns}ns is behind the previous record's "
                f"{self._last_timestamp_ns}ns; simulated time never moves backwards"
            )
        sequence = self._sequence + 1
        frame = encode_record(sequence, timestamp_ns, kind, payload)
        self._stream.write(frame)
        self._sequence = sequence
        self._last_timestamp_ns = timestamp_ns
        self._records += 1
        self._bytes += len(frame)
        self._sync()
        return sequence

    def _sync(self) -> None:
        if self.durability == "none":
            return
        self._stream.flush()
        if self.durability == "fsync":
            os.fsync(self._stream.fileno())

    def flush(self) -> None:
        """Push buffered bytes to the operating system.

        Enough to survive the process dying, which is the failure this module
        exists for. Not enough to survive the machine losing power: for that
        the durability policy has to be ``"fsync"``, which costs a real disk
        round trip per order. Measured on this machine, see the numbers in
        ``tests/test_journal.py::test_durability_costs``.
        """
        self._stream.flush()

    def sync(self) -> None:
        """Flush and fsync. Survives power loss for everything written so far."""
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def to_bytes(self) -> bytes:
        """The journal's bytes so far. In-memory journals only."""
        if not isinstance(self._stream, io.BytesIO):
            raise JournalError("to_bytes() is only available for in-memory journals")
        return self._stream.getvalue()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.durability != "none":
                self._sync()
        finally:
            if self._owns_stream and not isinstance(self._stream, io.BytesIO):
                self._stream.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _build_header(
    engine_version: str,
    metadata: Mapping[str, Any] | None,
    created_ns: int,
    first_sequence: int,
) -> dict[str, Any]:
    if not isinstance(engine_version, str) or not engine_version:
        raise ValueError("engine_version must be a non-empty string")
    if not isinstance(created_ns, int) or isinstance(created_ns, bool) or created_ns < 0:
        raise ValueError(f"created_ns must be a non-negative int, got {created_ns!r}")
    if not isinstance(first_sequence, int) or first_sequence < 1:
        raise ValueError(f"first_sequence must be >= 1, got {first_sequence!r}")
    payload = dict(metadata or {})
    _check_persistable(payload, "metadata")
    return {
        "format": FORMAT_VERSION,
        "engine_version": engine_version,
        # Not always 1: a journal segmented after a snapshot starts wherever
        # the previous segment stopped, and replay checks the chain against it.
        "first_sequence": first_sequence,
        "created_ns": created_ns,
        "metadata": payload,
    }


def _encode_frame(body: bytes) -> bytes:
    length = struct.pack(">I", len(body))
    crc = zlib.crc32(length + body) & 0xFFFFFFFF
    return length + struct.pack(">I", crc) + body


def repair(path: str | os.PathLike[str]) -> int:
    """Truncate a torn tail from a journal on disk. Returns bytes removed.

    Read-only recovery does not need this; appending does, and so does anyone
    who wants the file on disk to stop containing bytes that are not a record.
    Raises rather than truncating if the damage is mid-file, because a
    checksum failure with valid records behind it means loss, not an interrupted
    write, and silently discarding the records after it would be exactly the
    "dropping instead of delaying" failure ``CONTRIBUTING.md`` calls out.
    """
    target = os.fspath(path)
    with open(target, "rb") as probe:
        reader = _Reader(probe, target)
        for _frame, _record in reader.records():
            pass
        if reader.torn_offset is None:
            return 0
        message = _torn_message(target, reader)
        intact = reader.intact_offset
        removed = reader.torn_bytes
    LOG.warning("%s", message)
    with open(target, "r+b") as fix:
        fix.truncate(intact)
        fix.flush()
        os.fsync(fix.fileno())
    return removed


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Snapshot:
    """State as of a sequence boundary, plus the boundary itself.

    ``last_applied_sequence`` is the sequence of the last journal record whose
    effects are already in ``state``. Recovery resumes at the record *after*
    it. Raft calls this ``lastIncludedIndex`` and it is not optional: Flink
    describes a checkpoint as "a pointer into each of the data sources plus a
    copy of the state", and a snapshot without the pointer cannot be used at
    all, because there is no way to tell which inputs it already contains.

    ``engine_version`` is repeated here rather than trusted from the journal,
    because a snapshot is a serialized product of the engine's own logic and is
    exactly as version-sensitive as a replay.
    """

    state: Any
    last_applied_sequence: int
    engine_version: str
    timestamp_ns: int = 0

    def encode(self) -> bytes:
        _check_persistable(self.state, "state")
        if self.last_applied_sequence < 0:
            raise ValueError("last_applied_sequence must be >= 0")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be >= 0")
        body = _canonical_json(
            {
                "format": FORMAT_VERSION,
                "engine_version": self.engine_version,
                "last_applied_sequence": self.last_applied_sequence,
                "timestamp_ns": self.timestamp_ns,
                "state": self.state,
            }
        )
        return SNAPSHOT_MAGIC + _encode_frame(body)

    @classmethod
    def decode(cls, data: bytes, *, engine_version: str | None = None) -> Snapshot:
        label = "<snapshot>"
        if data[: len(SNAPSHOT_MAGIC)] != SNAPSHOT_MAGIC:
            raise JournalFormatError(f"{label}: not a snapshot")
        rest = data[len(SNAPSHOT_MAGIC) :]
        if len(rest) < _FRAME_HEADER.size:
            raise JournalFormatError(f"{label}: frame header is truncated")
        length, crc = _FRAME_HEADER.unpack(rest[: _FRAME_HEADER.size])
        body = rest[_FRAME_HEADER.size : _FRAME_HEADER.size + length]
        if len(body) < length:
            # No torn-tail leniency for a snapshot. A journal missing its last
            # record still replays everything before it; a snapshot missing its
            # last bytes is not a state at all. This is why the writer renames
            # a fully fsynced temporary file into place instead of writing here.
            raise JournalFormatError(
                f"{label}: truncated, {len(body)} of {length} bytes. A partial "
                "snapshot is not partially useful."
            )
        if zlib.crc32(struct.pack(">I", length) + body) & 0xFFFFFFFF != crc:
            raise JournalCorruption(f"{label}: checksum failed")
        parsed = json.loads(body.decode("ascii"))
        if parsed.get("format") != FORMAT_VERSION:
            raise JournalFormatError(f"{label}: format {parsed.get('format')!r}")
        # Named rather than left to a KeyError, because the missing field this is
        # most likely to catch is `last_applied_sequence`, and "KeyError" does not
        # tell the operator that what they have is a state with no idea which
        # inputs are already in it, which is the one way a snapshot is useless
        # rather than merely stale.
        for required in ("engine_version", "last_applied_sequence", "state"):
            if required not in parsed:
                raise JournalFormatError(
                    f"{label}: missing {required!r}. Without it there is no way to "
                    "know where to resume the journal, so the snapshot cannot be "
                    "applied at all."
                )
        found = parsed["engine_version"]
        if engine_version is not None:
            _check_version(found, engine_version, label)
        return cls(
            state=parsed["state"],
            last_applied_sequence=int(parsed["last_applied_sequence"]),
            engine_version=found,
            timestamp_ns=int(parsed.get("timestamp_ns", 0)),
        )


def write_snapshot(path: str | os.PathLike[str], snapshot: Snapshot) -> int:
    """Write a snapshot atomically. Returns bytes written.

    Temporary file, fsync, ``os.replace``. ``os.replace`` is atomic on Windows
    as well as POSIX, so a reader either sees the whole previous snapshot or
    the whole new one and never a half-written file. The fsync happens before
    the rename, because a rename that reaches the disk before the data it
    points at is the classic way to end up with a zero-length "snapshot".

    Deliberately synchronous and deliberately not a fork. Redis publishes
    measured fork costs of 62ms for 6.9GB on a physical Xeon and 1460ms on an
    older EC2 instance; at this exchange's rates a 1.4 second stall is
    thousands of orders, and the state here is small enough that serializing it
    inline is cheaper than the fork would be.
    """
    target = os.fspath(path)
    data = snapshot.encode()
    temporary = f"{target}.tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return len(data)


def load_snapshot(
    path: str | os.PathLike[str] | bytes, *, engine_version: str | None = None
) -> Snapshot:
    """Load a snapshot, refusing one written by a different engine version."""
    if isinstance(path, (bytes, bytearray)):
        return Snapshot.decode(bytes(path), engine_version=engine_version)
    with open(os.fspath(path), "rb") as handle:
        return Snapshot.decode(handle.read(), engine_version=engine_version)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """What a replay consumed, and what it refused to consume."""

    applied: int
    scanned: int
    last_sequence: int
    last_timestamp_ns: int
    stopped_early: bool
    truncated_bytes: int
    intact_bytes: int
    warnings: tuple[str, ...]
    stream_digest: str

    @property
    def torn(self) -> bool:
        return self.truncated_bytes > 0


@dataclass(frozen=True, slots=True)
class Recovery:
    """A snapshot restore plus the journal tail replayed on top of it."""

    snapshot: Snapshot | None
    replay: ReplayResult

    @property
    def last_sequence(self) -> int:
        if self.replay.applied:
            return self.replay.last_sequence
        return self.snapshot.last_applied_sequence if self.snapshot else 0


def replay(
    journal: Any,
    apply: Callable[[Record], None],
    *,
    engine_version: str,
    start_after_sequence: int = 0,
    until_sequence: int | None = None,
    until_timestamp_ns: int | None = None,
    on_warning: Callable[[str], None] | None = None,
    tolerate_torn_tail: bool = True,
) -> ReplayResult:
    """Read records in order and hand each one to ``apply``.

    ``journal`` is a path, a bytes blob or an open binary stream. ``apply``
    receives a :class:`Record` and is responsible for raising on a kind it does
    not understand: a callback that ignores an unknown kind would rebuild a
    market missing those inputs and report success.

    Both bounds are **inclusive**, matching exchange-core's
    ``journalTimestampNs`` point-in-time recovery: ``until_timestamp_ns=T``
    applies every input injected at or before simulated nanosecond ``T`` and
    stops at the first one after it. Timestamps are non-decreasing, so that
    stop is a single well-defined point in the file.

    ``start_after_sequence`` skips records a snapshot already contains, but the
    records before it are still read, checksummed and chain-checked. Verifying
    them costs a scan and catches a journal that has quietly lost its middle;
    trusting them costs nothing and catches nothing.

    Every record handed to ``apply`` also goes into ``stream_digest``, a
    sha256 over the exact frame bytes consumed. Two replays of the same journal
    must report the same digest, which is a one-line way for the caller to
    assert the property this whole design rests on.
    """
    stream, owned, label = _open_source(journal)
    warnings: list[str] = []
    digest = hashlib.sha256()
    applied = 0
    scanned = 0
    last_sequence = start_after_sequence
    last_timestamp = 0
    stopped_early = False
    try:
        reader = _Reader(stream, label)
        _check_version(reader.header["engine_version"], engine_version, label)
        for frame, record in reader.records():
            if until_sequence is not None and record.sequence > until_sequence:
                stopped_early = True
                break
            if until_timestamp_ns is not None and record.timestamp_ns > until_timestamp_ns:
                stopped_early = True
                break
            scanned += 1
            if record.sequence <= start_after_sequence:
                continue
            digest.update(frame)
            apply(record)
            applied += 1
            last_sequence = record.sequence
            last_timestamp = record.timestamp_ns
        if reader.torn_offset is not None and not stopped_early:
            message = _torn_message(label, reader)
            if not tolerate_torn_tail:
                raise JournalTornTail(message)
            warnings.append(message)
            LOG.warning("%s", message)
            if on_warning is not None:
                on_warning(message)
        return ReplayResult(
            applied=applied,
            scanned=scanned,
            last_sequence=last_sequence,
            last_timestamp_ns=last_timestamp,
            stopped_early=stopped_early,
            truncated_bytes=0 if stopped_early else reader.torn_bytes,
            intact_bytes=reader.intact_offset,
            warnings=tuple(warnings),
            stream_digest="sha256:" + digest.hexdigest(),
        )
    finally:
        if owned:
            stream.close()


def recover(
    journal: Any,
    apply: Callable[[Record], None],
    *,
    engine_version: str,
    snapshot: Snapshot | str | os.PathLike[str] | None = None,
    restore: Callable[[Snapshot], None] | None = None,
    until_sequence: int | None = None,
    until_timestamp_ns: int | None = None,
    on_warning: Callable[[str], None] | None = None,
    tolerate_torn_tail: bool = True,
) -> Recovery:
    """Restore a snapshot, then replay only the journal records after it.

    The ordering is the whole point and is why ``restore`` is a callback rather
    than something the caller does before calling: the state has to be in place
    before the first tail record is applied, and making that the caller's
    responsibility is how you get a recovery that applies the tail to an empty
    market and looks fine.

    Passing no snapshot replays the journal from the beginning, which must land
    on the same state. That equality is the property worth testing, because a
    snapshot taken at the wrong boundary produces a market that is wrong by
    exactly the records it double-counted or skipped, and nothing else notices.
    """
    loaded: Snapshot | None
    if snapshot is None:
        loaded = None
    elif isinstance(snapshot, Snapshot):
        _check_version(snapshot.engine_version, engine_version, "<snapshot>")
        loaded = snapshot
    else:
        loaded = load_snapshot(snapshot, engine_version=engine_version)

    if loaded is not None and restore is not None:
        restore(loaded)

    result = replay(
        journal,
        apply,
        engine_version=engine_version,
        start_after_sequence=0 if loaded is None else loaded.last_applied_sequence,
        until_sequence=until_sequence,
        until_timestamp_ns=until_timestamp_ns,
        on_warning=on_warning,
        tolerate_torn_tail=tolerate_torn_tail,
    )
    return Recovery(snapshot=loaded, replay=result)
