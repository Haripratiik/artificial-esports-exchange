"""Crash recovery: the input journal, its framing, and replay.

Everything this exchange knows lives in memory, so the journal is the only
thing standing between a process dying and the market never having happened.
That makes its *failure* modes the interesting part, not its happy path. Four
of them would each destroy a market quietly, which is why each gets a test
constructed to reach it:

**A damaged length prefix reported as a torn tail.** This is the sharpest one
and it is the reason this file exists in the shape it does. A torn tail is
expected wear -- the process died mid-write -- and is dropped with a warning
while the load reports success. Corruption is unrecoverable and must raise. The
two are physically identical from the outside: a frame asking for more bytes
than the file can supply. Measured against the reader before the fix, on the
8-record journal these tests build, with the final record's 41-byte length
prefix bumped to 42, 43, 1041 and 4194345: all four returned 7 records and
success, warning "truncated final record at offset 923; dropping 49 trailing
bytes". A single flipped bit therefore erased a fully written order and told
nobody. See ``test_damaged_length_prefix_*``.

**A version mismatch replayed anyway.** Agent behaviour is regenerated from the
seed rather than journalled, so changing the number or order of RNG draws makes
every prior journal replay into a *different but entirely plausible* market --
right accounts, sensible prices, no error anywhere. No checksum can catch it,
because every byte on disk is still correct. Only the version can.

**A snapshot without its sequence number.** Raft's snapshot metadata is
``lastIncludedIndex`` plus the state. Without it there is no way to know which
inputs the state already contains, so the tail is applied to the wrong base and
the market is wrong by exactly the records it double-counted.

**A replay that is not reproducible.** The whole design rests on replay being a
function of the bytes. If two replays of one journal disagree, nothing else
here means anything, so it is asserted directly rather than assumed.

The state machine below is deliberately a toy. This module takes bytes and
callbacks and knows nothing about the venue, which is the property that lets
recovery be tested without building a market -- and
``test_journal_does_not_import_the_venue`` keeps it that way.
"""

from __future__ import annotations

import json
import logging
import struct
import time
import zlib
from decimal import Decimal
from typing import Any

import pytest

from arena.sim.journal import (
    FORMAT_VERSION,
    JOURNAL_MAGIC,
    KINDS,
    KIND_CANCEL_ALL,
    KIND_KEY_ISSUED,
    KIND_ORDER_CANCELLED,
    KIND_ORDER_SUBMITTED,
    KIND_SEAT_JOINED,
    SNAPSHOT_MAGIC,
    Journal,
    JournalCorruption,
    JournalError,
    JournalFormatError,
    JournalTornTail,
    JournalVersionMismatch,
    Record,
    Snapshot,
    encode_record,
    header_of,
    load_snapshot,
    read_records,
    recover,
    repair,
    replay,
    write_snapshot,
)

ENGINE = "arena-engine-1"
SYMBOL = "EMBER_OBJECTIVE_WR"


# ---------------------------------------------------------------------------
# a toy deterministic state machine
# ---------------------------------------------------------------------------


class ToyBook:
    """The smallest state machine that can tell a wrong replay from a right one.

    Not a market. It exists so the recovery tests can assert on *state* rather
    than on a record count, because the failures that matter here (a snapshot
    resumed one record early, a tail applied to an empty base) all produce the
    right number of records and the wrong state.

    Every value is an integer or a string. A float here would be rejected at
    ``append`` and rightly so: money is integer minor units, prices integer
    ticks.
    """

    def __init__(self) -> None:
        self.seats: dict[str, int] = {}
        self.keys: dict[str, str] = {}
        self.working: dict[str, dict[str, Any]] = {}
        self.applied: list[int] = []

    def apply(self, record: Record) -> None:
        payload = record.payload
        if record.kind == KIND_SEAT_JOINED:
            self.seats[payload["seat"]] = payload["cash_minor"]
        elif record.kind == KIND_KEY_ISSUED:
            self.keys[payload["key_id"]] = payload["seat"]
        elif record.kind == KIND_ORDER_SUBMITTED:
            self.working[payload["client_order_id"]] = dict(payload)
        elif record.kind == KIND_ORDER_CANCELLED:
            self.working.pop(payload["client_order_id"], None)
        elif record.kind == KIND_CANCEL_ALL:
            for oid in [
                oid for oid, o in self.working.items() if o["seat"] == payload["seat"]
            ]:
                del self.working[oid]
        else:
            # A callback that ignored an unknown kind would rebuild a market
            # missing those inputs and report success. The module's docstring
            # makes this the callback's job; this is that job being done.
            raise AssertionError(f"unknown kind {record.kind!r}")
        self.applied.append(record.sequence)

    def state(self) -> dict[str, Any]:
        return {
            "seats": dict(self.seats),
            "keys": dict(self.keys),
            "working": {k: dict(v) for k, v in self.working.items()},
        }

    def restore(self, snapshot: Snapshot) -> None:
        self.seats = dict(snapshot.state["seats"])
        self.keys = dict(snapshot.state["keys"])
        self.working = {k: dict(v) for k, v in snapshot.state["working"].items()}


def sample_inputs() -> list[tuple[str, int, dict[str, Any]]]:
    """One of every kind, in an order where each one changes the state."""
    return [
        (KIND_SEAT_JOINED, 1_000, {"seat": "ann", "cash_minor": 100_000_000}),
        (KIND_SEAT_JOINED, 2_000, {"seat": "bob", "cash_minor": 250_000_000}),
        (KIND_KEY_ISSUED, 3_000, {"key_id": "ak_01", "seat": "ann"}),
        (
            KIND_ORDER_SUBMITTED,
            4_000,
            {
                "seat": "ann",
                "client_order_id": "c-1",
                "symbol": SYMBOL,
                "side": "buy",
                "price_ticks": 4_663,
                "quantity": 5,
                "time_in_force": "gtc",
            },
        ),
        (
            KIND_ORDER_SUBMITTED,
            5_000,
            {
                "seat": "bob",
                "client_order_id": "c-2",
                "symbol": SYMBOL,
                "side": "sell",
                "price_ticks": 4_670,
                "quantity": 3,
                "time_in_force": "gtc",
            },
        ),
        (
            KIND_ORDER_SUBMITTED,
            6_000,
            {
                "seat": "ann",
                "client_order_id": "c-3",
                "symbol": SYMBOL,
                "side": "buy",
                "price_ticks": 4_660,
                "quantity": 2,
                "time_in_force": "gtc",
            },
        ),
        (KIND_ORDER_CANCELLED, 7_000, {"seat": "bob", "client_order_id": "c-2"}),
        (KIND_CANCEL_ALL, 8_000, {"seat": "ann"}),
    ]


def build_journal(
    inputs: list[tuple[str, int, dict[str, Any]]] | None = None,
    *,
    engine_version: str = ENGINE,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """A journal as bytes. No filesystem, because the module does not need one."""
    journal = Journal.in_memory(
        engine_version=engine_version,
        metadata=metadata if metadata is not None else {"seed": 20260830},
    )
    for kind, timestamp_ns, payload in inputs if inputs is not None else sample_inputs():
        journal.append(kind, timestamp_ns, payload)
    return journal.to_bytes()


def frame_spans(data: bytes) -> list[tuple[int, int]]:
    """``(offset, body_length)`` for each record frame, by walking the framing.

    Written out rather than reusing the reader, so that a test which corrupts a
    frame is not locating that frame with the code it is trying to catch.
    """
    offset = len(JOURNAL_MAGIC)
    (header_length,) = struct.unpack(">I", data[offset : offset + 4])
    offset += 8 + header_length
    spans: list[tuple[int, int]] = []
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        spans.append((offset, length))
        offset += 8 + length
    return spans


def apply_all(data: bytes, **kwargs: Any) -> tuple[ToyBook, Any]:
    book = ToyBook()
    result = replay(data, book.apply, engine_version=ENGINE, **kwargs)
    return book, result


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


def test_round_trip_of_every_record_kind():
    """Each kind survives the encoder byte for byte, payload included.

    Compared field by field rather than by count, because the encoding failure
    worth catching is a payload that comes back a *different type* -- a tuple as
    a list, an int key as a string -- which a count would never see.
    """
    inputs = sample_inputs()
    kinds_used = {kind for kind, _, _ in inputs}
    assert kinds_used == set(KINDS), "every kind the exchange records must be exercised"

    records = list(read_records(build_journal(inputs), engine_version=ENGINE))

    assert len(records) == len(inputs)
    for index, (record, (kind, timestamp_ns, payload)) in enumerate(zip(records, inputs)):
        assert record.sequence == index + 1
        assert record.kind == kind
        assert record.timestamp_ns == timestamp_ns
        assert record.payload == payload
        assert type(record.payload) is type(payload)


def test_records_are_self_delimiting_with_a_length_and_a_checksum():
    """The frame is length, crc, body -- and the crc covers the length prefix.

    Covering the length is what makes a damaged prefix a checksum failure rather
    than a wander into the middle of the next record.
    """
    frame = encode_record(7, 1_234_000, KIND_ORDER_CANCELLED, {"client_order_id": "c-9"})
    length, crc = struct.unpack(">II", frame[:8])

    assert length == len(frame) - 8
    assert crc == zlib.crc32(frame[:4] + frame[8:]) & 0xFFFFFFFF
    assert crc != zlib.crc32(frame[8:]) & 0xFFFFFFFF, "crc must include the length"


def test_the_header_carries_the_engine_version_and_the_seed():
    """The seed is the other half of the recording: without it, no market.

    Agent draws regenerate from the seed, so a journal that does not carry it
    records inputs to a market nobody can reconstruct. The module cannot enforce
    this -- it knows nothing about the venue -- so it is checked here.
    """
    data = build_journal(metadata={"seed": 20260830, "build": "three-maker"})
    header = header_of(data)

    assert header["engine_version"] == ENGINE
    assert header["format"] == FORMAT_VERSION
    assert header["first_sequence"] == 1
    assert header["metadata"]["seed"] == 20260830


def test_one_sweeping_order_is_one_record():
    """The size argument for input journalling, asserted rather than assumed.

    An order that sweeps 200 resting levels is one input. Logging the outputs
    instead would be 200 fills and 201 book updates. Measured: 1 record and
    109 bytes against the 200 the output log would need.
    """
    journal = Journal.in_memory(engine_version=ENGINE)
    journal.append(
        KIND_ORDER_SUBMITTED,
        1_000,
        {
            "seat": "ann",
            "client_order_id": "sweep-1",
            "symbol": SYMBOL,
            "side": "buy",
            "price_ticks": 9_999,
            "quantity": 200,
            "time_in_force": "ioc",
        },
    )

    assert journal.records_written == 1
    assert journal.last_sequence == 1
    assert journal.bytes_written < 200


# ---------------------------------------------------------------------------
# no floats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ({"price": 46.63}, "no floats"),
        ({"price": Decimal("46.63")}, "Decimal"),
        ({"levels": [1, 2.5]}, "no floats"),
        ({"legs": (1, 2)}, "tuple"),
        ({"nested": {"deep": [{"x": 0.1}]}}, "no floats"),
        ({1: "int key"}, "dict keys must be str"),
    ],
)
def test_persisted_records_refuse_lossy_values(payload, needle):
    """Rejected at write time, where the cost is one conversion.

    A float in a journal makes the rounding permanent at the one place it cannot
    be undone, and this project's conservation check returns an exact integer
    zero. A ``Decimal`` or a ``tuple`` is worse than wrong: it round-trips
    without complaint as a ``str`` or a ``list``, which is bug class 1 in
    CONTRIBUTING.md -- a value crossing a boundary in a form its label does not claim.
    """
    journal = Journal.in_memory(engine_version=ENGINE)
    with pytest.raises(TypeError, match=needle):
        journal.append(KIND_ORDER_SUBMITTED, 1_000, payload)


# ---------------------------------------------------------------------------
# torn tail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [1, 4, 7, 8, 12, 30])
def test_torn_final_record_is_dropped_and_everything_before_it_survives(cut, caplog):
    """A process that died mid-write. Redis's aof-load-truncated, on by default.

    Cut at six offsets inside the final frame, including inside the 8-byte frame
    header (1, 4, 7) and inside the body (8, 12, 30), because the reader takes a
    different path for each and only one of them was covered before.
    """
    data = build_journal()
    last_offset, _ = frame_spans(data)[-1]

    with caplog.at_level(logging.WARNING, logger="arena.sim.journal"):
        book, result = apply_all(data[: last_offset + cut])

    assert result.applied == 7, "the seven whole records must survive"
    assert result.torn is True
    assert result.truncated_bytes == cut
    assert result.intact_bytes == last_offset
    assert any("truncated final record" in message for message in result.warnings)
    assert any("truncated final record" in r.getMessage() for r in caplog.records)

    # The state is the state at record 7 -- ann's two orders still working,
    # because the cancel_all that removed them is the record that was torn off.
    assert sorted(book.working) == ["c-1", "c-3"]
    assert book.applied == [1, 2, 3, 4, 5, 6, 7]


def test_a_torn_tail_can_be_refused_when_the_caller_needs_certainty():
    data = build_journal()
    last_offset, _ = frame_spans(data)[-1]

    with pytest.raises(JournalTornTail, match="truncated final record"):
        replay(
            data[: last_offset + 10],
            ToyBook().apply,
            engine_version=ENGINE,
            tolerate_torn_tail=False,
        )


def test_repair_removes_a_torn_tail_from_the_file(tmp_path):
    path = tmp_path / "torn.journal"
    data = build_journal()
    last_offset, _ = frame_spans(data)[-1]
    path.write_bytes(data[: last_offset + 20])

    removed = repair(path)

    assert removed == 20
    assert path.stat().st_size == last_offset
    assert repair(path) == 0, "repairing a clean journal removes nothing"
    _, result = apply_all(path.read_bytes())
    assert result.torn is False and result.applied == 7


def test_open_for_append_truncates_the_tear_then_continues(tmp_path):
    """Appending after a torn record would wedge garbage into the middle.

    Where the next replay classifies it as unrecoverable corruption rather than
    a droppable tail -- the file goes from "lost one order" to "unloadable".
    """
    path = tmp_path / "resume.journal"
    data = build_journal()
    last_offset, _ = frame_spans(data)[-1]
    path.write_bytes(data[: last_offset + 15])

    with Journal.open_for_append(path, engine_version=ENGINE) as journal:
        assert journal.last_sequence == 7
        journal.append(KIND_SEAT_JOINED, 9_000, {"seat": "cat", "cash_minor": 1})

    book, result = apply_all(path.read_bytes())
    assert result.torn is False
    assert result.applied == 8
    assert book.seats["cat"] == 1


# ---------------------------------------------------------------------------
# corruption in the middle
# ---------------------------------------------------------------------------


def test_corrupt_middle_record_raises_instead_of_truncating():
    """Mid-file damage is loss, not wear.

    Truncating here would silently discard every record behind the damage, which
    is the "dropping instead of delaying" failure CONTRIBUTING.md names: the consumer
    is never told what it lost.
    """
    data = bytearray(build_journal())
    offset, length = frame_spans(bytes(data))[3]
    data[offset + 8 + length - 2] ^= 0xFF

    with pytest.raises(JournalCorruption, match="bytes are all present"):
        apply_all(bytes(data))


def test_corruption_is_reported_rather_than_partially_applied():
    """The records before the damage are handed over before the raise.

    Worth pinning: a caller that catches JournalCorruption must not assume the
    callback was never invoked, or it will resume onto a half-built state.
    """
    data = bytearray(build_journal())
    offset, length = frame_spans(bytes(data))[5]
    data[offset + 8 + length - 3] ^= 0x01
    book = ToyBook()

    with pytest.raises(JournalCorruption):
        replay(bytes(data), book.apply, engine_version=ENGINE)

    assert book.applied == [1, 2, 3, 4, 5]


def test_sequence_gap_raises():
    """A gap means an input was lost; replaying past it silently skips it."""
    journal = Journal.in_memory(engine_version=ENGINE)
    journal.append(KIND_SEAT_JOINED, 1_000, {"seat": "ann", "cash_minor": 1})
    forged = encode_record(3, 2_000, KIND_SEAT_JOINED, {"seat": "bob", "cash_minor": 2})

    with pytest.raises(JournalCorruption, match="expected 2"):
        list(read_records(journal.to_bytes() + forged, engine_version=ENGINE))


def test_backwards_timestamp_raises():
    """Point-in-time recovery is meaningless if the clock can go backwards."""
    journal = Journal.in_memory(engine_version=ENGINE)
    journal.append(KIND_SEAT_JOINED, 5_000, {"seat": "ann", "cash_minor": 1})
    forged = encode_record(2, 4_000, KIND_SEAT_JOINED, {"seat": "bob", "cash_minor": 2})

    with pytest.raises(JournalCorruption, match="never moves backwards"):
        list(read_records(journal.to_bytes() + forged, engine_version=ENGINE))

    # And refused at write time too, so it cannot get into a file this way.
    with pytest.raises(ValueError, match="never moves backwards"):
        journal.append(KIND_SEAT_JOINED, 4_000, {"seat": "bob", "cash_minor": 2})


# ---------------------------------------------------------------------------
# a damaged length prefix is corruption, not a torn tail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bump", [1, 2, 1_000, 4 * 1024 * 1024])
def test_damaged_length_prefix_on_the_final_record_is_corruption(bump):
    """The bug this file was written for, at the position where it is dangerous.

    A length prefix damaged upward on the *last* record looks exactly like a torn
    tail: a frame asking for more bytes than the file holds. The reader used to
    call it one, and a torn tail is dropped with a warning while the load reports
    success. Measured before the fix, on this journal, whose final record is 41
    bytes at offset 923: every bump below returned ``applied=7``, ``torn=True``,
    ``truncated_bytes=49`` and no error. One flipped bit erased a fully written
    order and nothing said so.

    The discriminator is the checksum, which covers the length prefix. If the
    prefix is what was damaged, the record's real bytes are all still on disk and
    the stored crc still describes them, so re-checking at the length the file can
    actually supply validates. A real tear is missing bytes that were never
    written, and no length validates.
    """
    data = bytearray(build_journal())
    offset, length = frame_spans(bytes(data))[-1]
    struct.pack_into(">I", data, offset, length + bump)

    with pytest.raises(JournalCorruption, match="length prefix is damaged"):
        apply_all(bytes(data))


@pytest.mark.parametrize("index", [0, 3, -1])
def test_absurd_length_prefix_is_corruption_wherever_it_sits(index):
    """A length no legitimate record could have is damage, refused before the read.

    Checked before the read, not after, and it decides the diagnosis as well as
    stopping a corrupt prefix from asking for a gigabyte. The largest real record
    here is an order submission at 109 bytes.
    """
    data = bytearray(build_journal())
    offset, _ = frame_spans(bytes(data))[index]
    struct.pack_into(">I", data, offset, 0x7FFF_FFFF)

    with pytest.raises(JournalCorruption, match="byte limit"):
        apply_all(bytes(data))


@pytest.mark.parametrize("shrink", [1, 3, 20])
def test_length_prefix_damaged_downward_is_corruption(shrink):
    """Shrinking the prefix keeps the read inside the file, so the crc catches it.

    The complementary half of the case above, and the one that already worked --
    kept so a future change to the upward path cannot quietly break this one.
    """
    data = bytearray(build_journal())
    offset, length = frame_spans(bytes(data))[-1]
    struct.pack_into(">I", data, offset, length - shrink)

    with pytest.raises(JournalCorruption, match="checksum failed"):
        apply_all(bytes(data))


def test_the_length_check_does_not_turn_a_real_tear_into_corruption():
    """The guard against over-correcting, which would be just as bad.

    A genuine torn tail must stay a torn tail: refusing to load a journal whose
    process merely died mid-write would make every unclean shutdown fatal. Every
    cut position inside the final frame is checked, so the new check cannot be
    passing by luck at one offset.
    """
    data = build_journal()
    offset, length = frame_spans(data)[-1]

    for cut in range(1, length + 8):
        _, result = apply_all(data[: offset + cut])
        assert result.torn is True, f"cut at +{cut} must read as a tear"
        assert result.applied == 7


def test_a_tear_shorter_than_a_record_prefix_stays_a_tear():
    """Fewer bytes than a record prefix cannot validate, so there is nothing to test
    the checksum against and a tear is the only reading left."""
    data = build_journal()
    offset, _ = frame_spans(data)[-1]

    _, result = apply_all(data[: offset + 8 + 4])

    assert result.torn is True
    assert result.truncated_bytes == 12


# ---------------------------------------------------------------------------
# version mismatch
# ---------------------------------------------------------------------------


def test_replay_refuses_a_journal_from_a_different_engine_version():
    """The sharpest risk in the design, because its failure is silent.

    Change the number or order of RNG draws and every earlier journal still
    replays cleanly -- into a different market with the right accounts and
    sensible prices. Every byte on disk is still correct, so no checksum can
    catch it. Only the version can, and only by refusing.
    """
    data = build_journal(engine_version="arena-engine-0")
    book = ToyBook()

    with pytest.raises(JournalVersionMismatch, match="Refusing to replay"):
        replay(data, book.apply, engine_version=ENGINE)

    assert book.applied == [], "nothing may be applied before the version is checked"


def test_every_entry_point_checks_the_version(tmp_path):
    """One unchecked door is the same as no lock.

    ``read_records`` and ``open_for_append`` are checked as well as ``replay``:
    appending a new build's records to an old build's file produces a journal
    that can never be replayed at all, by anything.
    """
    path = tmp_path / "old.journal"
    path.write_bytes(build_journal(engine_version="arena-engine-0"))

    with pytest.raises(JournalVersionMismatch):
        list(read_records(path, engine_version=ENGINE))
    with pytest.raises(JournalVersionMismatch):
        Journal.open_for_append(path, engine_version=ENGINE)
    with pytest.raises(JournalVersionMismatch):
        recover(path, ToyBook().apply, engine_version=ENGINE)

    snapshot = Snapshot(state={"seats": {}}, last_applied_sequence=3, engine_version="old")
    with pytest.raises(JournalVersionMismatch):
        load_snapshot(snapshot.encode(), engine_version=ENGINE)


def test_a_matching_version_replays_and_an_unversioned_read_is_still_possible():
    """``read_records`` without a version is for tools, and must not become the
    convenient way to skip the check in recovery -- so ``replay`` has no such
    mode and ``engine_version`` is required there."""
    data = build_journal(engine_version="arena-engine-0")

    assert len(list(read_records(data))) == 8
    with pytest.raises(TypeError):
        replay(data, ToyBook().apply)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# point-in-time replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("until", "expected_orders"),
    [
        (1, []),
        (4, ["c-1"]),
        (5, ["c-1", "c-2"]),
        (6, ["c-1", "c-2", "c-3"]),
        (7, ["c-1", "c-3"]),
        (8, []),
    ],
)
def test_replay_to_a_target_sequence(until, expected_orders):
    """Inclusive, so ``until_sequence=N`` lands on the state after record N."""
    book, result = apply_all(build_journal(), until_sequence=until)

    assert sorted(book.working) == expected_orders
    assert result.applied == until
    assert result.last_sequence == until
    assert result.stopped_early is (until < 8)


@pytest.mark.parametrize(
    ("until_ns", "expected_sequence"),
    [
        (999, 0),
        (1_000, 1),
        (4_500, 4),
        (5_999, 5),
        (6_000, 6),
        (100_000, 8),
    ],
)
def test_replay_to_a_target_timestamp(until_ns, expected_sequence):
    """Point-in-time recovery in the market's own clock, as exchange-core does.

    Both bounds inclusive, and the off-boundary values (999, 4500, 5999) are the
    ones that matter: an exclusive bound would land one record early and rebuild
    a market missing its last input, which looks completely normal.
    """
    book, result = apply_all(build_journal(), until_timestamp_ns=until_ns)

    assert result.applied == expected_sequence
    assert book.applied == list(range(1, expected_sequence + 1))
    if expected_sequence:
        assert result.last_timestamp_ns <= until_ns


def test_a_bounded_replay_does_not_report_the_tail_it_chose_not_to_read():
    """Stopping early is not truncation, and must not be reported as one.

    Otherwise a point-in-time query on a healthy journal would warn about damage
    that is not there, and the warning that means something would stop being
    read.
    """
    _, result = apply_all(build_journal(), until_sequence=3)

    assert result.stopped_early is True
    assert result.torn is False
    assert result.truncated_bytes == 0
    assert result.warnings == ()


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


def test_snapshot_carries_state_sequence_and_version(tmp_path):
    path = tmp_path / "state.snapshot"
    book, _ = apply_all(build_journal(), until_sequence=5)
    written = write_snapshot(
        path, Snapshot(book.state(), last_applied_sequence=5, engine_version=ENGINE)
    )

    loaded = load_snapshot(path, engine_version=ENGINE)

    assert written == path.stat().st_size
    assert path.read_bytes()[: len(SNAPSHOT_MAGIC)] == SNAPSHOT_MAGIC
    assert loaded.last_applied_sequence == 5
    assert loaded.engine_version == ENGINE
    assert loaded.state == book.state()


def test_a_snapshot_without_its_sequence_is_unloadable():
    """Raft's lastIncludedIndex, and it is not decoration.

    Without it there is no way to tell which inputs the state already contains,
    so there is no way to know where to resume -- the state is not partially
    useful, it is unusable. Rejected by name rather than by KeyError so the
    operator is told what is actually wrong.
    """
    good = Snapshot({"seats": {}}, last_applied_sequence=4, engine_version=ENGINE).encode()
    parsed = json.loads(good[len(SNAPSHOT_MAGIC) + 8 :].decode("ascii"))
    del parsed["last_applied_sequence"]
    rebuilt = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("ascii")
    length = struct.pack(">I", len(rebuilt))
    crc = struct.pack(">I", zlib.crc32(length + rebuilt) & 0xFFFFFFFF)

    with pytest.raises(JournalFormatError, match="last_applied_sequence"):
        load_snapshot(SNAPSHOT_MAGIC + length + crc + rebuilt, engine_version=ENGINE)


def test_a_truncated_snapshot_is_refused_outright():
    """No torn-tail leniency here. A journal missing its last record still
    replays everything before it; a snapshot missing its last bytes is not a
    state at all, which is why the writer renames a fully fsynced file into
    place instead of writing in situ."""
    data = Snapshot({"a": 1}, last_applied_sequence=2, engine_version=ENGINE).encode()

    with pytest.raises(JournalFormatError, match="not partially useful"):
        load_snapshot(data[:-4], engine_version=ENGINE)


def test_a_corrupt_snapshot_fails_its_checksum():
    snapshot = Snapshot({"a": 1}, last_applied_sequence=2, engine_version=ENGINE)
    data = bytearray(snapshot.encode())
    data[-3] ^= 0xFF

    with pytest.raises(JournalCorruption, match="checksum failed"):
        load_snapshot(bytes(data), engine_version=ENGINE)


def test_snapshot_write_leaves_no_temporary_file_behind(tmp_path):
    """Temporary file, fsync, os.replace -- atomic on Windows as well as POSIX,
    so a reader sees the whole previous snapshot or the whole new one."""
    path = tmp_path / "s.snapshot"
    write_snapshot(path, Snapshot({"v": 1}, last_applied_sequence=1, engine_version=ENGINE))
    write_snapshot(path, Snapshot({"v": 2}, last_applied_sequence=2, engine_version=ENGINE))

    assert [p.name for p in sorted(tmp_path.iterdir())] == ["s.snapshot"]
    assert load_snapshot(path).state == {"v": 2}


@pytest.mark.parametrize("boundary", [0, 1, 3, 5, 7, 8])
def test_snapshot_plus_tail_lands_where_a_full_replay_lands(tmp_path, boundary):
    """The property the whole snapshot design exists to have.

    A snapshot taken at the wrong boundary produces a market wrong by exactly the
    records it double-counted or skipped, and nothing else notices: the account
    count is right, the prices are plausible, no error is raised. So the state is
    compared, not the record count, and at every boundary including both ends.
    """
    data = build_journal()
    full, _ = apply_all(data)

    at_boundary, _ = apply_all(data, until_sequence=boundary) if boundary else (ToyBook(), None)
    snapshot = Snapshot(
        at_boundary.state(), last_applied_sequence=boundary, engine_version=ENGINE
    )
    path = tmp_path / "b.snapshot"
    write_snapshot(path, snapshot)

    resumed = ToyBook()
    recovery = recover(
        data,
        resumed.apply,
        engine_version=ENGINE,
        snapshot=path,
        restore=resumed.restore,
    )

    assert resumed.state() == full.state()
    assert recovery.last_sequence == 8
    assert recovery.replay.applied == 8 - boundary
    assert recovery.replay.scanned == 8, "records before the snapshot are still verified"
    assert resumed.applied == list(range(boundary + 1, 9))


def test_recovery_still_checksums_the_records_the_snapshot_already_contains():
    """Skipping them would cost nothing and catch nothing.

    Verifying them costs one scan and catches a journal that has quietly lost its
    middle -- which is exactly the journal you are about to keep appending to.
    """
    data = bytearray(build_journal())
    offset, length = frame_spans(bytes(data))[1]
    data[offset + 8 + length - 2] ^= 0xFF
    snapshot = Snapshot({"seats": {}, "keys": {}, "working": {}}, 6, ENGINE)

    with pytest.raises(JournalCorruption):
        recover(bytes(data), ToyBook().apply, engine_version=ENGINE, snapshot=snapshot)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_the_same_journal_replayed_twice_is_byte_identical():
    """If two replays of one journal disagree, nothing else here means anything.

    Three things are compared, because each catches a different way of being
    wrong: the sha256 over the exact frame bytes consumed (the reader saw the
    same bytes), the resulting state (the callback did the same thing), and a
    re-encode of every record back to the original file (the encoder agrees with
    itself, so a journal can be copied or segmented without being re-read).
    """
    data = build_journal()

    first, first_result = apply_all(data)
    second, second_result = apply_all(data)

    assert first_result.stream_digest == second_result.stream_digest
    assert first_result.stream_digest.startswith("sha256:")
    assert first.state() == second.state()
    assert first.applied == second.applied

    rebuilt = data[: frame_spans(data)[0][0]]
    for record in read_records(data):
        rebuilt += encode_record(
            record.sequence, record.timestamp_ns, record.kind, record.payload
        )
    assert rebuilt == data, "re-encoding the records must reproduce the file exactly"


def test_two_journals_of_the_same_inputs_are_byte_identical():
    """No wall clock anywhere in the format.

    ``created_ns`` defaults to 0 rather than ``time.time_ns()`` on purpose: a
    wall-clock stamp would make two recordings of the same events differ in their
    first hundred bytes, and byte-identity of the encoding is the cheapest
    determinism check available.
    """
    assert build_journal() == build_journal()


def test_a_bounded_replay_digests_only_what_it_applied():
    """So a point-in-time recovery can be compared against the prefix of a full
    one, rather than only against another identically bounded run."""
    data = build_journal()
    _, bounded = apply_all(data, until_sequence=5)

    prefix_end = frame_spans(data)[5][0]
    _, prefix = apply_all(data[:prefix_end])

    assert bounded.stream_digest == prefix.stream_digest


# ---------------------------------------------------------------------------
# framing errors that are not damage
# ---------------------------------------------------------------------------


def test_an_empty_or_foreign_file_is_a_format_error_not_corruption(tmp_path):
    """Different diagnosis, different response: a format error means this is not
    a journal, and telling an operator their journal is corrupt when they pointed
    at the wrong file sends them looking for damage that does not exist."""
    empty = tmp_path / "empty.journal"
    empty.write_bytes(b"")
    with pytest.raises(JournalFormatError, match="empty file"):
        header_of(empty)

    with pytest.raises(JournalFormatError, match="not a journal"):
        header_of(b"not-a-journal-at-all" + b"\x00" * 64)

    with pytest.raises(JournalFormatError, match="header is truncated"):
        header_of(build_journal()[:20])


def test_create_refuses_to_overwrite_an_existing_journal(tmp_path):
    """A journal is the only copy of what happened. Opening the exchange twice
    against one file must fail loudly rather than truncate the first session."""
    path = tmp_path / "once.journal"
    Journal.create(path, engine_version=ENGINE).close()

    with pytest.raises(FileExistsError):
        Journal.create(path, engine_version=ENGINE)


def test_a_closed_journal_refuses_further_appends():
    journal = Journal.in_memory(engine_version=ENGINE)
    journal.append(KIND_SEAT_JOINED, 1, {"seat": "ann", "cash_minor": 1})
    journal.close()

    with pytest.raises(JournalError, match="closed"):
        journal.append(KIND_SEAT_JOINED, 2, {"seat": "bob", "cash_minor": 1})


@pytest.mark.parametrize("kind", ["", "Order", "order submitted", "x" * 33, "1st"])
def test_a_malformed_kind_is_refused(kind):
    """The kind is 1 to 32 lowercase ascii bytes and is length-prefixed, so a
    kind that does not fit the shape would not survive its own framing."""
    journal = Journal.in_memory(engine_version=ENGINE)
    with pytest.raises(ValueError, match="kind must be"):
        journal.append(kind, 1_000, {})


# ---------------------------------------------------------------------------
# independence from the venue
# ---------------------------------------------------------------------------


def test_journal_does_not_import_the_venue():
    """The property that lets every test above run without building a market.

    Asserted on the source rather than on behaviour, because the dependency this
    guards against would arrive as a convenience import that all these tests
    would still pass with.
    """
    import arena.sim.journal as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("arena.market", "arena.exchange", "arena.sim.kernel", "arena.portfolio"):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------


def test_a_hundred_thousand_records_round_trip(tmp_path):
    """Measured on this machine: 100,000 records write in 363ms and replay in
    245ms, a 605ms round trip, over a 10.9MB file at 109.3 bytes per record.

    That is 276,000 appends and 400,000 replayed records per second, so a session
    of any length this simulator produces recovers in well under a second. The
    assertion is deliberately loose at 60 seconds -- it is there to catch an
    accidental O(n^2), not to fail on a slow machine, and the number that means
    something is the one in this docstring.
    """
    path = tmp_path / "big.journal"
    total = 100_000
    kinds = [KIND_SEAT_JOINED, KIND_ORDER_SUBMITTED, KIND_ORDER_CANCELLED]

    started = time.perf_counter()
    with Journal.create(path, engine_version=ENGINE, durability="none") as journal:
        for i in range(total):
            kind = kinds[i % 3]
            if kind == KIND_SEAT_JOINED:
                payload = {"seat": f"trader-{i}", "cash_minor": 100_000_000}
            elif kind == KIND_ORDER_SUBMITTED:
                payload = {
                    "seat": f"trader-{i % 97}",
                    "client_order_id": f"c-{i}",
                    "symbol": SYMBOL,
                    "side": "buy" if i % 2 else "sell",
                    "price_ticks": 4_600 + (i % 200),
                    "quantity": 1 + (i % 9),
                }
            else:
                payload = {"seat": f"trader-{i % 97}", "client_order_id": f"c-{i - 1}"}
            journal.append(kind, 1_000_000 * (i + 1), payload)
    write_seconds = time.perf_counter() - started

    seen = 0
    last = 0

    def apply(record: Record) -> None:
        nonlocal seen, last
        seen += 1
        assert record.sequence == seen
        last = record.timestamp_ns

    started = time.perf_counter()
    result = replay(path, apply, engine_version=ENGINE)
    replay_seconds = time.perf_counter() - started

    assert seen == total
    assert result.applied == total
    assert result.last_sequence == total
    assert last == 1_000_000 * total
    assert result.torn is False
    assert write_seconds + replay_seconds < 60.0

    # And it is still deterministic at this size, which is where a reader that
    # buffers or chunks differently on a large file would show up.
    again = replay(path, lambda _r: None, engine_version=ENGINE)
    assert again.stream_digest == result.stream_digest


def test_durability_costs(tmp_path):
    """Named by the module docstring, and the reason ``flush`` is the default.

    Measured on this machine over 2,000 appends: durability="none" 353,000/s,
    "flush" 205,000/s, "fsync" 9,100/s. An fsync per order is a real disk round
    trip and costs roughly 22x, which is the price of surviving power loss rather
    than merely surviving the process dying -- and the process dying is the
    failure this module exists for.

    Asserted only on ordering, not on the ratio, because the ratio is a property
    of the disk under the test runner and would be a flaky assertion on anyone
    else's machine.
    """
    timings = {}
    for policy in ("none", "flush", "fsync"):
        path = tmp_path / f"{policy}.journal"
        started = time.perf_counter()
        with Journal.create(path, engine_version=ENGINE, durability=policy) as journal:
            for i in range(500):
                journal.append(KIND_ORDER_SUBMITTED, 1_000 * (i + 1), {"i": i})
        timings[policy] = time.perf_counter() - started
        assert len(list(read_records(path, engine_version=ENGINE))) == 500

    assert timings["fsync"] > timings["none"], "an fsync per record must cost something"

    with pytest.raises(ValueError, match="unknown durability"):
        Journal.create(tmp_path / "bad.journal", engine_version=ENGINE, durability="maybe")
