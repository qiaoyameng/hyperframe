import struct

import pytest

from hyperframe.envelope import FrameEnvelope
from hyperframe.exceptions import (
    InvalidFrameError,
    InvalidPaddingError,
    UnknownFrameError,
)
from hyperframe.frame import DataFrame, ExtensionFrame, SettingsFrame


def make_frame(frame_type, flags, stream_id, body, reserved_bit=False):
    stream_word = stream_id | (0x80000000 if reserved_bit else 0)
    header = struct.pack(
        ">HBBBL",
        (len(body) >> 8) & 0xFFFF,
        len(body) & 0xFF,
        frame_type,
        flags,
        stream_word,
    )
    return header + body


class TestEnvelopeParsing:
    def test_unedited_envelope_forwards_identical_bytes(self):
        data = make_frame(0x0, 0x0, 1, b"hello")
        envelope = FrameEnvelope.parse(data)
        assert envelope.serialize() == data
        assert envelope.raw == data
        assert envelope.raw_header == data[:9]
        assert envelope.raw_body == b"hello"
        assert not envelope.modified

    def test_parse_accepts_bytes_like_objects(self):
        data = make_frame(0x0, 0x0, 1, b"hello")
        assert FrameEnvelope.parse(bytearray(data)).serialize() == data
        assert FrameEnvelope.parse(memoryview(data)).serialize() == data

    def test_frame_metadata_is_exposed(self):
        data = make_frame(0x0, 0x1, 1, b"hello")
        envelope = FrameEnvelope.parse(data)
        assert envelope.type_byte == 0x0
        assert envelope.length == 5
        assert envelope.stream_id == 1
        assert envelope.flags == {"END_STREAM"}
        assert isinstance(envelope.frame, DataFrame)
        assert envelope.frame.data == b"hello"

    def test_repr(self):
        data = make_frame(0x0, 0x0, 1, b"hello")
        envelope = FrameEnvelope.parse(data)
        assert repr(envelope) == (
            "FrameEnvelope(type=0x00, length=5, stream_id=1, modified=False)"
        )

    def test_truncated_header_is_rejected(self):
        with pytest.raises(InvalidFrameError):
            FrameEnvelope.parse(b"\x00\x00\x05\x00")

    def test_truncated_body_is_rejected(self):
        data = make_frame(0x0, 0x0, 1, b"hello")
        with pytest.raises(InvalidFrameError):
            FrameEnvelope.parse(data[:-2])

    def test_trailing_data_is_rejected(self):
        data = make_frame(0x0, 0x0, 1, b"hello")
        with pytest.raises(InvalidFrameError):
            FrameEnvelope.parse(data + b"\x00")

    def test_unknown_frame_type_is_preserved(self):
        data = make_frame(0xFF, 0x13, 0, b"\xde\xad\xbe\xef")
        envelope = FrameEnvelope.parse(data)
        assert isinstance(envelope.frame, ExtensionFrame)
        assert envelope.frame.type == 0xFF
        assert envelope.serialize() == data

    def test_unknown_frame_type_strict_is_rejected(self):
        data = make_frame(0xFF, 0x13, 0, b"\xde\xad\xbe\xef")
        with pytest.raises(UnknownFrameError):
            FrameEnvelope.parse(data, strict=True)

    def test_invalid_body_is_rejected(self):
        # A pad length larger than the body is invalid padding.
        data = make_frame(0x0, 0x8, 1, b"\x05ab")
        with pytest.raises(InvalidPaddingError):
            FrameEnvelope.parse(data)

    def test_constructor_validates_header_length(self):
        with pytest.raises(ValueError):
            FrameEnvelope(DataFrame(1), b"\x00" * 8, b"")


class TestDuplicateSettings:
    def test_duplicate_settings_are_forwarded_verbatim(self):
        body = struct.pack(">HLHL", 0x1, 100, 0x1, 200)
        data = make_frame(0x4, 0x0, 0, body)
        envelope = FrameEnvelope.parse(data)
        # The analysis view merges duplicates into a dictionary...
        assert isinstance(envelope.frame, SettingsFrame)
        assert envelope.frame.settings == {0x1: 200}
        # ...but the forwarded bytes keep both entries, unlike a
        # re-serialization of the parsed frame.
        assert envelope.frame.serialize() != data
        assert envelope.serialize() == data


class TestNonZeroPadding:
    def test_non_zero_padding_is_forwarded_verbatim(self):
        body = b"\x03" + b"hi" + b"\xab\xab\xab"
        data = make_frame(0x0, 0x8, 1, body)
        envelope = FrameEnvelope.parse(data)
        assert envelope.frame.data == b"hi"
        assert envelope.frame.pad_length == 3
        # Re-serializing the parsed frame would zero out the padding.
        assert envelope.frame.serialize() != data
        assert envelope.serialize() == data


class TestUnknownFlags:
    def test_unknown_flag_bits_survive_forwarding(self):
        # 0x40 is not a defined DATA flag; 0x01 is END_STREAM.
        data = make_frame(0x0, 0x41, 1, b"hi")
        envelope = FrameEnvelope.parse(data)
        assert envelope.flags == {"END_STREAM"}
        assert envelope.serialize() == data

    def test_editing_known_flags_preserves_unknown_bits(self):
        data = make_frame(0x0, 0x41, 1, b"hi")
        envelope = FrameEnvelope.parse(data)
        envelope.flags = set()
        assert envelope.serialize()[4] == 0x40
        assert envelope.flags == set()
        assert set(envelope.frame.flags) == set()
        assert envelope.modified
        envelope.flags = {"END_STREAM", "PADDED"}
        assert envelope.serialize()[4] == 0x49
        assert envelope.flags == {"END_STREAM", "PADDED"}
        assert set(envelope.frame.flags) == {"END_STREAM", "PADDED"}
        # The payload is untouched by flag edits.
        assert envelope.serialize()[9:] == b"hi"

    def test_unknown_flag_name_is_rejected(self):
        data = make_frame(0x0, 0x41, 1, b"hi")
        envelope = FrameEnvelope.parse(data)
        with pytest.raises(ValueError):
            envelope.flags = {"NOT_A_FLAG"}


class TestStreamIdEdits:
    def test_reserved_bit_is_preserved_on_forwarding(self):
        data = make_frame(0x0, 0x0, 11, b"hi", reserved_bit=True)
        envelope = FrameEnvelope.parse(data)
        assert envelope.stream_id == 11
        assert envelope.serialize() == data

    def test_stream_id_edit_preserves_reserved_bit(self):
        data = make_frame(0x0, 0x0, 11, b"hi", reserved_bit=True)
        envelope = FrameEnvelope.parse(data)
        envelope.stream_id = 7
        (stream_word,) = struct.unpack(">L", envelope.serialize()[5:9])
        assert stream_word == 0x80000007
        assert envelope.stream_id == 7
        assert envelope.frame.stream_id == 7
        assert envelope.modified

    @pytest.mark.parametrize("value", [-1, 2**31])
    def test_invalid_stream_id_is_rejected(self, value):
        data = make_frame(0x0, 0x0, 11, b"hi")
        envelope = FrameEnvelope.parse(data)
        with pytest.raises(ValueError):
            envelope.stream_id = value


class TestAnalysisMutationsAreIsolated:
    def test_mutating_the_analysis_frame_does_not_change_output(self):
        data = make_frame(0x0, 0x0, 1, b"hello")
        envelope = FrameEnvelope.parse(data)
        envelope.frame.data = b"tampered"
        envelope.frame.stream_id = 99
        envelope.frame.flags.add("END_STREAM")
        assert envelope.serialize() == data
        assert not envelope.modified


class TestReplaceFrame:
    def test_replacement_regenerates_a_valid_length(self):
        data = make_frame(0x0, 0x0, 1, b"hello")
        envelope = FrameEnvelope.parse(data)
        replacement = DataFrame(stream_id=3, data=b"a much longer payload")
        replacement.flags.add("END_STREAM")
        envelope.replace_frame(replacement)
        assert envelope.serialize() == replacement.serialize()
        assert int.from_bytes(envelope.serialize()[:3], "big") == len(
            b"a much longer payload"
        )
        assert envelope.modified
        assert envelope.frame is replacement
        # The original wire bytes are still available, but not emitted.
        assert envelope.raw == data

    def test_envelope_edits_apply_to_the_replacement(self):
        data = make_frame(0x0, 0x0, 1, b"hello")
        envelope = FrameEnvelope.parse(data)
        envelope.replace_frame(DataFrame(stream_id=3, data=b"payload"))
        envelope.stream_id = 4
        envelope.flags = {"END_STREAM"}
        assert envelope.frame.stream_id == 4
        assert envelope.serialize()[4] == 0x01
        (stream_word,) = struct.unpack(">L", envelope.serialize()[5:9])
        assert stream_word == 4
