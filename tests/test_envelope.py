import struct

import pytest

from hyperframe.envelope import FrameEnvelope
from hyperframe.exceptions import (
    InvalidDataError,
    InvalidFrameError,
    UnknownFrameError,
)
from hyperframe.frame import DataFrame, ExtensionFrame, SettingsFrame


def raw_frame(frame_type, flags, stream_id, body):
    header = struct.pack(
        ">HBBBL",
        (len(body) >> 8) & 0xFFFF,
        len(body) & 0xFF,
        frame_type,
        flags,
        stream_id,
    )
    return header + body


def duplicate_settings_raw():
    # INITIAL_WINDOW_SIZE (0x04) appears twice: a dict merge would drop one.
    body = struct.pack(">HLHL", 0x04, 100, 0x04, 200)
    return raw_frame(0x04, 0x00, 0, body)


def padded_data_raw():
    # PADDED DATA frame with non-zero padding bytes.
    body = b"\x03" + b"hi" + b"\xaa\xbb\xcc"
    return raw_frame(0x00, 0x08, 1, body)


class TestWireFormatPreservation:
    def test_duplicate_settings_forwarded_byte_identical(self):
        raw = duplicate_settings_raw()
        envelope = FrameEnvelope.parse(raw)
        assert envelope.serialize() == raw
        assert not envelope.modified
        # The analysis view shows the dict-merged settings (last wins)...
        assert envelope.frame.settings == {0x04: 200}
        # ...while a naive parse/serialize round-trip would lose an entry.
        assert envelope.frame.serialize() != raw

    def test_nonzero_padding_forwarded_byte_identical(self):
        raw = padded_data_raw()
        envelope = FrameEnvelope.parse(raw)
        assert envelope.serialize() == raw
        # hyperframe would normalize the padding to zero bytes.
        assert envelope.frame.serialize() != raw
        assert envelope.frame.data == b"hi"
        assert envelope.frame.pad_length == 3

    def test_unknown_flags_forwarded_byte_identical(self):
        # END_STREAM plus undefined flag bit 0x80.
        raw = raw_frame(0x00, 0x81, 1, b"hello")
        envelope = FrameEnvelope.parse(raw)
        assert envelope.serialize() == raw
        assert "END_STREAM" in envelope.frame.flags
        # hyperframe drops the unknown bit when re-serializing.
        assert envelope.frame.serialize() != raw

    def test_reserved_stream_id_bit_forwarded_byte_identical(self):
        raw = raw_frame(0x00, 0x00, 0x80000001, b"hello")
        envelope = FrameEnvelope.parse(raw)
        assert envelope.serialize() == raw
        assert envelope.frame.stream_id == 1
        assert envelope.stream_id == 1

    def test_unknown_frame_type_forwarded_byte_identical(self):
        raw = raw_frame(0xFF, 0xA5, 0xDEADBEEF, b"\x01\x02\x03")
        envelope = FrameEnvelope.parse(raw)
        assert isinstance(envelope.frame, ExtensionFrame)
        assert envelope.serialize() == raw

    def test_accepts_memoryview(self):
        raw = duplicate_settings_raw()
        envelope = FrameEnvelope.parse(memoryview(raw))
        assert envelope.serialize() == raw


class TestExplicitModification:
    def test_set_stream_id_preserves_reserved_bit_and_payload(self):
        raw = raw_frame(0x00, 0x81, 0x80000001, b"hello")
        envelope = FrameEnvelope.parse(raw)
        envelope.set_stream_id(5)
        out = envelope.serialize()
        assert envelope.modified
        assert envelope.stream_id == 5
        # Reserved bit kept, stream ID replaced, everything else untouched.
        assert out == raw_frame(0x00, 0x81, 0x80000005, b"hello")

    @pytest.mark.parametrize("stream_id", [-1, 2**31])
    def test_set_stream_id_out_of_range(self, stream_id):
        envelope = FrameEnvelope.parse(raw_frame(0x00, 0x00, 1, b"x"))
        with pytest.raises(InvalidDataError):
            envelope.set_stream_id(stream_id)
        assert not envelope.modified

    def test_set_and_clear_known_flag_preserve_unknown_bits(self):
        raw = raw_frame(0x00, 0x80, 1, b"hello")
        envelope = FrameEnvelope.parse(raw)
        envelope.set_flag("END_STREAM")
        assert envelope.serialize() == raw_frame(0x00, 0x81, 1, b"hello")
        envelope.clear_flag("END_STREAM")
        assert envelope.serialize() == raw
        assert not envelope.modified

    def test_set_flag_on_duplicate_settings_keeps_duplicates(self):
        raw = duplicate_settings_raw()
        envelope = FrameEnvelope.parse(raw)
        envelope.set_flag("ACK")
        out = envelope.serialize()
        assert out[:5] == raw[:3] + b"\x04\x01"
        assert out[9:] == raw[9:]

    def test_set_unknown_flag_raises(self):
        envelope = FrameEnvelope.parse(raw_frame(0x00, 0x00, 1, b"x"))
        with pytest.raises(InvalidDataError):
            envelope.set_flag("NOT_A_FLAG")
        with pytest.raises(InvalidDataError):
            envelope.clear_flag("NOT_A_FLAG")
        assert not envelope.modified

    def test_set_flag_on_extension_frame_raises(self):
        envelope = FrameEnvelope.parse(raw_frame(0xFF, 0x00, 1, b"x"))
        with pytest.raises(InvalidDataError):
            envelope.set_flag("END_STREAM")

    def test_analysis_view_mutation_does_not_leak_into_output(self):
        raw = padded_data_raw()
        envelope = FrameEnvelope.parse(raw)
        envelope.frame.data = b"tampered"
        envelope.frame.flags.add("END_STREAM")
        envelope.frame.stream_id = 99
        assert envelope.serialize() == raw
        assert not envelope.modified

    def test_replace_frame_regenerates_valid_length(self):
        raw = raw_frame(0x00, 0x81, 0x80000001, b"hi")
        envelope = FrameEnvelope.parse(raw)
        envelope.set_flag("PADDED")  # discarded by the replacement
        replacement = DataFrame(stream_id=3, data=b"a much longer body")
        envelope.replace_frame(replacement)
        out = envelope.serialize()
        assert envelope.frame is replacement
        assert envelope.stream_id == 3
        assert envelope.body_len == len(b"a much longer body")
        assert out == replacement.serialize()
        assert out != raw
        # The regenerated header carries a correct length field.
        assert int.from_bytes(out[0:3], "big") == len(out) - 9

    def test_header_patches_rejected_after_replace(self):
        envelope = FrameEnvelope.parse(raw_frame(0x00, 0x00, 1, b"x"))
        envelope.replace_frame(DataFrame(stream_id=1, data=b"y"))
        with pytest.raises(InvalidDataError):
            envelope.set_stream_id(2)
        with pytest.raises(InvalidDataError):
            envelope.set_flag("END_STREAM")
        with pytest.raises(InvalidDataError):
            envelope.clear_flag("END_STREAM")

    def test_repr(self):
        envelope = FrameEnvelope.parse(raw_frame(0x00, 0x00, 1, b"x"))
        assert "unmodified" in repr(envelope)
        envelope.set_stream_id(2)
        assert "modified" in repr(envelope)


class TestParsingStrictness:
    def test_truncated_header(self):
        with pytest.raises(InvalidFrameError, match="Truncated frame header"):
            FrameEnvelope.parse(b"\x00\x00\x01")

    def test_truncated_body(self):
        raw = raw_frame(0x00, 0x00, 1, b"hello")
        with pytest.raises(InvalidFrameError, match="Truncated frame body"):
            FrameEnvelope.parse(raw[:-2])

    def test_trailing_bytes_rejected(self):
        raw = raw_frame(0x00, 0x00, 1, b"hello") + b"extra"
        with pytest.raises(InvalidFrameError, match="Trailing data"):
            FrameEnvelope.parse(raw)

    def test_unknown_frame_type_strict(self):
        raw = raw_frame(0xFF, 0x00, 1, b"x")
        with pytest.raises(UnknownFrameError):
            FrameEnvelope.parse(raw, strict=True)

    def test_malformed_body_raises(self):
        # RST_STREAM requires exactly 4 body bytes.
        raw = raw_frame(0x03, 0x00, 1, b"\x00\x00")
        with pytest.raises(InvalidFrameError):
            FrameEnvelope.parse(raw)

    def test_constructor_rejects_bad_header_length(self):
        with pytest.raises(InvalidFrameError):
            FrameEnvelope(b"\x00" * 8, b"", DataFrame(stream_id=1))

    def test_envelope_properties(self):
        raw = duplicate_settings_raw()
        envelope = FrameEnvelope.parse(raw)
        assert envelope.type == 0x04
        assert envelope.stream_id == 0
        assert envelope.body_len == 12
        assert envelope.raw_header == raw[:9]
        assert envelope.raw_body == raw[9:]
        assert isinstance(envelope.frame, SettingsFrame)
