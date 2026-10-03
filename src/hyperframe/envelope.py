"""
Wire-format-preserving frame envelopes.

This module provides :class:`FrameEnvelope`, a wrapper around exactly one
serialized HTTP/2 frame (9-byte header plus payload) for use in
man-in-the-middle analysis and forwarding tools.

Parsing a frame into a :class:`~hyperframe.frame.Frame` and serializing it
again is lossy: unknown flag bits and the reserved stream-ID bit are dropped,
padding bytes are normalized to zero, and duplicate SETTINGS entries are
collapsed into a dictionary. ``FrameEnvelope`` avoids that by keeping the
original header and payload bytes next to the parsed frame:

- The parsed frame is exposed as a read-only-by-convention analysis view.
  Mutating it never changes the forwarded bytes.
- Unmodified envelopes serialize back to the exact original bytes.
- The stream ID and known (spec-defined) flags can be patched explicitly.
  All other header bits and the payload are left untouched.
- The semantic frame can be replaced wholesale, in which case the output is
  fully re-serialized with a regenerated, valid length field.

The envelope deliberately does not implement any connection state machine:
it does not validate whether a frame or a modification is legal in the
current connection state. Its parsing strictness is limited to the wire
format itself:

- Fewer than 9 bytes: truncated header, raises
  :class:`~hyperframe.exceptions.InvalidFrameError`.
- Fewer bytes than the header's length field demands: truncated body,
  raises :class:`~hyperframe.exceptions.InvalidFrameError`.
- More bytes than the header's length field demands: the envelope covers
  exactly one frame, so trailing bytes raise
  :class:`~hyperframe.exceptions.InvalidFrameError`. Split a stream of
  frames into individual frames before wrapping them.
- Unknown frame types raise
  :class:`~hyperframe.exceptions.UnknownFrameError` in strict mode and are
  wrapped in :class:`~hyperframe.frame.ExtensionFrame` otherwise.
- Bodies are parsed with hyperframe's own validation; malformed bodies
  raise the corresponding hyperframe exception.
"""
from __future__ import annotations

from .exceptions import InvalidDataError, InvalidFrameError
from .frame import Frame

_HEADER_LEN = 9
_STREAM_ID_MASK = 0x7FFFFFFF
_RESERVED_BIT = 0x80


class FrameEnvelope:
    """
    A single HTTP/2 frame that preserves its original wire format.

    Use :meth:`parse` to wrap received bytes, inspect :attr:`frame` for
    analysis, optionally apply explicit modifications, and call
    :meth:`serialize` to obtain the bytes to forward.
    """

    def __init__(self, header: bytes, body: bytes, frame: Frame) -> None:
        if len(header) != _HEADER_LEN:
            msg = f"Frame header must be exactly {_HEADER_LEN} bytes: got {len(header)}"
            raise InvalidFrameError(msg)
        #: The original 9-byte frame header, as received.
        self.raw_header = header
        #: The original frame payload, as received.
        self.raw_body = body
        self._header = bytearray(header)
        self._frame = frame
        self._replacement: Frame | None = None

    @classmethod
    def parse(cls, data: bytes | memoryview, strict: bool = False) -> FrameEnvelope:
        """
        Wrap exactly one serialized frame (header plus payload) in an
        envelope.

        :param data: The raw bytes of a single complete frame. Must contain
            neither fewer nor more bytes than the frame occupies on the
            wire.

        :param strict: Whether to raise
            :class:`~hyperframe.exceptions.UnknownFrameError` for frame
            types not defined by the specification. When ``False`` (the
            default), unknown frames are wrapped in
            :class:`~hyperframe.frame.ExtensionFrame` and forwarded
            byte-identically.

        :raises hyperframe.exceptions.InvalidFrameError: If the data is
            truncated or contains trailing bytes beyond a single frame.

        :raises hyperframe.exceptions.UnknownFrameError: If ``strict`` is
            set and the frame type is unknown.
        """
        data = bytes(data)
        if len(data) < _HEADER_LEN:
            msg = f"Truncated frame header: got {len(data)} of {_HEADER_LEN} bytes"
            raise InvalidFrameError(msg)

        header = data[:_HEADER_LEN]
        frame, length = Frame.parse_frame_header(memoryview(header), strict=strict)

        body = data[_HEADER_LEN:]
        if len(body) < length:
            msg = f"Truncated frame body: header promises {length} bytes, got {len(body)}"
            raise InvalidFrameError(msg)
        if len(body) > length:
            msg = f"Trailing data after frame: header promises {length} bytes, got {len(body)}"
            raise InvalidFrameError(msg)

        frame.parse_body(memoryview(body))
        return cls(header, body, frame)

    @property
    def frame(self) -> Frame:
        """
        The parsed frame, for analysis only.

        This reflects the frame as received. Mutating it has no effect on
        the bytes produced by :meth:`serialize`; use :meth:`set_stream_id`,
        :meth:`set_flag`, :meth:`clear_flag`, or :meth:`replace_frame` to
        change the forwarded bytes. After :meth:`replace_frame`, this
        property returns the replacement frame.
        """
        if self._replacement is not None:
            return self._replacement
        return self._frame

    @property
    def type(self) -> int:
        """The frame type byte."""
        return self._header[3]

    @property
    def stream_id(self) -> int:
        """The stream ID currently carried by the envelope."""
        if self._replacement is not None:
            return self._replacement.stream_id
        return int.from_bytes(self._header[5:9], "big") & _STREAM_ID_MASK

    @property
    def body_len(self) -> int:
        """The length of the payload currently carried by the envelope."""
        if self._replacement is not None:
            return len(self._replacement.serialize_body())
        return len(self.raw_body)

    @property
    def modified(self) -> bool:
        """Whether any explicit modification has been applied."""
        return self._replacement is not None or bytes(self._header) != self.raw_header

    def set_stream_id(self, stream_id: int) -> None:
        """
        Rewrite the 31-bit stream ID in the frame header.

        The reserved high bit of the stream ID field is preserved as
        received. The payload is untouched. Cannot be combined with
        :meth:`replace_frame`.

        :raises hyperframe.exceptions.InvalidDataError: If the stream ID is
            out of range or a replacement frame has been set.
        """
        self._ensure_not_replaced("set_stream_id")
        if not 0 <= stream_id <= _STREAM_ID_MASK:
            msg = f"Stream ID must be between 0 and 2**31 - 1: got {stream_id}"
            raise InvalidDataError(msg)
        reserved = self._header[5] & _RESERVED_BIT
        packed = stream_id.to_bytes(4, "big")
        self._header[5:9] = bytes([packed[0] | reserved]) + packed[1:]

    def set_flag(self, name: str) -> None:
        """
        Set a known (spec-defined) flag bit in the frame header.

        Flag bits not defined for this frame type are preserved as
        received. The payload is untouched. Cannot be combined with
        :meth:`replace_frame`.

        :raises hyperframe.exceptions.InvalidDataError: If the flag is not
            defined for this frame type or a replacement frame has been
            set.
        """
        self._header[4] |= self._flag_bit(name)

    def clear_flag(self, name: str) -> None:
        """
        Clear a known (spec-defined) flag bit in the frame header.

        Flag bits not defined for this frame type are preserved as
        received. The payload is untouched. Cannot be combined with
        :meth:`replace_frame`.

        :raises hyperframe.exceptions.InvalidDataError: If the flag is not
            defined for this frame type or a replacement frame has been
            set.
        """
        self._header[4] &= ~self._flag_bit(name) & 0xFF

    def replace_frame(self, frame: Frame) -> None:
        """
        Replace the semantic frame wholesale.

        The output is fully re-serialized from the given frame with a
        regenerated, valid length field. The original wire format
        (including unknown flag bits, the reserved stream-ID bit, padding
        bytes, and duplicate SETTINGS entries) is discarded. Any previous
        header patches are discarded as well.
        """
        self._replacement = frame

    def serialize(self) -> bytes:
        """
        Return the bytes to forward.

        Without explicit modifications this is byte-identical to the
        original input. Header patches only touch the patched bits; the
        payload is forwarded as received. After :meth:`replace_frame`, the
        replacement frame is serialized instead.
        """
        if self._replacement is not None:
            return self._replacement.serialize()
        return bytes(self._header) + self.raw_body

    def _flag_bit(self, name: str) -> int:
        self._ensure_not_replaced("flag modification")
        for flag in self._frame.defined_flags:
            if flag.name == name:
                return flag.bit
        msg = f"Flag {name!r} is not defined for {type(self._frame).__name__}"
        raise InvalidDataError(msg)

    def _ensure_not_replaced(self, operation: str) -> None:
        if self._replacement is not None:
            msg = f"Cannot apply {operation} after replace_frame: modify the replacement frame instead"
            raise InvalidDataError(msg)

    def __repr__(self) -> str:
        state = "modified" if self.modified else "unmodified"
        return f"FrameEnvelope({state}, {self.frame!r})"
