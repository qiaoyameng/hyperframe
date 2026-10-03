"""
Wire-format-preserving envelopes for HTTP/2 frames.

This module provides :class:`FrameEnvelope`, a wrapper that pairs a parsed
:class:`hyperframe.frame.Frame` (used for inspection and analysis) with the
exact bytes that frame was read from. It exists for tooling that sits between
two HTTP/2 peers: such tools need to look at frames, but must forward them
byte-for-byte unless an edit was explicitly requested.

Re-serializing a parsed frame loses information: unknown flag bits are
dropped, the reserved bit of the stream identifier is masked off, padding
bytes are rewritten as zeroes, and a duplicate SETTINGS entry is collapsed
into a single dictionary item. The envelope avoids all of that by emitting
the original wire bytes until the caller explicitly edits the stream ID or
the known flags, or replaces the semantic frame wholesale.

The envelope deliberately knows nothing about the HTTP/2 connection state
machine: it does not track streams, SETTINGS negotiation, or flow control.
It only shuttles bytes and structured views of single frames.

.. versionadded:: 6.2.0
"""
from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from .exceptions import InvalidFrameError
from .flags import Flags
from .frame import Frame

if TYPE_CHECKING:
    from collections.abc import Iterable

_STRUCT_L = struct.Struct(">L")

#: The length in bytes of an HTTP/2 frame header, RFC 9113, Section 4.1.
HEADER_LEN = 9

#: The reserved high bit of the 32-bit stream identifier field.
_RESERVED_BIT = 0x80000000

#: The mask for the 31-bit stream identifier, RFC 9113, Section 4.1.
_STREAM_ID_MASK = 0x7FFFFFFF


class FrameEnvelope:
    """
    A wire-format-preserving wrapper around a single HTTP/2 frame.

    An envelope is created by parsing exactly one frame off the wire with
    :meth:`parse`. It then exposes two views of the same frame:

    - The semantic view, :attr:`frame`, is a regular parsed frame object
      suitable for analysis. Mutating it has *no* effect on the bytes the
      envelope emits: analysis can never silently alter what is forwarded.
    - The wire view is the original header and payload bytes, available via
      :attr:`raw_header` and :attr:`raw_body`. As long as no explicit edit
      has been made, :meth:`serialize` returns those bytes unchanged.

    Explicit edits are possible at the envelope level only:

    - Assigning :attr:`stream_id` rewrites the stream identifier in the
      emitted header, preserving the reserved high bit and the payload.
    - Assigning :attr:`flags` rewrites the *known* flag bits for the frame
      type in the emitted header, preserving all unknown flag bits and the
      payload.
    - Calling :meth:`replace_frame` discards the wire bytes entirely and
      emits the replacement frame's own serialization, which regenerates a
      valid length field for the new payload.

    The envelope does not implement any part of the HTTP/2 connection state
    machine: validating whether a frame (edited or not) is legal at a given
    point in a connection is the caller's responsibility.
    """

    def __init__(self, frame: Frame, raw_header: bytes, raw_body: bytes) -> None:
        if len(raw_header) != HEADER_LEN:
            msg = f"Raw header must be exactly {HEADER_LEN} bytes, got {len(raw_header)}"
            raise ValueError(msg)

        self._frame = frame
        self._raw_header = raw_header
        self._raw_body = raw_body
        self._stream_id_override: int | None = None
        self._known_flags_override: set[str] | None = None
        self._replaced = False

    @classmethod
    def parse(cls, data: bytes | bytearray | memoryview, strict: bool = False) -> FrameEnvelope:
        """
        Parse exactly one frame from ``data`` into an envelope.

        The parsing policy is strict about framing boundaries:

        - Fewer bytes than the 9-byte header, or fewer body bytes than the
          length declared in the header, raise
          :class:`hyperframe.exceptions.InvalidFrameError` (truncation).
        - More bytes than the declared length (i.e. ``data`` is not exactly
          one frame) also raise
          :class:`hyperframe.exceptions.InvalidFrameError`. Callers reading
          from a stream must slice off exactly one frame themselves, using
          the length in the 9-byte header.
        - Frames of unknown type are wrapped in
          :class:`hyperframe.frame.ExtensionFrame` and their bytes are
          preserved, unless ``strict`` is true, in which case
          :class:`hyperframe.exceptions.UnknownFrameError` is raised.
        - Semantically invalid bodies (for example invalid padding) raise
          the usual hyperframe exceptions: an envelope is only created for
          frames hyperframe can parse.

        :param data: The bytes of exactly one complete frame, header and
                     body.
        :param strict: Whether to raise an exception when encountering a
            frame type not defined by the spec and implemented by
            hyperframe.

        :raises hyperframe.exceptions.InvalidFrameError: If the data is
            truncated or does not contain exactly one frame.
        :raises hyperframe.exceptions.UnknownFrameError: If ``strict`` is
            true and the frame type is unknown.
        """
        view = memoryview(data)
        if len(view) < HEADER_LEN:
            msg = f"Truncated frame header: expected {HEADER_LEN} bytes, got {len(view)}"
            raise InvalidFrameError(msg)

        frame, length = Frame.parse_frame_header(view[:HEADER_LEN], strict=strict)

        expected = HEADER_LEN + length
        if len(view) < expected:
            msg = (
                f"Truncated frame body: header declares {length} bytes, "
                f"but only {len(view) - HEADER_LEN} are present"
            )
            raise InvalidFrameError(msg)
        if len(view) > expected:
            msg = (
                f"Trailing data after frame: header declares {length} body bytes, "
                f"but {len(view) - HEADER_LEN} are present"
            )
            raise InvalidFrameError(msg)

        raw_body = view[HEADER_LEN:].tobytes()
        frame.parse_body(memoryview(raw_body))
        return cls(frame, view[:HEADER_LEN].tobytes(), raw_body)

    def __repr__(self) -> str:
        return (
            f"FrameEnvelope(type=0x{self.type_byte:02X}, length={self.length}, "
            f"stream_id={self.stream_id}, modified={self.modified})"
        )

    @property
    def frame(self) -> Frame:
        """
        The semantic frame, parsed from the wire bytes.

        This object is a view for analysis. Mutating it does not change the
        bytes emitted by :meth:`serialize`; only edits made through the
        envelope itself (or :meth:`replace_frame`) do. Edits made through
        the envelope are reflected back into this object so that it always
        shows the values that will be emitted.
        """
        return self._frame

    @property
    def raw_header(self) -> bytes:
        """
        The 9-byte frame header exactly as read from the wire, without any
        envelope edits applied.
        """
        return self._raw_header

    @property
    def raw_body(self) -> bytes:
        """
        The frame payload exactly as read from the wire, without any
        envelope edits applied.
        """
        return self._raw_body

    @property
    def raw(self) -> bytes:
        """
        The complete frame (header and body) exactly as read from the wire,
        without any envelope edits applied.
        """
        return self._raw_header + self._raw_body

    @property
    def length(self) -> int:
        """
        The payload length declared in the wire header.
        """
        return (self._raw_header[0] << 16) | (self._raw_header[1] << 8) | self._raw_header[2]

    @property
    def type_byte(self) -> int:
        """
        The frame type byte from the wire header.
        """
        return self._raw_header[3]

    @property
    def stream_id(self) -> int:
        """
        The stream identifier that will be emitted by :meth:`serialize`.
        """
        if self._stream_id_override is not None:
            return self._stream_id_override
        return self._frame.stream_id

    @stream_id.setter
    def stream_id(self, value: int) -> None:
        if not 0 <= value <= _STREAM_ID_MASK:
            msg = f"Stream ID must be between 0 and {_STREAM_ID_MASK}, got {value}"
            raise ValueError(msg)

        if self._replaced:
            self._frame.stream_id = value
        else:
            self._stream_id_override = value
            self._frame.stream_id = value

    @property
    def flags(self) -> set[str]:
        """
        The known flags that will be emitted by :meth:`serialize`, as a set
        of flag names.
        """
        if self._known_flags_override is not None:
            return set(self._known_flags_override)
        return set(self._frame.flags)

    @flags.setter
    def flags(self, names: Iterable[str]) -> None:
        validated = Flags(self._frame.defined_flags)
        for name in names:
            validated.add(name)

        if self._replaced:
            self._frame.flags = validated
        else:
            self._known_flags_override = set(validated)
            self._frame.flags = validated

    @property
    def modified(self) -> bool:
        """
        Whether any explicit edit has been made through this envelope.
        """
        return (
            self._replaced
            or self._stream_id_override is not None
            or self._known_flags_override is not None
        )

    def replace_frame(self, frame: Frame) -> None:
        """
        Replace the semantic frame wholesale.

        After this call, :meth:`serialize` emits the replacement frame's own
        serialization, which regenerates a valid length field for the new
        payload. The original wire bytes remain available via
        :attr:`raw_header` and :attr:`raw_body` but are no longer emitted.

        :param frame: The frame to emit instead of the parsed one.
        """
        self._frame = frame
        self._replaced = True
        self._stream_id_override = None
        self._known_flags_override = None

    def serialize(self) -> bytes:
        """
        Return the bytes to forward for this frame.

        Without edits, this is byte-for-byte identical to the data the
        envelope was parsed from. Envelope-level edits to :attr:`stream_id`
        or :attr:`flags` are applied to a copy of the original header, while
        the reserved stream-ID bit, unknown flag bits, and the payload are
        emitted exactly as received. After :meth:`replace_frame`, the
        replacement frame's own serialization is returned.
        """
        if self._replaced:
            return self._frame.serialize()
        return self._effective_header() + self._raw_body

    def _effective_header(self) -> bytes:
        """
        The original wire header with any envelope-level edits applied.
        """
        flag_byte = self._raw_header[4]
        if self._known_flags_override is not None:
            known_mask = 0
            known_bits = 0
            for flag in self._frame.defined_flags:
                known_mask |= flag.bit
                if flag.name in self._known_flags_override:
                    known_bits |= flag.bit
            flag_byte = (flag_byte & ~known_mask) | known_bits

        stream_word = _STRUCT_L.unpack(self._raw_header[5:HEADER_LEN])[0]
        if self._stream_id_override is not None:
            stream_word = (stream_word & _RESERVED_BIT) | self._stream_id_override

        return self._raw_header[:4] + bytes([flag_byte]) + _STRUCT_L.pack(stream_word)
