import 'dart:convert';
import 'dart:typed_data';

import 'core_snapshot.dart';

/// Framed HSP2/HSP3 / uint32 metadata length / uint32 JPEG length / JSON / JPEG.
/// Every allocation is bounded before reading a body; no resynchronization on errors.
Stream<CoreSnapshot> decodePreviewPackets(Stream<List<int>> input) async* {
  var builder = BytesBuilder(copy: false);
  var required = 12;
  var metadataSize = 0;
  var imageSize = 0;
  var prefix = true;
  int? expectedVersion;
  await for (final chunk in input) {
    var offset = 0;
    while (offset < chunk.length) {
      final count = (required - builder.length).clamp(0, chunk.length - offset);
      builder.add(chunk.sublist(offset, offset + count));
      offset += count;
      if (builder.length != required) continue;
      final bytes = builder.takeBytes();
      if (prefix) {
        final magic = ascii.decode(bytes.sublist(0, 4), allowInvalid: true);
        if (magic != 'HSP2' && magic != 'HSP3') {
          throw const FormatException('Unsupported binary protocol');
        }
        final version = magic == 'HSP3' ? 3 : 2;
        if (expectedVersion != null && version != expectedVersion) {
          throw const FormatException(
            'Protocol version changed within session',
          );
        }
        expectedVersion = version;
        final header = ByteData.sublistView(bytes);
        metadataSize = header.getUint32(4);
        imageSize = header.getUint32(8);
        if (metadataSize < 1 || metadataSize > 16384 || imageSize > 131072) {
          throw const FormatException('Preview exceeds framing limits');
        }
        required = metadataSize + imageSize;
        prefix = false;
      } else {
        final snapshot = CoreSnapshot.parse(
          utf8.decode(bytes.sublist(0, metadataSize)),
          jpeg: imageSize == 0
              ? null
              : Uint8List.sublistView(bytes, metadataSize),
        );
        if (snapshot.version != expectedVersion) {
          throw const FormatException('Header/payload version mismatch');
        }
        yield snapshot;
        prefix = true;
        required = 12;
      }
    }
  }
  if (!prefix || builder.length != 0) {
    throw const FormatException('Truncated preview packet');
  }
}
