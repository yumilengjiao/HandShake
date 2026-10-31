import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/core/core_repository.dart';
import 'package:handshake_stage/core/core_snapshot.dart';

import 'test_support.dart';

void main() {
  test('Python golden fixture preserves nanoseconds above 2^53', () {
    expect(snapshot().timestampNs, BigInt.parse('9007199254740999'));
    expect(snapshot().landmarks, isEmpty);
    expect(
      () => snapshot().landmarks.add(const StagePoint(0, 0, 0)),
      throwsUnsupportedError,
    );
  });

  for (final change in <Map<String, Object?>>[
    {'version': 2},
    {'version': true},
    {'timestamp_ns': 9007199254740999},
    {'epoch': '01'},
    {'control_available': true},
    {'source': 'camera'},
    {'model_status': 'loaded'},
    {'extra': 'field'},
    {'reason': 'unknown'},
    {
      'landmarks': [
        {'x': 0.5, 'y': 0.2, 'z': 0.0},
      ],
    },
    {'state': 'preview', 'frame_timestamp_ns': null},
    {'state': 'stopped', 'frame_timestamp_ns': '1'},
  ]) {
    test('invalid snapshot is rejected: $change', () {
      expect(() => snapshot(change), throwsFormatException);
    });
  }

  test('bounded decoder accepts fragmented UTF-8 and multiple lines', () async {
    final bytes = utf8.encode('手势\n{}\n');
    final lines = await boundedLines(
      Stream.fromIterable([
        bytes.sublist(0, 1),
        bytes.sublist(1, 4),
        bytes.sublist(4),
      ]),
    ).toList();
    expect(lines, ['手势', '{}']);
  });

  test(
    'unterminated and oversized messages fail before unbounded buffering',
    () async {
      await expectLater(
        boundedLines(Stream.value([123])).toList(),
        throwsFormatException,
      );
      await expectLater(
        boundedLines(Stream.value(List.filled(30, 32)), limit: 20).toList(),
        throwsFormatException,
      );
    },
  );
}
