import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/core/core_snapshot.dart';
import 'package:handshake_stage/core/preview_packets.dart';
import 'package:handshake_stage/stage/stage_screen.dart';
import 'package:handshake_stage/stage/stage_view_model.dart';

import 'test_support.dart';

Uint8List fixture() =>
    File('test/fixtures/model_preview_v3.bin').readAsBytesSync();

CoreSnapshot classified({
  Map<String, Object?> fields = const {},
  Map<String, Object?> classification = const {},
  bool image = true,
}) {
  final bytes = fixture();
  final length = ByteData.sublistView(bytes).getUint32(4);
  final data = jsonDecode(
    utf8.decode(bytes.sublist(12, 12 + length)),
  ) as Map<String, dynamic>;
  (data['classification'] as Map<String, dynamic>).addAll(classification);
  data.addAll(fields);
  return CoreSnapshot.parse(
    jsonEncode(data),
    jpeg: image ? Uint8List.sublistView(bytes, 12 + length) : null,
  );
}

void main() {
  test(
    'v3 model notice is bound to frame, raw scores and synthetic provenance',
    () async {
      final bytes = fixture();
      final result = (await decodePreviewPackets(
        Stream.fromIterable([bytes.sublist(0, 7), bytes.sublist(7)]),
      ).toList()).single;
      expect(result.version, 3);
      expect(result.classification!.syntheticTraining, isTrue);
      expect(result.classification!.topToken, 'GRB');
      expect(result.classification!.topLogit, 1.4);
      expect(result.classification!.sourceNs, BigInt.parse('9007199254740999'));
      expect(
        () => result.classification!.logits[0] = 1,
        throwsUnsupportedError,
      );
    },
  );

  for (final change in <Map<String, Object?>>[
    {'calibrated': true},
    {'calibrated': 0},
    {'context_epoch': '2'},
    {'source_frame_id': '18'},
    {'source_timestamp_ns': '9007199254740998'},
    {'window_length': 31},
    {'state': 'rejected'},
    {
      'logits': [0.0],
    },
    {'vocabulary_version': 'unknown'},
    {'inference_ns': null},
  ]) {
    test('invalid classification rejected: $change', () {
      expect(() => classified(classification: change), throwsFormatException);
    });
  }

  test(
    'cannot inject model fields into v2 or switch versions in one stream',
    () async {
      expect(() => classified(fields: {'version': 2}), throwsFormatException);
      expect(
        () => classified(
          fields: {'hands': [], 'perception_status': 'processed_no_hand'},
        ),
        throwsFormatException,
      );
      final v2 = File('test/fixtures/preview_v2.bin').readAsBytesSync();
      await expectLater(
        decodePreviewPackets(Stream.value([...v2, ...fixture()])).toList(),
        throwsFormatException,
      );
    },
  );

  testWidgets(
    'raw class/logit, synthetic-training warning, stop clears prediction',
    (tester) async {
      tester.view.physicalSize = const Size(1280, 1000);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final core = FakeCore();
      final model = StageViewModel(() => core);
      await model.connect();
      core.controller.add(classified());
      await tester.pumpWidget(MaterialApp(home: StageScreen(model: model)));
      expect(find.textContaining('合成训练 · 仅管线诊断'), findsOneWidget);
      expect(find.textContaining('来源类 GRB / logit 1.400'), findsOneWidget);
      expect(find.textContaining('未校准，不是操作意图'), findsOneWidget);
      expect(core.commands, isEmpty);
      model.stop();
      core.controller.add(
        classified(
          fields: {
            'state': 'stopped',
            'reason': 'user_stop',
            'sequence': '2',
            'epoch': '2',
            'last_command_id': '1',
            'frame_width': 0,
            'frame_height': 0,
            'frame_timestamp_ns': null,
            'hands': [],
            'model_status': 'unavailable',
            'classification': null,
          },
          image: false,
        ),
      );
      await tester.pump();
      expect(model.stopPending, isFalse);
      expect(find.textContaining('来源类 GRB'), findsNothing);
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    },
  );
}
