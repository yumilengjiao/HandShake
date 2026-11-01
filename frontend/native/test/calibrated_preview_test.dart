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
    File('test/fixtures/calibrated_preview_v4.bin').readAsBytesSync();

CoreSnapshot calibrated({
  Map<String, Object?> probability = const {},
  Map<String, Object?> fields = const {},
  bool image = true,
}) {
  final bytes = fixture();
  final size = ByteData.sublistView(bytes).getUint32(4);
  final data = jsonDecode(
    utf8.decode(bytes.sublist(12, 12 + size)),
  ) as Map<String, dynamic>;
  (data['probability'] as Map<String, dynamic>).addAll(probability);
  data.addAll(fields);
  return CoreSnapshot.parse(
    jsonEncode(data),
    jpeg: image ? Uint8List.sublistView(bytes, 12 + size) : null,
  );
}

void main() {
  test('v4 probability, raw score, big timestamps, immutable lists', () async {
    final value = (await decodePreviewPackets(
      Stream.value(fixture()),
    ).toList()).single;
    expect(value.version, 4);
    expect(value.probability!.rejectionStatus, 'rejected');
    expect(value.classification!.sourceNs, BigInt.parse('9007199254740999'));
    expect(value.classification!.topToken, 'GRB');
    expect(
      () => value.probability!.probabilities[0] = 0,
      throwsUnsupportedError,
    );
  });
  for (final changes in <Map<String, Object?>>[
    {'actions_enabled': true},
    {'actions_enabled': 0},
    {'calibrated': 1},
    {'status': 'frozen'},
    {'model_version': 'wrong'},
    {'synthetic': false},
    {'temperature': 0},
    {
      'probabilities': [1.0],
    },
    {'probabilities': List.filled(15, .2)},
    {'rejection_status': 'accepted'},
    {'threshold': null},
  ]) {
    test('reject forged calibrated telemetry $changes', () {
      expect(() => calibrated(probability: changes), throwsFormatException);
    });
  }
  test(
    'v3 cannot accept v4 fields; stopped snapshots cannot carry probabilities',
    () async {
      expect(() => calibrated(fields: {'version': 3}), throwsFormatException);
      expect(
        () => calibrated(fields: {'classification': null}),
        throwsFormatException,
      );
      final v3 = File('test/fixtures/model_preview_v3.bin').readAsBytesSync();
      await expectLater(
        decodePreviewPackets(Stream.value([...v3, ...fixture()])).toList(),
        throwsFormatException,
      );
    },
  );
  testWidgets(
    'draft probability and rejection visible; no action; stop clears',
    (tester) async {
      tester.view.physicalSize = const Size(1280, 1000);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final core = FakeCore();
      final model = StageViewModel(() => core);
      await model.connect();
      core.controller.add(calibrated());
      await tester.pumpWidget(MaterialApp(home: StageScreen(model: model)));
      expect(find.textContaining('验证集温度校准 draft'), findsOneWidget);
      expect(find.textContaining('分类拒识'), findsOneWidget);
      expect(find.textContaining('不授权动作'), findsOneWidget);
      expect(core.commands, isEmpty);
      model.stop();
      core.controller.add(
        calibrated(
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
            'probability': null,
          },
          image: false,
        ),
      );
      await tester.pump();
      expect(find.textContaining('验证集温度校准 draft'), findsNothing);
      expect(model.canStartCamera, isTrue);
      expect(find.byKey(const Key('start_camera')), findsOneWidget);
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    },
  );
}
