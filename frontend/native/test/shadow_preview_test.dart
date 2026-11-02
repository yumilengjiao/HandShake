import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/core/core_repository.dart';
import 'package:handshake_stage/core/core_snapshot.dart';
import 'package:handshake_stage/core/preview_packets.dart';
import 'package:handshake_stage/stage/preview_surface.dart';
import 'package:handshake_stage/stage/stage_screen.dart';
import 'package:handshake_stage/stage/stage_view_model.dart';

import 'test_support.dart';

Uint8List fixture() =>
    File('test/fixtures/shadow_preview_v5.bin').readAsBytesSync();
CoreSnapshot shadowSnapshot({
  Map<String, Object?> fields = const {},
  Map<String, Object?> shadow = const {},
  Map<String, Object?> classification = const {},
  bool image = true,
}) {
  final bytes = fixture();
  final size = ByteData.sublistView(bytes).getUint32(4);
  final data = jsonDecode(
    utf8.decode(bytes.sublist(12, 12 + size)),
  ) as Map<String, dynamic>;
  (data['shadow'] as Map<String, dynamic>).addAll(shadow);
  (data['classification'] as Map<String, dynamic>).addAll(classification);
  data.addAll(fields);
  return CoreSnapshot.parse(
    jsonEncode(data),
    jpeg: image ? Uint8List.sublistView(bytes, 12 + size) : null,
  );
}

class ShadowCore extends FakeCore {
  final sent = <Map<String, Object?>>[];
  @override
  void send(String kind, CoreSnapshot context, BigInt commandId) {
    super.send(kind, context, commandId);
    sent.add({
      'kind': kind,
      'epoch': context.epoch.toString(),
      'id': commandId.toString(),
      ...shadowCommandFields(kind, context),
    });
  }

  String get lastId => sent.isEmpty ? '0' : sent.last['id']! as String;
}

void main() {
  test('v5 golden has distinct source/model epochs and immutable calibrated values', () async {
    final s = (await decodePreviewPackets(
      Stream.value(fixture()),
    ).toList()).single;
    expect(s.version, 5);
    expect(s.epoch, BigInt.one);
    expect(s.shadow!.epoch, BigInt.from(3));
    expect(s.classification!.epoch, s.shadow!.epoch);
    expect(s.shadow!.eligibleOwner, 'test-hand');
    expect(s.classification!.sourceNs, BigInt.parse('9007199254740999'));
    expect(shadowCommandFields('request_owner', s)['model_epoch'], '3');
    expect(shadowCommandFields('stop', s)['model_epoch'], isNull);
  });
  for (final fields in <Map<String, Object?>>[
    {'actions_enabled': true},
    {'actions_enabled': 0},
    {'actual_page': 2},
    {'actual_page': 1.0},
    {'actual_page': true},
    {'model_revision': '1'},
    {'execution': 'live'},
    {'model_epoch': '1'},
    {'eligible_owner': 'other'},
    {'state': 'DISABLED'},
    {'owner_track_id': 'unbound'},
    {'model_version': 'wrong'},
    {'synthetic': false},
  ]) {
    test(
      'reject invalid shadow contract $fields',
      () => expect(() => shadowSnapshot(shadow: fields), throwsFormatException),
    );
  }
  test('no protocol downgrade or version switching', () async {
    expect(() => shadowSnapshot(fields: {'version': 4}), throwsFormatException);
    final v4 = File('test/fixtures/calibrated_preview_v4.bin')
        .readAsBytesSync();
    await expectLater(
      decodePreviewPackets(Stream.value([...v4, ...fixture()])).toList(),
      throwsFormatException,
    );
  });

  testWidgets(
    'explicit owner command uses captured model epoch; review and stop disable controls',
    (tester) async {
      tester.view.physicalSize = const Size(1280, 1000);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final core = ShadowCore();
      final model = StageViewModel(() => core);
      await model.connect();
      core.controller.add(shadowSnapshot());
      await tester.pumpWidget(MaterialApp(home: StageScreen(model: model)));
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 40)),
      );
      await tester.pump();
      await tester.pump();
      expect(find.textContaining('所有候选均不执行'), findsOneWidget);
      expect(model.canRequestOwner, isTrue);
      expect(find.byKey(const Key('start')), findsNothing);
      model.inspect(0);
      await tester.pump();
      expect(model.canRequestOwner, isFalse);
      model.requestOwner();
      expect(core.commands.where((x) => x == 'request_owner'), isEmpty);
      model.returnToNow();
      await tester.pump();
      await tester.tap(find.byKey(const Key('request_owner')));
      final request = core.sent.lastWhere((v) => v['kind'] == 'request_owner');
      expect(request['epoch'], '1');
      expect(request['model_epoch'], '3');
      expect(request['track_id'], 'test-hand');
      core.controller.add(
        shadowSnapshot(
          fields: {
            'sequence': '2',
            'last_command_id': core.lastId,
            'classification': null,
            'probability': null,
          },
          shadow: {
            'model_sequence': '42',
            'model_epoch': '4',
            'state': 'WAIT_RELEASE',
            'owner_track_id': 'test-hand',
            'eligible_owner': null,
            'model_warming': true,
          },
        ),
      );
      await tester.pump();
      expect(model.canRequestOwner, isFalse);
      expect(find.textContaining('绑定后重新预热'), findsOneWidget);
      model.stop();
      expect(core.sent.last['kind'], 'stop');
      expect(core.sent.last['model_epoch'], isNull);
      core.controller.add(
        shadowSnapshot(
          fields: {
            'sequence': '3',
            'last_command_id': core.lastId,
            'state': 'stopping',
            'reason': 'user_stop',
            'epoch': '2',
            'frame_width': 0,
            'frame_height': 0,
            'frame_timestamp_ns': null,
            'hands': [],
            'model_status': 'unavailable',
            'classification': null,
            'probability': null,
            'shadow': null,
          },
          image: false,
        ),
      );
      await tester.pump();
      expect(model.stopPending, isTrue);
      core.controller.add(
        shadowSnapshot(
          fields: {
            'sequence': '4',
            'last_command_id': core.lastId,
            'state': 'stopped',
            'reason': 'user_stop',
            'epoch': '2',
            'frame_width': 0,
            'frame_height': 0,
            'frame_timestamp_ns': null,
            'hands': [],
            'model_status': 'unavailable',
            'classification': null,
            'probability': null,
            'shadow': null,
          },
          image: false,
        ),
      );
      await tester.pump();
      expect(model.stopPending, isFalse);
      expect(model.history, isEmpty);
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    },
  );

  testWidgets(
    'surface ACK waits for decoded image drawing and is suppressed in review',
    (tester) async {
      final first = shadowSnapshot();
      final second = shadowSnapshot(fields: {'sequence': '2'});
      final decoded = await tester.runAsync(() => decodePreviewImage(first));
      final gate = Completer<ui.Image>();
      final presented = <CoreSnapshot>[];
      Widget surface(CoreSnapshot value, bool review) => MaterialApp(
        home: SizedBox(
          width: 640,
          height: 480,
          child: PreviewSurface(
            snapshot: value,
            history: [value],
            mirror: false,
            reviewing: review,
            decoder: (_) => gate.future,
            onPresented: presented.add,
          ),
        ),
      );
      await tester.pumpWidget(surface(first, false));
      await tester.pump();
      expect(presented, isEmpty);
      gate.complete(decoded!);
      await tester.pump();
      await tester.pump();
      expect(presented, isNotEmpty);
      expect(presented.last.sequence, first.sequence);
      presented.clear();
      await tester.pumpWidget(surface(second, true));
      await tester.pump();
      expect(presented, isEmpty);
      await tester.pumpWidget(const SizedBox.shrink());
    },
  );

  testWidgets(
    'enable needs a fresh acknowledged surface; failed decode cannot acknowledge',
    (tester) async {
      final disabled = shadowSnapshot(
        fields: {'classification': null, 'probability': null},
        shadow: {
          'state': 'DISABLED',
          'eligible_owner': null,
          'surface_ready': false,
        },
      );
      final core = ShadowCore();
      final model = StageViewModel(() => core);
      await model.connect();
      core.controller.add(disabled);
      expect(model.canEnableShadow, isFalse);
      model.enableShadow();
      expect(core.commands, isEmpty);
      final acknowledgements = <CoreSnapshot>[];
      await tester.pumpWidget(
        MaterialApp(
          home: PreviewSurface(
            snapshot: disabled,
            history: [],
            mirror: false,
            reviewing: false,
            decoder: (_) => Future<ui.Image>.error(StateError('decode failed')),
            onPresented: acknowledgements.add,
          ),
        ),
      );
      await tester.pump();
      expect(acknowledgements, isEmpty);
      core.controller.add(
        shadowSnapshot(
          fields: {
            'sequence': '2',
            'classification': null,
            'probability': null,
          },
          shadow: {
            'state': 'DISABLED',
            'eligible_owner': null,
            'surface_ready': true,
            'model_sequence': '42',
          },
        ),
      );
      expect(model.canEnableShadow, isTrue);
      model.enableShadow();
      expect(core.commands.last, 'enable_shadow');
      expect(core.sent.last['model_epoch'], '3');
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    },
  );

  test(
    'owner offer requires a presented and still-fresh observation',
    () async {
      final core = ShadowCore();
      final model = StageViewModel(() => core);
      await model.connect();
      final first = shadowSnapshot();
      core.controller.add(first);
      expect(model.canRequestOwner, isFalse);
      model.surfacePresented(first);
      expect(model.canRequestOwner, isTrue);
      final later = (first.frameTimestampNs! + BigInt.from(200000000))
          .toString();
      core.controller.add(
        shadowSnapshot(
          fields: {
            'sequence': '2',
            'frame_id': '18',
            'frame_timestamp_ns': later,
            'timestamp_ns': later,
            'last_command_id': core.lastId,
          },
          shadow: {'model_sequence': '42'},
          classification: {
            'source_frame_id': '18',
            'source_timestamp_ns': later,
          },
        ),
      );
      expect(model.canRequestOwner, isFalse);
      model.requestOwner();
      expect(core.commands.where((k) => k == 'request_owner'), isEmpty);
      model.dispose();
    },
  );

  test(
    'paint completion may lag telemetry but keeps its captured token',
    () async {
      final core = ShadowCore();
      final model = StageViewModel(() => core);
      await model.connect();
      final first = shadowSnapshot();
      core.controller.add(first);
      core.controller.add(
        shadowSnapshot(
          fields: {'sequence': '2'},
          shadow: {'model_sequence': '42'},
        ),
      );
      model.surfacePresented(first);
      expect(model.canRequestOwner, isTrue);
      expect(core.sent.last['kind'], 'surface_ack');
      expect(
        core.sent.last['model_sequence'],
        first.shadow!.sequence.toString(),
      );
      final count = core.sent.length;
      model.surfacePresented(first);
      expect(core.sent.length, count);
      model.requestOwner();
      expect(
        core.sent.last['model_sequence'],
        first.shadow!.sequence.toString(),
      );
      model.dispose();
    },
  );

  test(
    'old or expired paint completions cannot acknowledge a new context',
    () async {
      final core = ShadowCore();
      final model = StageViewModel(() => core);
      await model.connect();
      final first = shadowSnapshot();
      core.controller.add(first);
      core.controller.add(
        shadowSnapshot(
          fields: {'sequence': '2'},
          shadow: {'model_epoch': '4', 'model_sequence': '42'},
          classification: {'context_epoch': '4'},
        ),
      );
      model.surfacePresented(first);
      expect(core.sent, isEmpty);
      expect(model.canRequestOwner, isFalse);
      final current = model.latest!;
      core.controller.add(
        shadowSnapshot(
          fields: {
            'sequence': '3',
            'timestamp_ns': (first.timestampNs + BigInt.from(500000000))
                .toString(),
          },
          shadow: {'model_epoch': '4', 'model_sequence': '43'},
          classification: {'context_epoch': '4'},
        ),
      );
      model.surfacePresented(current);
      expect(core.sent, isEmpty);
      model.dispose();
    },
  );

  testWidgets(
    'cold connect has a separate deadline; live silence still stops',
    (tester) async {
      final core = ShadowCore();
      final model = StageViewModel(() => core);
      await model.connect();
      await tester.pump(const Duration(seconds: 3));
      expect(model.link, LinkState.connecting);
      core.controller.add(shadowSnapshot());
      expect(model.link, LinkState.connected);
      await tester.pump(const Duration(seconds: 2));
      expect(model.link, LinkState.disconnected);
      model.dispose();
    },
  );

  test('old nested epochs cannot replace the current owner view', () async {
    final core = ShadowCore();
    final model = StageViewModel(() => core);
    await model.connect();
    core.controller.add(shadowSnapshot());
    core.controller.add(
      shadowSnapshot(
        fields: {'sequence': '2'},
        shadow: {'model_epoch': '2', 'model_sequence': '40'},
        classification: {'context_epoch': '2'},
      ),
    );
    expect(model.latest!.sequence, BigInt.one);
    model.dispose();
  });

  for (final width in [360.0, 1280.0]) {
    testWidgets('shadow controls fit $width with doubled text', (tester) async {
      tester.view.physicalSize = Size(width, 900);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final core = ShadowCore();
      final model = StageViewModel(() => core);
      await model.connect();
      core.controller.add(shadowSnapshot());
      await tester.pumpWidget(
        MaterialApp(
          home: MediaQuery(
            data: MediaQueryData(
              size: Size(width, 900),
              textScaler: const TextScaler.linear(2),
            ),
            child: StageScreen(model: model),
          ),
        ),
      );
      expect(find.byKey(const Key('stop')), findsOneWidget);
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    });
  }
}
