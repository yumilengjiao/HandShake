import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/gesture/gesture_repository.dart';
import 'package:handshake_stage/gesture/gesture_screen.dart';
import 'package:handshake_stage/gesture/gesture_snapshot.dart';
import 'package:handshake_stage/gesture/gesture_view_model.dart';

Map<String, dynamic> fixture() =>
    jsonDecode(File('test/fixtures/native_gesture.json').readAsStringSync())
        as Map<String, dynamic>;
GestureSnapshot snapshot([Map<String, Object?> changes = const {}]) =>
    GestureSnapshot.parse(jsonEncode(fixture()..addAll(changes)));

Map<String, Object?> progress([Map<String, Object?> changes = const {}]) => {
  'window_length': 32,
  'warmup_valid': 1,
  'source_gap_ms': 133.3,
  'frame_age_ms': 120.0,
  'max_gap_ms': 100.0,
  'reset_reason': 'sampling_gap',
  ...changes,
};

class FakeGesture implements GestureRepository {
  final output = StreamController<GestureSnapshot>.broadcast(sync: true);
  final commands = <String>[];
  final contexts = <GestureSnapshot>[];
  final data = fixture();
  int sequence = 0;
  BigInt id = BigInt.zero;
  @override
  Stream<GestureSnapshot> get snapshots => output.stream;
  @override
  Future<void> connect() async {}
  @override
  void send(String kind, GestureSnapshot context, BigInt id) {
    commands.add(kind);
    contexts.add(context);
    this.id = id;
  }

  @override
  Future<void> close() async {
    await output.close();
  }

  void emit([Map<String, Object?> changes = const {}]) {
    data.addAll(changes);
    data['sequence'] = '${++sequence}';
    data['last_command_id'] = id.toString();
    output.add(GestureSnapshot.parse(jsonEncode(data)));
  }
}

void main() {
  test(
    'optional warmup diagnostics are bounded, immutable and do not authorize',
    () {
      expect(snapshot().diagnostics, isNull);
      final value = snapshot({'diagnostics': progress()});
      expect(value.warmupValid, 1);
      expect(value.warmupResetText, '处理帧间隔超过上限');
      expect(value.flag('eligible_owner'), isFalse);
      expect(
        () => value.diagnostics!['warmup_valid'] = 32,
        throwsUnsupportedError,
      );
      for (final changes in <Map<String, Object?>>[
        {'warmup_valid': true},
        {'warmup_valid': 33},
        {'window_length': 32.0},
        {'source_gap_ms': -1},
        {'frame_age_ms': '120'},
        {'max_gap_ms': 0},
        {'reset_reason': ''},
        {'capture_hz': -1},
        {'capture_hz': 1001},
        {'perception_ms': '18'},
        {'model_ms': true},
        {'model_ms': 60001},
        {'extra': 1},
      ]) {
        expect(
          () => snapshot({'diagnostics': progress(changes)}),
          throwsFormatException,
        );
      }
      expect(
        () => snapshot({
          'diagnostics': {'warmup_valid': 1},
        }),
        throwsFormatException,
      );
    },
  );

  testWidgets(
    'warming shows real consecutive progress and reset cause, O stays disabled',
    (tester) async {
      final repo = FakeGesture();
      final model = GestureViewModel(() => repo)..active = true;
      await model.connect();
      repo.emit({
        'camera_state': 'running',
        'camera_released': false,
        'enabled': true,
        'control_released': false,
        'model_epoch': '1',
        'policy_reason': 'warming_up',
        'diagnostics': progress(),
      });
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(body: GestureScreen(model: model)),
        ),
      );
      await tester.pump();
      expect(model.canChoose, isFalse);
      expect(model.message, contains('连续有效 1/32'));
      expect(model.message, contains('处理帧间隔超过上限'));
      expect(find.text('连续有效帧：1 / 32 · 历史：32 / 32'), findsOneWidget);
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    },
  );

  testWidgets(
    'recovered sampling is not reported as a continuing historical gap',
    (tester) async {
      final repo = FakeGesture();
      final model = GestureViewModel(() => repo)..active = true;
      await model.connect();
      repo.emit({
        'camera_state': 'running',
        'camera_released': false,
        'enabled': true,
        'control_released': false,
        'model_epoch': '1',
        'policy_reason': 'warming_up',
        'diagnostics': progress({
          'warmup_valid': 8,
          'source_gap_ms': 33.3,
          'capture_hz': 30.0,
          'perception_ms': 21.5,
          'model_ms': 2.4,
        }),
      });
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(body: GestureScreen(model: model)),
        ),
      );
      await tester.pump();
      expect(model.latest!.samplingOverBudget, isFalse);
      expect(model.message, contains('连续有效 8/32'));
      expect(model.message, contains('当前帧间隔 33.3 ms 正常'));
      expect(model.message, isNot(contains('超过上限')));
      expect(find.text('上次重置：处理帧间隔超过上限（历史记录）'), findsOneWidget);
      expect(
        find.text('相机采集：30.0 FPS · 手部感知：21.5 ms · 模型/策略：2.4 ms'),
        findsOneWidget,
      );
      expect(model.canChoose, isFalse);
      repo.emit({
        'diagnostics': progress({'source_gap_ms': 100.0}),
      });
      expect(model.latest!.samplingOverBudget, isFalse);
      repo.emit({
        'diagnostics': progress({'source_gap_ms': 100.1}),
      });
      expect(model.latest!.samplingOverBudget, isTrue);
      expect(model.message, contains('当前处理帧间隔超过上限'));
      expect(model.message, contains('100.1 / 100 ms'));
      expect(model.canChoose, isFalse);
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    },
  );

  test('native protocol is exact, immutable and rejects forged control', () {
    expect(snapshot().timestampNs, BigInt.parse('9007199254740999'));
    expect(() => snapshot().lines.add('mutate'), throwsUnsupportedError);
    for (final changes in <Map<String, Object?>>[
      {'version': true},
      {'version': 1.0},
      {'scope': 'external'},
      {'external_control_available': true},
      {'experimental': false},
      {'extra': 1},
      {'enabled': 1},
      {'sequence': 1},
      {'page': 0},
      {'page': 5},
      {'page_count': 33},
      {'actions_enabled': true},
      {'camera_state': 'running'},
      {'jpeg': 'not_base64!'},
      {'model_epoch': '-1'},
    ]) {
      expect(
        () => snapshot(changes),
        throwsFormatException,
        reason: changes.toString(),
      );
    }
  });

  testWidgets(
    'open enable owner are separate; painted page is not authorization',
    (tester) async {
      final repo = FakeGesture();
      final model = GestureViewModel(() => repo)..active = true;
      await model.connect();
      repo.emit();
      expect(repo.commands, isEmpty);
      expect(model.canOpen, isTrue);
      model.surfacePainted(model.latest!);
      expect(repo.commands, isEmpty);
      model.openCamera();
      model.openCamera();
      expect(repo.commands, ['open_camera']);
      repo.emit({
        'camera_state': 'running',
        'camera_epoch': '1',
        'camera_released': false,
        'model_epoch': '0',
      });
      expect(model.canEnable, isFalse);
      model.surfacePainted(model.latest!);
      expect(repo.commands.last, 'surface_ack');
      expect(model.canEnable, isTrue);
      model.enable();
      expect(repo.commands.last, 'enable');
      repo.emit({
        'enabled': true,
        'control_released': false,
        'model_epoch': '1',
        'eligible_owner': true,
      });
      expect(
        model.canChoose,
        isFalse,
      ); // old painted epoch cannot select a new owner
      model.surfacePainted(model.latest!);
      expect(model.canChoose, isTrue);
      model.chooseOwner();
      expect(repo.commands.last, 'choose_owner');
      repo.emit({'owner_pending': true});
      expect(model.canChoose, isFalse);
      model.stop();
      repo.emit({'owner_pending': false});
      expect(model.pending, isTrue); // command ACK alone is not a release ACK
      repo.emit({
        'enabled': false,
        'eligible_owner': false,
        'control_released': true,
      });
      expect(model.pending, isFalse);
      expect(repo.commands.any((v) => v == 'next' || v == 'previous'), isFalse);
      model.dispose();
    },
  );

  testWidgets('O stays disabled with an explanatory neutral-pose reason', (
    tester,
  ) async {
    final repo = FakeGesture();
    final model = GestureViewModel(() => repo)..active = true;
    await model.connect();
    repo.emit({
      'camera_state': 'running',
      'camera_released': false,
      'enabled': true,
      'control_released': false,
      'model_epoch': '1',
      'policy_reason': 'owner_requires_neutral_pose',
    });
    model.surfacePainted(model.latest!);
    expect(model.canChoose, isFalse);
    expect(model.message, contains('不要完全张掌或握拳'));
    final count = repo.commands.length;
    model.chooseOwner();
    expect(repo.commands.length, count);
    model.dispose();
  });

  for (final (reason, topClass, expected) in [
    ('owner_requires_static_neutral', 'C2F', '等待 NOG / P1F / P2F'),
    ('owner_requires_confident_neutral', 'P2F', 'P2F 置信度不足'),
    ('owner_request_rejected_or_expired', 'P1F', '确认未通过或已过期'),
  ]) {
    testWidgets(
      'local neutral reason stays actionable after rejected O: $reason',
      (tester) async {
        final repo = FakeGesture();
        final model = GestureViewModel(() => repo)..active = true;
        await model.connect();
        repo.emit({
          'camera_state': 'running',
          'camera_released': false,
          'enabled': true,
          'control_released': false,
          'model_epoch': '1',
          'reason': 'owner_rejected',
          'policy_reason': reason,
          'top_class': topClass,
        });
        model.surfacePainted(model.latest!);
        expect(model.canChoose, isFalse);
        expect(model.message, contains(expected));
        final count = repo.commands.length;
        model.chooseOwner();
        expect(repo.commands.length, count);
        repo.emit({
          'eligible_owner': true,
          'policy_reason': 'waiting_explicit_owner',
          'top_class': 'P2F',
        });
        model.surfacePainted(model.latest!);
        expect(model.canChoose, isTrue);
        expect(model.message, contains('当前中立手已就绪'));
        expect(repo.commands, isNot(contains('choose_owner')));
        model.chooseOwner();
        expect(repo.commands.last, 'choose_owner');
        model.dispose();
      },
    );
  }

  testWidgets('bound owner shows actual rewarm and release instead of ready', (
    tester,
  ) async {
    final repo = FakeGesture();
    final model = GestureViewModel(() => repo)..active = true;
    await model.connect();
    repo.emit({
      'camera_state': 'running',
      'camera_released': false,
      'enabled': true,
      'control_released': false,
      'model_epoch': '2',
      'owner_bound': true,
      'warming': true,
      'diagnostics': progress(),
    });
    expect(model.message, contains('连续有效 1/32'));
    expect(model.message, contains('处理帧间隔超过上限'));
    repo.emit({'warming': false, 'model_state': 'WAIT_RELEASE'});
    expect(model.message, contains('等待释放期完成'));
    expect(model.canChoose, isFalse);
    model.dispose();
  });

  for (final (reason, expected) in [
    ('not_processed', '不是手型问题'),
    ('perception_not_processed:model_not_loaded', '手部感知模型未加载'),
    ('perception_not_processed:same_mediapipe_millisecond', '帧时间戳'),
  ]) {
    testWidgets(
      'unprocessed diagnostics identify the pipeline failure: $reason',
      (tester) async {
        final repo = FakeGesture();
        final model = GestureViewModel(() => repo)..active = true;
        await model.connect();
        repo.emit({
          'camera_state': 'running',
          'camera_released': false,
          'enabled': true,
          'control_released': false,
          'model_epoch': '1',
          'policy_reason': reason,
        });
        model.surfacePainted(model.latest!);
        expect(model.canChoose, isFalse);
        expect(model.message, contains(expected));
        model.dispose();
      },
    );
  }

  testWidgets(
    'old surfaces do not ACK new page and mode change waits for camera release',
    (tester) async {
      final repo = FakeGesture();
      final model = GestureViewModel(() => repo)..active = true;
      await model.connect();
      repo.emit({
        'camera_state': 'running',
        'camera_released': false,
        'model_epoch': '1',
      });
      final old = model.latest!;
      repo.emit({'page': 2, 'revision': '1'});
      model.surfacePainted(old);
      expect(repo.commands, isEmpty);
      model.surfacePainted(model.latest!);
      expect(repo.commands, ['surface_ack']);
      final waiting = model.closeCameraAndWait();
      repo.emit({'camera_state': 'stopping'});
      expect(model.pending, isTrue);
      repo.emit({'camera_state': 'stopped', 'camera_released': true});
      expect(await waiting, isTrue);
      model.dispose();
    },
  );

  testWidgets(
    'timeout clears stale video and reconnect never enables or opens camera',
    (tester) async {
      final repos = [FakeGesture(), FakeGesture()];
      var i = 0;
      final model = GestureViewModel(() => repos[i++])..active = true;
      await model.connect();
      repos[0].emit();
      await tester.pump(const Duration(seconds: 3));
      expect(model.link, GestureLink.disconnected);
      expect(model.latest, isNull);
      await model.connect();
      repos[1].emit();
      expect(repos[1].commands, isEmpty);
      expect(model.canOpen, isTrue);
      model.dispose();
    },
  );

  for (final useShift in [false, true]) {
    testWidgets(
      'first painted P2F offer enables O without ACK loop (shift=$useShift)',
      (tester) async {
        final repo = FakeGesture();
        final model = GestureViewModel(() => repo)..active = true;
        await model.connect();
        repo.emit();
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(body: GestureScreen(model: model)),
          ),
        );
        Future<void> letter(LogicalKeyboardKey key) async {
          if (useShift) {
            await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
          }
          await tester.sendKeyDownEvent(key);
          await tester.sendKeyRepeatEvent(key);
          await tester.sendKeyUpEvent(key);
          if (useShift) {
            await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
          }
        }

        await letter(LogicalKeyboardKey.keyC);
        expect(repo.commands.last, 'open_camera');
        repo.emit({
          'camera_state': 'running',
          'camera_epoch': '1',
          'camera_released': false,
          'model_epoch': '0',
        });
        await tester.pump();
        await tester.pump();
        expect(model.canEnable, isTrue);
        await letter(LogicalKeyboardKey.keyE);
        expect(repo.commands.last, 'enable');
        repo.emit({
          'enabled': true,
          'control_released': false,
          'model_epoch': '1',
          'eligible_owner': true,
          'top_class': 'P2F',
        });
        await tester.pump();
        await tester.pump();
        expect(
          tester
              .widget<OutlinedButton>(find.byKey(const Key('gesture-owner')))
              .onPressed,
          isNotNull,
        );
        final ackCount = repo.commands.where((v) => v == 'surface_ack').length;
        for (var i = 0; i < 5; i++) {
          await tester.pump();
        }
        expect(repo.commands.where((v) => v == 'surface_ack').length, ackCount);
        expect(repo.commands, isNot(contains('choose_owner')));
        await letter(LogicalKeyboardKey.keyO);
        expect(repo.commands.last, 'choose_owner');
        expect(repo.commands.where((v) => v == 'choose_owner').length, 1);
        await tester.pumpWidget(const SizedBox.shrink());
        model.dispose();
      },
    );
  }

  testWidgets(
    'empty telemetry keeps decoder and image until bounded stall, close clears it',
    (tester) async {
      final wire = File('test/fixtures/preview_v2.bin').readAsBytesSync();
      final size = ByteData.sublistView(wire).getUint32(4);
      final jpeg = base64Encode(wire.sublist(12 + size));
      final repo = FakeGesture();
      final model = GestureViewModel(() => repo)..active = true;
      await model.connect();
      repo.emit({
        'camera_state': 'running',
        'camera_epoch': '1',
        'camera_released': false,
        'jpeg': jpeg,
      });
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(body: GestureScreen(model: model)),
        ),
      );
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 100)),
      );
      await tester.pump();
      final image = tester.widget<RawImage>(find.byType(RawImage)).image;
      expect(image, isNotNull);
      repo.emit(); // Same JPEG with a new telemetry sequence must not decode again.
      await tester.pump();
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 100)),
      );
      await tester.pump();
      expect(tester.widget<RawImage>(find.byType(RawImage)).image, same(image));
      repo.emit({'jpeg': ''});
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 200));
      expect(tester.widget<RawImage>(find.byType(RawImage)).image, same(image));
      repo.emit({'jpeg': jpeg});
      await tester.pump();
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 100)),
      );
      await tester.pump();
      expect(tester.widget<RawImage>(find.byType(RawImage)).image, isNotNull);
      repo.emit({'jpeg': ''});
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 751));
      expect(find.byType(RawImage), findsNothing);
      expect(find.text('等待新鲜相机画面'), findsOneWidget);
      repo.emit({'camera_state': 'stopping', 'camera_epoch': '2'});
      await tester.pump();
      expect(find.text('相机未打开'), findsOneWidget);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    },
  );

  for (final (width, scale) in [(1280.0, 1.0), (360.0, 2.0)]) {
    testWidgets('combined preview at $width/$scale keeps stop reachable', (
      tester,
    ) async {
      tester.view.physicalSize = Size(width, 900);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final repo = FakeGesture();
      final model = GestureViewModel(() => repo)..active = true;
      await model.connect();
      repo.emit();
      await tester.pumpWidget(
        MaterialApp(
          home: MediaQuery(
            data: MediaQueryData(
              size: Size(width, 900),
              textScaler: TextScaler.linear(scale),
            ),
            child: Scaffold(body: GestureScreen(model: model)),
          ),
        ),
      );
      expect(tester.takeException(), isNull);
      expect(
        find.byKey(const Key('gesture-stop')).hitTestable(),
        findsOneWidget,
      );
      expect(
        find.byKey(const Key('gesture-close')).hitTestable(),
        findsOneWidget,
      );
      expect(repo.commands, isEmpty);
      await tester.sendKeyEvent(LogicalKeyboardKey.space);
      expect(repo.commands.last, 'stop');
      await tester.sendKeyEvent(LogicalKeyboardKey.escape);
      expect(repo.commands.last, 'close_camera');
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    });
  }
}
