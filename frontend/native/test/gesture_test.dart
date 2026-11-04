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

  testWidgets(
    'first painted owner offer enables O without another packet or ACK loop',
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
      Future<void> shifted(LogicalKeyboardKey key) async {
        await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
        await tester.sendKeyEvent(key);
        await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
      }

      await shifted(LogicalKeyboardKey.keyC);
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
      await shifted(LogicalKeyboardKey.keyE);
      expect(repo.commands.last, 'enable');
      repo.emit({
        'enabled': true,
        'control_released': false,
        'model_epoch': '1',
        'eligible_owner': true,
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
      await shifted(LogicalKeyboardKey.keyO);
      expect(repo.commands.last, 'choose_owner');
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    },
  );

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
