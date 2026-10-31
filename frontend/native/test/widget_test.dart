import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/core/core_snapshot.dart';
import 'package:handshake_stage/stage/skeleton_painter.dart';
import 'package:handshake_stage/stage/stage_screen.dart';
import 'package:handshake_stage/stage/stage_view_model.dart';

import 'test_support.dart';

void main() {
  testWidgets('history is bounded, review is read-only, stop waits for ack', (
    tester,
  ) async {
    final core = FakeCore();
    final model = StageViewModel(() => core);
    addTearDown(model.dispose);
    await model.connect();
    core.emit();
    model.start();
    for (var i = 2; i <= 42; i++) {
      core.emit({
        'sequence': '$i',
        'epoch': '1',
        'last_command_id': '1',
        'state': 'preview',
        'reason': 'synthetic_preview',
        'frame_id': '$i',
        'frame_timestamp_ns': '9007199254740999',
      });
    }
    expect(model.history.length, 32);
    model.inspect(0);
    final selected = model.displayed;
    model.start();
    model.toggleMirror();
    expect(core.commands, ['start']);
    core.emit({
      'sequence': '43',
      'epoch': '1',
      'last_command_id': '1',
      'state': 'preview',
      'reason': 'synthetic_preview',
      'frame_id': '43',
      'frame_timestamp_ns': '9007199254740999',
    });
    expect(model.displayed, same(selected));
    model.stop();
    expect(model.stopPending, isTrue);
    expect(model.message, contains('等待核心确认'));
    core.emit({
      'sequence': '44',
      'epoch': '2',
      'last_command_id': '2',
      'reason': 'user_stop',
    });
    expect(model.stopPending, isFalse);
    expect(model.history, isEmpty);
    expect(model.reviewing, isFalse);
    model.dispose();
  });

  testWidgets(
    'stale snapshots cannot refresh health, disconnect clears stage',
    (tester) async {
      final core = FakeCore();
      final model = StageViewModel(() => core);
      addTearDown(model.dispose);
      await model.connect();
      core.emit({'sequence': '5', 'epoch': '2'});
      core.emit({'sequence': '6', 'epoch': '1'});
      expect(model.latest!.sequence, BigInt.from(5));
      await tester.pump(const Duration(seconds: 1));
      core.emit({'sequence': '5', 'epoch': '2'});
      await tester.pump(const Duration(milliseconds: 1100));
      expect(model.link, LinkState.disconnected);
      expect(model.latest, isNull);
      expect(model.message, contains('停止未确认'));
      expect(core.closed, isTrue);
    },
  );

  testWidgets('new session requires explicit reconnect, never auto-starts', (
    tester,
  ) async {
    final cores = [FakeCore(), FakeCore()];
    var index = 0;
    final model = StageViewModel(() => cores[index++]);
    addTearDown(model.dispose);
    await model.connect();
    cores[0].emit();
    cores[0].emit({
      'session_id': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
      'sequence': '2',
    });
    expect(model.link, LinkState.disconnected);
    await tester.pump();
    await model.connect();
    cores[1].emit({'session_id': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'});
    expect(model.link, LinkState.connected);
    expect(cores[1].commands, isEmpty);
    expect(model.canStart, isTrue);
    model.dispose();
  });

  for (final (width, scale) in [(1280.0, 1.0), (360.0, 1.0), (360.0, 2.0)]) {
    testWidgets('stage at width $width and scale $scale keeps stop visible', (
      tester,
    ) async {
      tester.view.reset();
      tester.view.physicalSize = Size(width, 900);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final core = FakeCore();
      final model = StageViewModel(() => core);
      addTearDown(model.dispose);
      await model.connect();
      core.emit();
      await tester.pumpWidget(
        MaterialApp(
          home: MediaQuery(
            data: MediaQueryData(
              size: Size(width, 900),
              textScaler: TextScaler.linear(scale),
            ),
            child: StageScreen(model: model),
          ),
        ),
      );
      expect(tester.takeException(), isNull);
      expect(find.byKey(const Key('stop')).hitTestable(), findsOneWidget);
      await tester.tap(find.byKey(const Key('start')));
      expect(core.commands, ['start']);
      await tester.sendKeyEvent(LogicalKeyboardKey.escape);
      await tester.pump();
      expect(core.commands, ['start', 'stop']);
      expect(find.text('等待停止确认'), findsOneWidget);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    });
  }

  testWidgets('stop ack timeout is not hidden by healthy snapshots', (
    tester,
  ) async {
    final core = FakeCore();
    final model = StageViewModel(() => core);
    addTearDown(model.dispose);
    await model.connect();
    core.emit();
    model.stop();
    for (var i = 2; i <= 4; i++) {
      await tester.pump(const Duration(milliseconds: 500));
      core.emit({'sequence': '$i'});
    }
    expect(model.stopPending, isTrue);
    expect(model.message, contains('等待核心确认'));
    await tester.pump(const Duration(milliseconds: 600));
    expect(model.link, LinkState.disconnected);
    expect(model.message, contains('停止未确认'));
    model.dispose();
  });

  test('letterbox and mirror share a single projection', () {
    const point = StagePoint(.25, .5, 0);
    const size = Size(800, 300);
    expect(projectPoint(point, size, mirror: false), const Offset(300, 150));
    expect(projectPoint(point, size, mirror: true), const Offset(500, 150));
    expect(point.x, .25);
  });
}
