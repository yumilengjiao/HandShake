import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/presentation/presentation_repository.dart';
import 'package:handshake_stage/presentation/presentation_screen.dart';
import 'package:handshake_stage/presentation/presentation_snapshot.dart';
import 'package:handshake_stage/presentation/presentation_view_model.dart';

PresentationSnapshot snapshot([Map<String, Object?> changes = const {}]) {
  final data = jsonDecode(
    File('test/fixtures/presentation.json').readAsStringSync(),
  ) as Map<String, dynamic>;
  data.addAll(changes);
  return PresentationSnapshot.parse(jsonEncode(data));
}

class FakePresenter implements PresentationRepository {
  final controller = StreamController<PresentationSnapshot>.broadcast(
    sync: true,
  );
  final commands = <String>[];
  @override
  Stream<PresentationSnapshot> get snapshots => controller.stream;
  @override
  Future<void> connect() async {}
  @override
  void send(String kind, PresentationSnapshot context, BigInt id) {
    commands.add(kind);
  }

  @override
  Future<void> close() async {
    await Future<void>.value();
    await controller.close();
  }

  void emit([Map<String, Object?> changes = const {}]) {
    controller.add(snapshot(changes));
  }
}

void main() {
  test(
    'wire is exact, bounded and cannot claim external/gesture authority',
    () {
      expect(snapshot().timestampNs, BigInt.parse('9007199254740999'));
      expect(() => snapshot().lines.add('change'), throwsUnsupportedError);
      for (final change in <Map<String, Object?>>[
        {'version': true},
        {'version': 2},
        {'external_control_available': true},
        {'gesture_control_available': true},
        {'scope': 'external'},
        {'page': 99},
        {'page': 0},
        {'rendered_revision': '9'},
        {'epoch': 1},
        {'extra': 'field'},
      ]) {
        expect(() => snapshot(change), throwsFormatException);
      }
    },
  );

  testWidgets(
    'render ACK is not play, and stop waits for actual core quiescence',
    (tester) async {
      final repo = FakePresenter();
      final model = PresentationViewModel(() => repo);
      model.active = true;
      await model.connect();
      repo.emit();
      expect(model.canPlay, isFalse);
      model.surfacePainted(model.latest!);
      expect(repo.commands, ['render_ack']);
      repo.emit({
        'sequence': '2',
        'last_command_id': '1',
        'rendered_revision': '0',
      });
      expect(model.canPlay, isTrue);
      model.play();
      expect(repo.commands.last, 'play');
      repo.emit({
        'sequence': '3',
        'last_command_id': '2',
        'state': 'playing',
        'epoch': '1',
        'reason': 'manual_playing',
      });
      expect(model.canNavigate, isFalse);
      model.surfacePainted(model.latest!);
      repo.emit({
        'sequence': '4',
        'last_command_id': '3',
        'state': 'playing',
        'epoch': '1',
        'rendered_revision': '0',
      });
      expect(model.canNavigate, isTrue);
      model.next();
      model.next(); // no queueing duplicate clicks while the first is pending
      expect(repo.commands.where((v) => v == 'next').length, 1);
      model.stop();
      repo.emit({
        'sequence': '5',
        'last_command_id': '5',
        'epoch': '2',
        'release_confirmed': false,
        'reason': 'user_stop',
      });
      expect(model.stopPending, isTrue);
      repo.emit({
        'sequence': '6',
        'last_command_id': '5',
        'epoch': '2',
        'reason': 'user_stop',
      });
      expect(model.stopPending, isFalse);
      model.dispose();
    },
  );

  testWidgets('health timeout clears page and reconnect never auto-plays', (
    tester,
  ) async {
    final repos = [FakePresenter(), FakePresenter()];
    var i = 0;
    final model = PresentationViewModel(() => repos[i++]);
    model.active = true;
    await model.connect();
    repos[0].emit();
    await tester.pump(const Duration(seconds: 3));
    expect(model.link, PresentationLink.disconnected);
    expect(model.latest, isNull);
    expect(model.message, contains('停止未确认'));
    await model.connect();
    repos[1].emit({'session_id': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'});
    expect(repos[1].commands, isEmpty);
    expect(model.canNavigate, isFalse);
    model.dispose();
  });

  for (final (width, scale) in [(1280.0, 1.0), (360.0, 2.0)]) {
    testWidgets('native slide controls at $width/$scale keep stop reachable', (
      tester,
    ) async {
      tester.view.physicalSize = Size(width, 900);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final repo = FakePresenter();
      final model = PresentationViewModel(() => repo);
      model.active = true;
      await model.connect();
      repo.emit();
      await tester.pumpWidget(
        MaterialApp(
          home: MediaQuery(
            data: MediaQueryData(
              size: Size(width, 900),
              textScaler: TextScaler.linear(scale),
            ),
            child: PresentationScreen(model: model),
          ),
        ),
      );
      expect(tester.takeException(), isNull);
      expect(
        find.byKey(const Key('presentation-stop')).hitTestable(),
        findsOneWidget,
      );
      expect(repo.commands, ['render_ack']);
      await tester.sendKeyEvent(LogicalKeyboardKey.escape);
      expect(repo.commands.last, 'stop');
      await tester.pumpWidget(const SizedBox.shrink());
      model.dispose();
    });
  }
}
