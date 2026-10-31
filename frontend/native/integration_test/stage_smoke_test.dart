import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/core/core_repository.dart';
import 'package:handshake_stage/stage/skeleton_painter.dart';
import 'package:handshake_stage/stage/stage_screen.dart';
import 'package:handshake_stage/stage/stage_view_model.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets(
    'native window and real Python pipe start/review/stop/reconnect',
    (tester) async {
      await tester.binding.setSurfaceSize(const Size(1280, 900));
      final python = Platform.environment['HANDSHAKE_PYTHON'];
      final root = Platform.environment['HANDSHAKE_ROOT'];
      expect(
        python,
        isNotNull,
        reason: 'Set HANDSHAKE_PYTHON to the project interpreter',
      );
      expect(root, isNotNull, reason: 'Set HANDSHAKE_ROOT to the repository');
      late ProcessCoreRepository repository;
      final model = StageViewModel(() {
        repository = ProcessCoreRepository(
          python: python!,
          workingDirectory: root,
          perceptionModel: Platform.environment['HANDSHAKE_PERCEPTION_MODEL'],
        );
        return repository;
      });
      final boundaryKey = GlobalKey();
      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData(
            brightness: Brightness.dark,
            useMaterial3: true,
            colorSchemeSeed: const Color(0xff91dcc7),
          ),
          home: RepaintBoundary(
            key: boundaryKey,
            child: StageScreen(model: model),
          ),
        ),
      );

      Future<void> until(bool Function() condition) async {
        final deadline = DateTime.now().add(const Duration(seconds: 8));
        while (!condition() && DateTime.now().isBefore(deadline)) {
          await tester.pump(const Duration(milliseconds: 20));
        }
        expect(condition(), isTrue, reason: model.message);
      }

      try {
        await model.connect();
        await until(() => model.canStart);
        final firstSession = model.latest!.sessionId;
        await tester.tap(find.byKey(const Key('start')));
        await until(
          () =>
              model.history.length >= 20 && model.latest!.landmarks.isNotEmpty,
        );
        expect(tester.takeException(), isNull);
        model.inspect(model.history.length - 1);
        final inspected = model.displayed;
        await tester.pump(const Duration(milliseconds: 400));
        expect(model.displayed, same(inspected));
        model.returnToNow();
        await tester.pump();

        // Render only this app's synthetic surface, never the user's desktop/camera.
        final path = Platform.environment['HANDSHAKE_SCREENSHOT'];
        if (path != null) {
          expect(model.latest!.isCamera, isFalse);
          final boundary =
              boundaryKey.currentContext!.findRenderObject()!
                  as RenderRepaintBoundary;
          final image = await boundary.toImage(pixelRatio: 1);
          final data = await image.toByteData(format: ui.ImageByteFormat.png);
          await File(path).writeAsBytes(data!.buffer.asUint8List());
          image.dispose();
        }
        await tester.tap(find.byKey(const Key('stop')));
        await until(
          () => !model.stopPending && model.latest?.state == 'stopped',
        );
        expect(model.latest!.reason, 'user_stop');
        expect(model.history, isEmpty);
        // Explicit opt-in only: routine integration does not touch a camera.
        if (Platform.environment['HANDSHAKE_CAMERA_TEST'] == '1') {
          for (var cycle = 0; cycle < 2; cycle++) {
            await tester.tap(find.byKey(const Key('start_camera')));
            await until(
              () => model.latest?.isCamera == true && model.history.length >= 5,
            );
            bool decodedImage() => tester
                .widgetList<CustomPaint>(find.byType(CustomPaint))
                .map((widget) => widget.painter)
                .whereType<SkeletonPainter>()
                .any(
                  (painter) =>
                      painter.image != null &&
                      painter.snapshot?.isCamera == true,
                );
            await until(decodedImage);
            final frame = model.latest!;
            if (Platform.environment['HANDSHAKE_PERCEPTION_MODEL'] != null) {
              expect(
                frame.perceptionStatus,
                anyOf('processed_hands', 'processed_no_hand'),
              );
            }
            // No images, joints, identities or video are written to the report.
            // ignore: avoid_print
            print(
              jsonEncode({
                'camera_cycle': cycle + 1,
                'decoded': true,
                'width': frame.width,
                'height': frame.height,
                'capture_hz': frame.captureHz,
                'perception_status': frame.perceptionStatus,
              }),
            );
            await tester.tap(find.byKey(const Key('stop')));
            await until(
              () => !model.stopPending && model.latest?.state == 'stopped',
            );
            expect(model.history, isEmpty);
            expect(model.latest!.jpeg, isNull);
          }
        }
        await repository.close();
        await until(() => model.link == LinkState.disconnected);
        expect(model.message, contains('停止未确认'));
        await model.connect();
        await until(() => model.canStart);
        expect(model.latest!.sessionId, isNot(firstSession));
        expect(model.latest!.state, 'stopped');
        expect(tester.takeException(), isNull);
      } finally {
        await tester.pumpWidget(const SizedBox.shrink());
        await tester.binding.setSurfaceSize(null);
        model.dispose();
        await repository.close();
      }
    },
  );
}
