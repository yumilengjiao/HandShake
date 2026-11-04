import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/main.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  testWidgets(
    'one app: native gesture child, stopped preview, manual tab and back',
    (tester) async {
      final key = GlobalKey();
      await tester.binding.setSurfaceSize(const Size(1280, 900));
      await tester.pumpWidget(
        RepaintBoundary(key: key, child: const HandShakeApp()),
      );
      Future<void> until(bool Function() check) async {
        final deadline = DateTime.now().add(const Duration(seconds: 12));
        while (!check() && DateTime.now().isBefore(deadline)) {
          await tester.pump(const Duration(milliseconds: 30));
        }
        expect(check(), isTrue);
      }

      bool exists(String key) => find.byKey(Key(key)).evaluate().isNotEmpty;
      try {
        await tester.tap(find.byKey(const Key('open-gesture')));
        await until(() => exists('gesture-page'));
        expect(
          tester.widget<Text>(find.byKey(const Key('gesture-page'))).data,
          '1 / 4',
        );
        expect(find.text('相机未打开'), findsOneWidget);
        expect(
          tester
              .widget<OutlinedButton>(find.byKey(const Key('gesture-enable')))
              .onPressed,
          isNull,
        );
        expect(
          tester
              .widget<OutlinedButton>(find.byKey(const Key('gesture-owner')))
              .onPressed,
          isNull,
        );
        await tester.sendKeyEvent(LogicalKeyboardKey.space);
        await tester.sendKeyEvent(LogicalKeyboardKey.escape);
        await until(() => find.textContaining('相机已关闭').evaluate().isNotEmpty);
        await tester.tap(find.byKey(const Key('open-presentation')));
        await until(() => exists('presentation-page'));
        await tester.tap(find.byKey(const Key('open-gesture')));
        await until(() => exists('gesture-page'));
        expect(find.text('相机未打开'), findsOneWidget);
        final destination =
            Platform.environment['HANDSHAKE_GESTURE_SCREENSHOT'];
        if (destination != null) {
          final boundary =
              key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
          final image = await boundary.toImage(pixelRatio: 1);
          final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
          await File(destination).writeAsBytes(bytes!.buffer.asUint8List());
          image.dispose();
        }
        expect(tester.takeException(), isNull);
      } finally {
        await tester.pumpWidget(const SizedBox.shrink());
        await tester.binding.setSurfaceSize(null);
      }
    },
  );
}
