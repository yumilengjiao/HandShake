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
    'native application-local presentation: 1 -> 2 -> 3 -> 2, render and stop',
    (tester) async {
      expect(Platform.environment['HANDSHAKE_PYTHON'], isNotNull);
      final key = GlobalKey();
      await tester.binding.setSurfaceSize(const Size(1280, 900));
      await tester.pumpWidget(
        RepaintBoundary(key: key, child: const HandShakeApp()),
      );
      Future<void> until(bool Function() check) async {
        final deadline = DateTime.now().add(const Duration(seconds: 8));
        while (!check() && DateTime.now().isBefore(deadline)) {
          await tester.pump(const Duration(milliseconds: 20));
        }
        expect(check(), isTrue);
      }

      bool enabled(String name) {
        final finder = find.byKey(Key(name));
        return finder.evaluate().isNotEmpty &&
            tester.widget<OutlinedButton>(finder).onPressed != null;
      }

      String? page() =>
          find.byKey(const Key('presentation-page')).evaluate().isEmpty
          ? null
          : tester
                .widget<Text>(find.byKey(const Key('presentation-page')))
                .data;
      try {
        await tester.tap(find.byKey(const Key('open-presentation')));
        await until(() => enabled('presentation-play'));
        expect(page(), '01 / 04');
        await tester.tap(find.byKey(const Key('presentation-play')));
        await until(() => enabled('presentation-next'));
        for (final (button, expected) in [
          ('presentation-next', '02 / 04'),
          ('presentation-next', '03 / 04'),
          ('presentation-previous', '02 / 04'),
        ]) {
          await tester.tap(find.byKey(Key(button)));
          await until(() => page() == expected && enabled('presentation-next'));
          expect(find.textContaining('Flutter已绘制该修订'), findsOneWidget);
        }
        final screenshot =
            Platform.environment['HANDSHAKE_PRESENTATION_SCREENSHOT'];
        if (screenshot != null) {
          final boundary =
              key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
          final image = await boundary.toImage(pixelRatio: 1);
          final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
          await File(screenshot).writeAsBytes(bytes!.buffer.asUint8List());
          image.dispose();
        }
        await tester.sendKeyDownEvent(LogicalKeyboardKey.arrowRight);
        await until(() => page() == '03 / 04' && enabled('presentation-next'));
        await tester.sendKeyRepeatEvent(LogicalKeyboardKey.arrowRight);
        await tester.pump(const Duration(milliseconds: 200));
        expect(page(), '03 / 04');
        await tester.sendKeyUpEvent(LogicalKeyboardKey.arrowRight);
        await tester.sendKeyEvent(LogicalKeyboardKey.escape);
        await until(() => find.textContaining('核心已停止').evaluate().isNotEmpty);
        expect(enabled('presentation-next'), isFalse);
        expect(tester.takeException(), isNull);
      } finally {
        await tester.pumpWidget(const SizedBox.shrink());
        await tester.binding.setSurfaceSize(null);
      }
    },
  );
}
