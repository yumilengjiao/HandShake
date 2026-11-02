import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/core/core_repository.dart';
import 'package:handshake_stage/core/core_snapshot.dart';
import 'package:handshake_stage/stage/stage_screen.dart';
import 'package:handshake_stage/stage/stage_view_model.dart';
import 'package:integration_test/integration_test.dart';

class DiagnosticCore extends ProcessCoreRepository {
  DiagnosticCore({
    required super.python,
    super.workingDirectory,
    super.modelPackage,
    super.modelSha256,
    super.calibration,
    super.calibrationSha256,
  }) : super(
         perceptionModel: 'TEST-SOURCE-NO-REAL-DETECTOR',
         shadowSession: true,
         allowSyntheticModel: true,
       );
  @override
  String get entryModule => 'tests.fixtures.native_shadow_core';

  @override
  void handleStderr(List<int> bytes) {
    debugPrint(utf8.decode(bytes, allowMalformed: true));
  }

  String? _lastState;
  @override
  Stream<CoreSnapshot> get snapshots => super.snapshots
      .map((snapshot) {
        final state =
            '${snapshot.shadow?.state}/${snapshot.shadow?.epoch}/${snapshot.shadow?.reason}';
        if (state != _lastState) {
          debugPrint('Shadow transition: $state');
          _lastState = state;
        }
        return snapshot;
      })
      .handleError((Object error) {
        debugPrint('Procedural shadow core: $error');
        throw error;
      });
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized().framePolicy =
      LiveTestWidgetsFlutterBindingFramePolicy.fullyLive;
  testWidgets(
    'native Python HSP5 owner/rewarm/candidate/stop with procedural video',
    (tester) async {
      expect(Platform.environment['HANDSHAKE_TEST_SOURCE'], '1');
      final python = Platform.environment['HANDSHAKE_PYTHON'];
      final root = Platform.environment['HANDSHAKE_ROOT'];
      expect(python, isNotNull);
      expect(root, isNotNull);
      await tester.binding.setSurfaceSize(const Size(1280, 1000));
      late DiagnosticCore repository;
      final model = StageViewModel(() {
        repository = DiagnosticCore(
          python: python!,
          workingDirectory: root,
          modelPackage: Platform.environment['HANDSHAKE_MODEL_PACKAGE'],
          modelSha256: Platform.environment['HANDSHAKE_MODEL_SHA256'],
          calibration: Platform.environment['HANDSHAKE_CALIBRATION'],
          calibrationSha256:
              Platform.environment['HANDSHAKE_CALIBRATION_SHA256'],
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
            child: Column(
              children: [
                const Material(
                  child: Text('集成测试：程序生成图像/关键点 + 未训练诊断网络；没有打开真实相机'),
                ),
                Expanded(child: StageScreen(model: model)),
              ],
            ),
          ),
        ),
      );
      Future<void> until(bool Function() condition) async {
        final deadline = DateTime.now().add(const Duration(seconds: 15));
        while (!condition() && DateTime.now().isBefore(deadline)) {
          await Future<void>.delayed(const Duration(milliseconds: 20));
        }
        expect(
          condition(),
          isTrue,
          reason:
              '${model.message}; model=${model.latest?.shadow?.reason}; epoch=${model.latest?.shadow?.epoch}; cls=${model.latest?.classification?.reason}/${model.latest?.classification?.windowLength}; offer=${model.latest?.shadow?.eligibleOwner}; surface=${model.latest?.shadow?.surfaceReady}; canOwner=${model.canRequestOwner}',
        );
      }

      Future<void> actWhenReady(
        bool Function() ready,
        VoidCallback intent,
      ) async {
        final completed = Completer<void>();
        var sent = false;
        void onChange() {
          if (!sent && ready()) {
            sent = true;
            intent();
            completed.complete();
          }
        }

        model.addListener(onChange);
        try {
          onChange();
          await completed.future.timeout(const Duration(seconds: 15));
        } finally {
          model.removeListener(onChange);
        }
      }

      try {
        await model.connect();
        await until(() => model.canStartCamera);
        expect(model.latest!.version, 5);
        await tester.pump();
        expect(find.byKey(const Key('start')), findsNothing);
        model.startCamera();
        // Widget tests cover button taps. Native async checks issue one explicit
        // intent on a fresh, genuinely painted offer, not after a pump boundary
        // that may already have consumed its 150ms evidence lifetime.
        await actWhenReady(() => model.canEnableShadow, model.enableShadow);
        late BigInt epoch;
        await actWhenReady(() => model.canRequestOwner, () {
          epoch = model.latest!.shadow!.epoch;
          model.requestOwner();
        });
        await until(() => model.latest?.shadow?.owner != null);
        expect(model.latest!.shadow!.epoch, greaterThan(epoch));
        await until(() => model.latest?.shadow?.candidate != null);
        expect(model.latest!.shadow!.candidate!.action, 'next_slide');
        await tester.pump();
        expect(find.textContaining('未执行'), findsWidgets);
        expect(tester.takeException(), isNull);
        final imagePath = Platform.environment['HANDSHAKE_SHADOW_SCREENSHOT'];
        if (imagePath != null) {
          final boundary =
              boundaryKey.currentContext!.findRenderObject()!
                  as RenderRepaintBoundary;
          final image = await boundary.toImage(pixelRatio: 1);
          final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
          await File(imagePath).writeAsBytes(bytes!.buffer.asUint8List());
          image.dispose();
        }
        await tester.tap(find.byKey(const Key('stop')));
        await until(
          () => !model.stopPending && model.latest?.state == 'stopped',
        );
        expect(model.latest!.shadow, isNull);
        expect(model.history, isEmpty);
        await repository.close();
        await until(() => model.link == LinkState.disconnected);
        await model.connect();
        await until(() => model.canStartCamera);
        expect(model.latest!.state, 'stopped');
      } finally {
        await tester.pumpWidget(const SizedBox.shrink());
        await tester.binding.setSurfaceSize(null);
        model.dispose();
        await repository.close();
      }
    },
  );
  testWidgets(
    'production HSP5 endpoint rejects missing assets before camera access',
    (tester) async {
      final temp = await Directory.systemTemp.createTemp(
        'handshake-shadow-reject-',
      );
      final repository = ProcessCoreRepository(
        python: Platform.environment['HANDSHAKE_PYTHON']!,
        workingDirectory: Platform.environment['HANDSHAKE_ROOT'],
        perceptionModel: '${temp.path}/missing.task',
        modelPackage: '${temp.path}/missing-model',
        modelSha256: List.filled(64, '0').join(),
        calibration: '${temp.path}/missing-calibration.json',
        calibrationSha256: List.filled(64, '0').join(),
        shadowSession: true,
      );
      final model = StageViewModel(() => repository);
      await tester.binding.setSurfaceSize(const Size(1280, 1000));
      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData.dark(),
          home: StageScreen(model: model),
        ),
      );
      Future<void> until(bool Function() condition) async {
        final deadline = DateTime.now().add(const Duration(seconds: 12));
        while (!condition() && DateTime.now().isBefore(deadline)) {
          await tester.pump(const Duration(milliseconds: 20));
        }
        expect(condition(), isTrue, reason: model.message);
      }

      try {
        await model.connect();
        await until(() => model.canStartCamera);
        expect(model.latest!.version, 5);
        await tester.tap(find.byKey(const Key('start_camera')));
        await until(
          () =>
              model.latest?.modelStatus == 'error' &&
              model.latest?.state == 'stopped',
        );
        expect(model.latest!.shadow, isNull);
        expect(model.latest!.jpeg, isNull);
        expect(model.latest!.classification, isNull);
        expect(model.canRequestOwner, isFalse);
        expect(tester.takeException(), isNull);
      } finally {
        await tester.pumpWidget(const SizedBox.shrink());
        await tester.binding.setSurfaceSize(null);
        model.dispose();
        await repository.close();
        await temp.delete(recursive: true);
      }
    },
  );
}
