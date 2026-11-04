import 'dart:io';

import 'package:flutter/material.dart';

import 'core/core_repository.dart';
import 'gesture/gesture_repository.dart';
import 'gesture/gesture_screen.dart';
import 'gesture/gesture_view_model.dart';
import 'presentation/presentation_repository.dart';
import 'presentation/presentation_screen.dart';
import 'presentation/presentation_view_model.dart';
import 'stage/stage_screen.dart';
import 'stage/stage_view_model.dart';

void main() => runApp(const HandShakeApp());

class HandShakeApp extends StatefulWidget {
  const HandShakeApp({super.key});

  @override
  State<HandShakeApp> createState() => _HandShakeAppState();
}

class _HandShakeAppState extends State<HandShakeApp> {
  late final StageViewModel model;
  late final PresentationViewModel presenter;
  late final GestureViewModel gesture;
  int selected = 0;
  bool switching = false;
  late final AppLifecycleListener lifecycle;

  @override
  void initState() {
    super.initState();
    model = StageViewModel(
      () => ProcessCoreRepository(
        python: Platform.environment['HANDSHAKE_PYTHON'] ?? 'python3',
        workingDirectory: Platform.environment['HANDSHAKE_ROOT'],
        cameraIndex:
            int.tryParse(Platform.environment['HANDSHAKE_CAMERA'] ?? '0') ?? 0,
        perceptionModel: Platform.environment['HANDSHAKE_PERCEPTION_MODEL'],
        modelPackage: Platform.environment['HANDSHAKE_MODEL_PACKAGE'],
        modelSha256: Platform.environment['HANDSHAKE_MODEL_SHA256'],
        calibration: Platform.environment['HANDSHAKE_CALIBRATION'],
        calibrationSha256: Platform.environment['HANDSHAKE_CALIBRATION_SHA256'],
        shadowSession: Platform.environment['HANDSHAKE_SHADOW_SESSION'] == '1',
        allowSyntheticModel:
            Platform.environment['HANDSHAKE_ALLOW_SYNTHETIC_MODEL'] == '1',
      ),
    );
    presenter = PresentationViewModel(
      () => ProcessPresentationRepository(
        python: Platform.environment['HANDSHAKE_PYTHON'] ?? 'python3',
        workingDirectory: Platform.environment['HANDSHAKE_ROOT'],
      ),
    );
    final liveModel =
        Platform.environment['HANDSHAKE_ALLOW_SYNTHETIC_MODEL'] != '1';
    gesture = GestureViewModel(
      () => ProcessGestureRepository(
        python: Platform.environment['HANDSHAKE_PYTHON'] ?? 'python3',
        workingDirectory: Platform.environment['HANDSHAKE_ROOT'],
        camera:
            int.tryParse(Platform.environment['HANDSHAKE_CAMERA'] ?? '0') ?? 0,
        perceptionModel: Platform.environment['HANDSHAKE_PERCEPTION_MODEL'],
        modelPackage: liveModel
            ? Platform.environment['HANDSHAKE_MODEL_PACKAGE']
            : null,
        modelSha256: liveModel
            ? Platform.environment['HANDSHAKE_MODEL_SHA256']
            : null,
        calibration: liveModel
            ? Platform.environment['HANDSHAKE_CALIBRATION']
            : null,
        calibrationSha256: liveModel
            ? Platform.environment['HANDSHAKE_CALIBRATION_SHA256']
            : null,
      ),
    );
    lifecycle = AppLifecycleListener(
      onPause: _stopAll,
      onHide: _stopAll,
      onInactive: _stopAll,
      onResume: () {
        gesture.active = selected == 2;
      },
    );
    // Connecting never opens a camera or enables control.
    model.connect();
    if (liveModel &&
        Platform.environment['HANDSHAKE_MODEL_PACKAGE'] != null &&
        Platform.environment['HANDSHAKE_CALIBRATION'] != null &&
        Platform.environment['HANDSHAKE_SHADOW_SESSION'] != '1') {
      selected = 2;
      gesture.activate();
    }
  }

  void _stopAll() {
    model.stop();
    presenter.stop();
    gesture.deactivate();
  }

  Future<void> _select(int page) async {
    if (selected == page || switching) return;
    setState(() => switching = true);
    final released = selected == 0
        ? await model.stopAndWait()
        : selected == 2
        ? await gesture.closeCameraAndWait()
        : true;
    if (!mounted) return;
    if (!released) {
      setState(() => switching = false);
      return; // Keep the stop/error controls visible; never allocate a second camera.
    }
    presenter.deactivate();
    gesture.active = page == 2;
    if (page == 1) presenter.activate();
    if (page == 2) gesture.activate();
    setState(() {
      selected = page;
      switching = false;
    });
  }

  @override
  void dispose() {
    lifecycle.dispose();
    presenter.dispose();
    gesture.dispose();
    model.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'HandShake · 动作有迹',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      brightness: Brightness.dark,
      useMaterial3: true,
      colorSchemeSeed: const Color(0xff91dcc7),
      scaffoldBackgroundColor: const Color(0xff102127),
    ),
    home: Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: Row(
                children: [
                  const Expanded(
                    child: Text(
                      'HandShake',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  TextButton(
                    onPressed: switching ? null : () => _select(0),
                    child: const Text('观测'),
                  ),
                  TextButton(
                    key: const Key('open-presentation'),
                    onPressed: switching ? null : () => _select(1),
                    child: const Text('人工演示'),
                  ),
                  TextButton(
                    key: const Key('open-gesture'),
                    onPressed: switching ? null : () => _select(2),
                    child: const Text('手势翻页'),
                  ),
                ],
              ),
            ),
            Expanded(
              child: switch (selected) {
                1 => PresentationScreen(model: presenter),
                2 => GestureScreen(model: gesture),
                _ => StageScreen(model: model),
              },
            ),
          ],
        ),
      ),
    ),
  );
}
