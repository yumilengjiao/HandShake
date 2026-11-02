import 'dart:io';

import 'package:flutter/material.dart';

import 'core/core_repository.dart';
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
  bool presenting = false;
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
    lifecycle = AppLifecycleListener(
      onPause: _stopAll,
      onHide: _stopAll,
      onInactive: () {
        presenter.stop();
        if (model.latest?.version == 5) model.stop();
      },
    );
    // Connecting does not start even a synthetic preview; user intent is required.
    model.connect();
  }

  void _stopAll() {
    model.stop();
    presenter.stop();
  }

  void _select(bool presentation) {
    if (presenting == presentation) return;
    if (presentation) {
      model.stop();
      presenter.activate();
    } else {
      presenter.deactivate();
    }
    setState(() {
      presenting = presentation;
    });
  }

  @override
  void dispose() {
    lifecycle.dispose();
    presenter.dispose();
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
                    onPressed: () => _select(false),
                    child: const Text('观测'),
                  ),
                  TextButton(
                    key: const Key('open-presentation'),
                    onPressed: () => _select(true),
                    child: const Text('演示'),
                  ),
                ],
              ),
            ),
            Expanded(
              child: presenting
                  ? PresentationScreen(model: presenter)
                  : StageScreen(model: model),
            ),
          ],
        ),
      ),
    ),
  );
}
