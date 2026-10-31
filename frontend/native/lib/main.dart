import 'dart:io';

import 'package:flutter/material.dart';

import 'core/core_repository.dart';
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
      ),
    );
    lifecycle = AppLifecycleListener(onPause: model.stop, onHide: model.stop);
    // Connecting does not start even a synthetic preview; user intent is required.
    model.connect();
  }

  @override
  void dispose() {
    lifecycle.dispose();
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
    home: StageScreen(model: model),
  );
}
