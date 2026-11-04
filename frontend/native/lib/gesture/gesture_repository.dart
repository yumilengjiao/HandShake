import 'dart:async';
import 'dart:convert';
import 'dart:io';

import '../core/core_repository.dart' show boundedLines;
import 'gesture_snapshot.dart';

abstract class GestureRepository {
  Stream<GestureSnapshot> get snapshots;
  Future<void> connect();
  void send(String kind, GestureSnapshot context, BigInt id);
  Future<void> close();
}

class ProcessGestureRepository implements GestureRepository {
  ProcessGestureRepository({
    required this.python,
    this.workingDirectory,
    this.camera = 0,
    this.perceptionModel,
    this.modelPackage,
    this.modelSha256,
    this.calibration,
    this.calibrationSha256,
  });
  final String python;
  final String? workingDirectory,
      perceptionModel,
      modelPackage,
      modelSha256,
      calibration,
      calibrationSha256;
  final int camera;
  final _snapshots = StreamController<GestureSnapshot>();
  Process? _process;
  StreamSubscription<String>? _output;
  StreamSubscription<List<int>>? _errors;
  bool _closed = false;
  Future<void>? _closing;

  @override
  Stream<GestureSnapshot> get snapshots => _snapshots.stream;
  void _fail(Object error) {
    if (!_closed) _snapshots.addError(error);
  }

  @override
  Future<void> connect() async {
    if (_closed || _process != null) {
      throw StateError('Repository already used');
    }
    final args = ['-m', 'backend.native_gesture_cli', '--camera', '$camera'];
    if (perceptionModel != null) {
      args.addAll(['--perception-model', perceptionModel!]);
    }
    // Raw-only/diagnostic previews do not gain live authority.
    if (modelPackage != null &&
        modelSha256 != null &&
        calibration != null &&
        calibrationSha256 != null) {
      args.addAll([
        '--model-package',
        modelPackage!,
        '--model-sha256',
        modelSha256!,
        '--calibration',
        calibration!,
        '--calibration-sha256',
        calibrationSha256!,
      ]);
    }
    final process = await Process.start(
      python,
      args,
      workingDirectory: workingDirectory,
      runInShell: false,
    );
    if (_closed) {
      process.kill();
      await _terminate(process);
      return;
    }
    _process = process;
    process.stdin.done.catchError((Object error) {
      _fail(error);
    });
    _errors = process.stderr.listen((_) {}, onError: _fail);
    _output = boundedLines(process.stdout, limit: 196608).listen(
      (line) {
        try {
          _snapshots.add(GestureSnapshot.parse(line));
        } on FormatException catch (error) {
          _fail(error);
        }
      },
      onError: _fail,
      onDone: () => _fail(StateError('Gesture core disconnected')),
    );
    unawaited(
      process.exitCode.then((_) => _fail(StateError('Gesture core exited'))),
    );
  }

  @override
  void send(String kind, GestureSnapshot context, BigInt id) {
    if (_closed || _process == null) throw StateError('Not connected');
    _process!.stdin.writeln(
      jsonEncode({
        'version': 1,
        'session_id': context.sessionId,
        'command_id': id.toString(),
        'camera_epoch': context.cameraEpoch.toString(),
        'based_on_sequence': context.sequence.toString(),
        'kind': kind,
      }),
    );
  }

  Future<void> _terminate(Process process) async {
    try {
      await process.stdin.close().timeout(const Duration(milliseconds: 200));
      await process.exitCode.timeout(const Duration(seconds: 1));
    } on Object {
      process.kill();
      try {
        await process.exitCode.timeout(const Duration(milliseconds: 500));
      } on Object {
        process.kill(ProcessSignal.sigkill);
        await process.exitCode.timeout(const Duration(seconds: 1));
      }
    }
  }

  @override
  Future<void> close() => _closing ??= _close();
  Future<void> _close() async {
    _closed = true;
    await _output?.cancel();
    await _errors?.cancel();
    if (_process != null) await _terminate(_process!);
    await _snapshots.close();
  }
}
