import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'core_diagnostics.dart';
import 'core_snapshot.dart';
import 'preview_packets.dart';

abstract class CoreRepository {
  Stream<CoreSnapshot> get snapshots;
  Future<void> connect();
  void send(String kind, CoreSnapshot context, BigInt commandId);
  Future<void> close();
}

/// Bound bytes *before* decoding; LineSplitter alone buffers unlimited input.
Stream<String> boundedLines(
  Stream<List<int>> bytes, {
  int limit = 16384,
}) async* {
  var pending = <int>[];
  await for (final chunk in bytes) {
    for (final byte in chunk) {
      if (byte == 10) {
        yield utf8.decode(pending);
        pending = <int>[];
      } else {
        pending.add(byte);
        if (pending.length > limit) {
          throw const FormatException('Snapshot line exceeds size limit');
        }
      }
    }
  }
  if (pending.isNotEmpty) {
    throw const FormatException('Truncated snapshot');
  }
}

Map<String, Object?> shadowCommandFields(String kind, CoreSnapshot context) {
  final shadow = context.shadow;
  final contextual =
      {'surface_ack', 'enable_shadow', 'request_owner'}.contains(kind) ||
      kind == 'heartbeat' && shadow != null;
  if (contextual && shadow == null) {
    throw StateError('No captured shadow context');
  }
  return {
    'model_epoch': contextual ? shadow!.epoch.toString() : null,
    'model_sequence': contextual ? shadow!.sequence.toString() : null,
    'model_revision': contextual ? shadow!.revision.toString() : null,
    'track_id': kind == 'request_owner' ? shadow?.eligibleOwner : null,
  };
}

class ProcessCoreRepository implements CoreRepository {
  ProcessCoreRepository({
    required this.python,
    this.workingDirectory,
    this.cameraIndex = 0,
    this.perceptionModel,
    this.modelPackage,
    this.modelSha256,
    this.calibration,
    this.calibrationSha256,
    this.shadowSession = false,
    this.allowSyntheticModel = false,
  });
  final String python;
  final String? workingDirectory;
  final int cameraIndex;
  final String? perceptionModel;
  final String? modelPackage;
  final String? modelSha256;
  final String? calibration;
  final String? calibrationSha256;
  final bool shadowSession;
  final bool allowSyntheticModel;
  final _snapshots = StreamController<CoreSnapshot>();
  Process? _process;
  StreamSubscription<CoreSnapshot>? _output;
  StreamSubscription<List<int>>? _errors;
  bool _closed = false;
  bool _failed = false;
  final diagnostics = CoreDiagnostics();

  @override
  Stream<CoreSnapshot> get snapshots => _snapshots.stream;

  /// Fixed in the application; test subclasses may select a procedural core.
  String get entryModule => 'backend.desktop_preview';

  /// Optional test sink. Production only retains bounded allowlisted lifecycle phases.
  void handleStderr(List<int> bytes) {}

  @override
  Future<void> connect() async {
    if (_closed || _process != null) {
      throw StateError('Repository already used');
    }
    final process = await Process.start(
      python,
      [
        '-m',
        entryModule,
        '--camera',
        '$cameraIndex',
        if (perceptionModel != null) ...[
          '--perception-model',
          perceptionModel!,
        ],
        if (modelPackage != null) ...['--model-package', modelPackage!],
        if (modelSha256 != null) ...['--model-sha256', modelSha256!],
        if (shadowSession) '--shadow-session',
        if (allowSyntheticModel) '--allow-synthetic-model',
        if (calibration != null) ...['--calibration', calibration!],
        if (calibrationSha256 != null) ...[
          '--calibration-sha256',
          calibrationSha256!,
        ],
      ],
      workingDirectory: workingDirectory,
      runInShell: false,
    );
    if (_closed) {
      process.kill();
      return;
    }
    _process = process;
    process.stdin.done.catchError((Object error) {
      _fail(error);
    });
    // Drain without collecting logs or exposing local paths in UI.
    _errors = process.stderr.listen((bytes) {
      diagnostics.add(bytes);
      handleStderr(bytes);
    }, onError: _fail);
    _output = decodePreviewPackets(process.stdout).listen(
      (snapshot) {
        if (!_closed && !_failed) _snapshots.add(snapshot);
      },
      onError: _fail,
      onDone: () => unawaited(_exited(process)),
    );
    unawaited(
      process.exitCode.then((code) {
        _fail(CoreProcessFailure(diagnostics.summary, exitCode: code));
      }),
    );
  }

  Future<void> _exited(Process process) async {
    // EOF and exit can arrive in either order; wait briefly for the real code,
    // never report exit=0 when a native library was killed or crashed.
    int? code;
    try {
      code = await process.exitCode.timeout(const Duration(milliseconds: 300));
    } on TimeoutException {
      // A closed protocol with a still-running process is also a failure.
    }
    _fail(CoreProcessFailure(diagnostics.summary, exitCode: code));
  }

  void _fail(Object error) {
    if (!_closed && !_failed && !_snapshots.isClosed) {
      _failed = true;
      _snapshots.addError(error);
    }
  }

  @override
  void send(String kind, CoreSnapshot context, BigInt commandId) {
    final process = _process;
    if (_closed || process == null) throw StateError('Core not connected');
    // ViewModel limits outstanding commands; never retry a state-changing intent.
    process.stdin.writeln(
      jsonEncode({
        'version': context.version,
        'session_id': context.sessionId,
        'command_id': commandId.toString(),
        'epoch': context.epoch.toString(),
        'kind': kind,
        if (context.version == 5) ...shadowCommandFields(kind, context),
      }),
    );
  }

  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await _output?.cancel();
    await _errors?.cancel();
    final process = _process;
    if (process != null) {
      // EOF is the graceful stop path. Kill is bounded and only for this child.
      try {
        await process.stdin.close().timeout(const Duration(milliseconds: 200));
        await process.exitCode.timeout(const Duration(milliseconds: 400));
      } on Object {
        process.kill();
        try {
          await process.exitCode.timeout(const Duration(milliseconds: 300));
        } on Object {
          process.kill(ProcessSignal.sigkill);
        }
      }
    }
    await _snapshots.close();
  }
}
