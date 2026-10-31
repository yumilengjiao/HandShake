import 'dart:async';
import 'dart:convert';
import 'dart:io';

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

class ProcessCoreRepository implements CoreRepository {
  ProcessCoreRepository({
    required this.python,
    this.workingDirectory,
    this.cameraIndex = 0,
    this.perceptionModel,
    this.modelPackage,
    this.modelSha256,
  });
  final String python;
  final String? workingDirectory;
  final int cameraIndex;
  final String? perceptionModel;
  final String? modelPackage;
  final String? modelSha256;
  final _snapshots = StreamController<CoreSnapshot>();
  Process? _process;
  StreamSubscription<CoreSnapshot>? _output;
  StreamSubscription<List<int>>? _errors;
  bool _closed = false;

  @override
  Stream<CoreSnapshot> get snapshots => _snapshots.stream;

  @override
  Future<void> connect() async {
    if (_closed || _process != null) {
      throw StateError('Repository already used');
    }
    final process = await Process.start(
      python,
      [
        '-m',
        'backend.desktop_preview',
        '--camera',
        '$cameraIndex',
        if (perceptionModel != null) ...[
          '--perception-model',
          perceptionModel!,
        ],
        if (modelPackage != null) ...['--model-package', modelPackage!],
        if (modelSha256 != null) ...['--model-sha256', modelSha256!],
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
    _errors = process.stderr.listen((_) {}, onError: _fail);
    _output = decodePreviewPackets(process.stdout).listen(
      (snapshot) => _snapshots.add(snapshot),
      onError: _fail,
      onDone: () => _fail(StateError('Core disconnected')),
    );
    unawaited(process.exitCode.then((_) => _fail(StateError('Core exited'))));
  }

  void _fail(Object error) {
    if (!_closed && !_snapshots.isClosed) _snapshots.addError(error);
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
