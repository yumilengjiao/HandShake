import 'dart:async';
import 'dart:convert';
import 'dart:io';

import '../core/core_repository.dart' show boundedLines;
import 'presentation_snapshot.dart';

abstract class PresentationRepository {
  Stream<PresentationSnapshot> get snapshots;
  Future<void> connect();
  void send(String kind, PresentationSnapshot context, BigInt id);
  Future<void> close();
}

/// Separate child from camera/inference. Only this child owns application actions.
class ProcessPresentationRepository implements PresentationRepository {
  ProcessPresentationRepository({required this.python, this.workingDirectory});
  final String python;
  final String? workingDirectory;
  final _snapshots = StreamController<PresentationSnapshot>();
  Process? _process;
  StreamSubscription<String>? _output;
  StreamSubscription<List<int>>? _errors;
  bool _closed = false;

  @override
  Stream<PresentationSnapshot> get snapshots => _snapshots.stream;

  void _fail(Object error) {
    if (!_closed) _snapshots.addError(error);
  }

  @override
  Future<void> connect() async {
    if (_closed || _process != null) {
      throw StateError('Repository already used');
    }
    final process = await Process.start(
      python,
      ['-m', 'backend.presentation_cli'],
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
    _errors = process.stderr.listen((_) {}, onError: _fail);
    _output = boundedLines(process.stdout).listen(
      (line) {
        try {
          _snapshots.add(PresentationSnapshot.parse(line));
        } on FormatException catch (error) {
          _fail(error);
        }
      },
      onError: _fail,
      onDone: () => _fail(StateError('Presentation core disconnected')),
    );
    unawaited(
      process.exitCode.then(
        (_) => _fail(StateError('Presentation core exited')),
      ),
    );
  }

  @override
  void send(String kind, PresentationSnapshot context, BigInt id) {
    if (_closed || _process == null) throw StateError('Not connected');
    _process!.stdin.writeln(
      jsonEncode({
        'version': 1,
        'session_id': context.sessionId,
        'command_id': id.toString(),
        'epoch': context.epoch.toString(),
        'revision': context.revision.toString(),
        'based_on_sequence': context.sequence.toString(),
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
