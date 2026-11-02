import 'dart:convert';

/// Bounded, allowlisted lifecycle only. Raw SDK stderr/path/landmarks are discarded.
class CoreDiagnostics {
  static const phases = {
    'worker_started',
    'model_loading',
    'model_ready',
    'model_failed',
    'perception_loading',
    'perception_ready',
    'perception_failed',
    'camera_opening',
    'camera_ready',
    'camera_failed',
    'first_frame',
    'worker_failed',
    'release_started',
    'release_complete',
    'release_failed',
  };
  final List<int> _pending = [];
  bool _discarding = false;
  String? lastPhase;
  String? failurePhase;

  void add(List<int> bytes) {
    for (final byte in bytes) {
      if (byte == 10) {
        if (!_discarding) _line();
        _pending.clear();
        _discarding = false;
      } else if (!_discarding) {
        if (_pending.length >= 1024) {
          _pending.clear();
          _discarding = true;
        } else {
          _pending.add(byte);
        }
      }
    }
  }

  void _line() {
    try {
      final line = utf8.decode(_pending);
      if (!line.startsWith('HS_DIAG ')) return;
      final value = jsonDecode(line.substring(8));
      if (value is! Map<String, dynamic>) return;
      final phase = value['phase'];
      if (phase is! String || !phases.contains(phase)) return;
      lastPhase = phase;
      if (phase.endsWith('_failed')) failurePhase = phase;
    } on FormatException {
      // Native libraries may emit arbitrary non-UTF8 or non-JSON diagnostics.
    }
  }

  String get summary => failurePhase ?? lastPhase ?? 'before_worker_start';
}

class CoreProcessFailure implements Exception {
  CoreProcessFailure(this.phase, {this.exitCode});
  final String phase;
  final int? exitCode;

  @override
  String toString() => '核心失联：阶段=$phase，退出码=${exitCode ?? "未知"}';
}
