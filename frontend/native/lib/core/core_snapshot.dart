import 'dart:convert';

/// Immutable v1 DTO. Times and counters never pass through a floating point value.
class StagePoint {
  const StagePoint(this.x, this.y, this.z);
  final double x;
  final double y;
  final double z;
}

BigInt wireCounter(Object? value) {
  if (value is! String || !RegExp(r'^(0|[1-9][0-9]{0,18})$').hasMatch(value)) {
    throw const FormatException('Invalid decimal counter');
  }
  return BigInt.parse(value);
}

class CoreSnapshot {
  CoreSnapshot._(
    this.sessionId,
    this.sequence,
    this.epoch,
    this.lastCommandId,
    this.timestampNs,
    this.state,
    this.reason,
    this.frameId,
    this.frameTimestampNs,
    List<StagePoint> landmarks,
  ) : landmarks = List.unmodifiable(landmarks);

  final String sessionId;
  final BigInt sequence;
  final BigInt epoch;
  final BigInt lastCommandId;
  final BigInt timestampNs;
  final String state;
  final String reason;
  final BigInt frameId;
  final BigInt? frameTimestampNs;
  final List<StagePoint> landmarks;
  bool get isPreview => state == 'preview';

  static const reasons = {
    'initial',
    'synthetic_preview',
    'user_stop',
    'heartbeat_timeout',
    'stale_epoch',
    'protocol_error',
    'input_closed',
    'shutdown',
  };
  static const fields = {
    'version',
    'session_id',
    'sequence',
    'epoch',
    'last_command_id',
    'timestamp_ns',
    'state',
    'reason',
    'source',
    'mode',
    'model_status',
    'control_available',
    'frame_id',
    'frame_timestamp_ns',
    'landmarks',
  };

  factory CoreSnapshot.parse(String line) {
    final value = jsonDecode(line);
    if (value is! Map<String, dynamic> ||
        value.length != fields.length ||
        !fields.containsAll(value.keys)) {
      throw const FormatException('Unknown snapshot shape');
    }
    if (value['version'] is! int ||
        value['version'] != 1 ||
        value['source'] != 'synthetic' ||
        value['mode'] != 'dry_run' ||
        value['model_status'] != 'unavailable' ||
        value['control_available'] != false) {
      throw const FormatException('Unsupported protocol or capability');
    }
    final session = value['session_id'];
    final state = value['state'];
    final reason = value['reason'];
    if (session is! String ||
        !RegExp(r'^[a-f0-9]{32}$').hasMatch(session) ||
        (state != 'stopped' && state != 'preview') ||
        !reasons.contains(reason)) {
      throw const FormatException('Invalid snapshot identity or state');
    }
    final rawPoints = value['landmarks'];
    if (rawPoints is! List ||
        (rawPoints.isNotEmpty && rawPoints.length != 21)) {
      throw const FormatException('Expected zero or 21 landmarks');
    }
    final points = <StagePoint>[];
    for (final point in rawPoints) {
      if (point is! Map ||
          point.length != 3 ||
          !point.keys.toSet().containsAll({'x', 'y', 'z'})) {
        throw const FormatException('Invalid landmark');
      }
      double coordinate(String key) {
        final number = point[key];
        if (number is! num ||
            !number.isFinite ||
            (key != 'z' && (number < 0 || number > 1))) {
          throw const FormatException('Invalid coordinate');
        }
        return number.toDouble();
      }

      points.add(StagePoint(coordinate('x'), coordinate('y'), coordinate('z')));
    }
    final frameTime = value['frame_timestamp_ns'] == null
        ? null
        : wireCounter(value['frame_timestamp_ns']);
    if ((state == 'stopped' && (points.isNotEmpty || frameTime != null)) ||
        (state == 'preview' && frameTime == null)) {
      throw const FormatException('Inconsistent frame state');
    }
    return CoreSnapshot._(
      session,
      wireCounter(value['sequence']),
      wireCounter(value['epoch']),
      wireCounter(value['last_command_id']),
      wireCounter(value['timestamp_ns']),
      state as String,
      reason as String,
      wireCounter(value['frame_id']),
      frameTime,
      points,
    );
  }
}
