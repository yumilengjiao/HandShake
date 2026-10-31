import 'dart:convert';
import 'dart:typed_data';

import 'model_classification.dart';
import 'wire_values.dart';

export 'wire_values.dart' show wireCounter;

/// Immutable protocol objects; nanoseconds/counters never pass through double.
class StagePoint {
  const StagePoint(this.x, this.y, this.z);
  final double x;
  final double y;
  final double z;
}

class StageHand {
  StageHand(List<StagePoint> points, List<bool> validity, this.usable)
    : points = List.unmodifiable(points),
      validity = List.unmodifiable(validity);
  final List<StagePoint> points;
  final List<bool> validity;
  final bool usable;
}

class CoreSnapshot {
  CoreSnapshot._({
    required this.version,
    required this.sessionId,
    required this.sequence,
    required this.epoch,
    required this.lastCommandId,
    required this.timestampNs,
    required this.state,
    required this.reason,
    required this.frameId,
    required this.frameTimestampNs,
    required this.source,
    required List<StageHand> hands,
    required this.width,
    required this.height,
    required this.perceptionStatus,
    required this.captureHz,
    required this.modelStatus,
    this.classification,
    Uint8List? jpeg,
  }) : hands = List.unmodifiable(hands),
       jpeg = jpeg == null
           ? null
           : Uint8List.fromList(jpeg).asUnmodifiableView();

  final int version;
  final String sessionId;
  final BigInt sequence;
  final BigInt epoch;
  final BigInt lastCommandId;
  final BigInt timestampNs;
  final String state;
  final String reason;
  final BigInt frameId;
  final BigInt? frameTimestampNs;
  final String source;
  final List<StageHand> hands;
  List<StagePoint> get landmarks =>
      hands.isEmpty ? const [] : hands.first.points;
  final int width;
  final int height;
  final String perceptionStatus;
  final double? captureHz;
  final Uint8List? jpeg;
  final String modelStatus;
  final ModelClassification? classification;
  bool get isPreview => state == 'preview';
  bool get isCamera => source == 'camera';

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
  static const cameraReasons = {
    'camera_starting',
    'camera_preview',
    'camera_unavailable',
    'perception_unavailable',
    'input_stalled',
    'preview_error',
  };
  static const perceptionStates = {
    'unavailable',
    'processed_hands',
    'processed_no_hand',
    'not_processed',
    'error',
  };

  factory CoreSnapshot.parse(String line, {Uint8List? jpeg}) {
    final value = jsonDecode(line);
    if (value is! Map<String, dynamic>) {
      throw const FormatException('Unknown snapshot shape');
    }
    final version = value['version'];
    final v2 = version == 2 || version == 3;
    final expected = v2
        ? ({...fields}
            ..remove('landmarks')
            ..addAll({
              'frame_width',
              'frame_height',
              'timestamp_kind',
              'capture_hz',
              'perception_status',
              'hands',
              if (version == 3) 'classification',
            }))
        : fields;
    if (value.length != expected.length || !expected.containsAll(value.keys)) {
      throw const FormatException('Unknown snapshot fields');
    }
    if (version is! int ||
        (version != 1 && version != 2 && version != 3) ||
        (value['source'] != 'synthetic' &&
            !(v2 && value['source'] == 'camera')) ||
        value['mode'] != 'dry_run' ||
        !(version == 3
            ? {
                'unavailable',
                'loaded_uncalibrated',
                'error',
              }.contains(value['model_status'])
            : value['model_status'] == 'unavailable') ||
        value['control_available'] != false) {
      throw const FormatException('Unsupported protocol or capability');
    }
    final session = value['session_id'];
    final state = value['state'];
    final reason = value['reason'];
    if (session is! String ||
        !RegExp(r'^[a-f0-9]{32}$').hasMatch(session) ||
        !{
          'stopped',
          'preview',
          if (v2) ...['starting', 'stopping'],
        }.contains(state) ||
        !{...reasons, if (v2) ...cameraReasons}.contains(reason)) {
      throw const FormatException('Invalid snapshot identity or state');
    }
    final hands = <StageHand>[];
    if (v2) {
      final raw = value['hands'];
      if (raw is! List || raw.length > 2) {
        throw const FormatException('Too many hands');
      }
      for (final hand in raw) {
        if (hand is! Map ||
            hand.length != 3 ||
            !hand.keys.toSet().containsAll({
              'landmarks',
              'validity',
              'usable',
            }) ||
            hand['usable'] is! bool ||
            hand['validity'] is! List ||
            (hand['validity'] as List).length != 21 ||
            (hand['validity'] as List).any((v) => v is! bool)) {
          throw const FormatException('Invalid hand');
        }
        final points = _points(hand['landmarks'], v2: true);
        if (points.length != 21) throw const FormatException('Incomplete hand');
        hands.add(
          StageHand(
            points,
            (hand['validity'] as List).cast<bool>(),
            hand['usable'] as bool,
          ),
        );
      }
    } else {
      final points = _points(value['landmarks'], v2: false);
      if (points.isNotEmpty) {
        hands.add(StageHand(points, List.filled(21, true), true));
      }
    }
    final frameTime = value['frame_timestamp_ns'] == null
        ? null
        : wireCounter(value['frame_timestamp_ns']);
    final width = v2 ? value['frame_width'] : (state == 'preview' ? 640 : 0);
    final height = v2 ? value['frame_height'] : (state == 'preview' ? 480 : 0);
    if (width is! int ||
        height is! int ||
        width < 0 ||
        width > 640 ||
        height < 0 ||
        height > 480 ||
        (state != 'preview' &&
            (hands.isNotEmpty ||
                frameTime != null ||
                width != 0 ||
                height != 0)) ||
        (state == 'preview' &&
            (frameTime == null || width == 0 || height == 0))) {
      throw const FormatException('Inconsistent frame state');
    }
    final camera = value['source'] == 'camera';
    final status = v2
        ? value['perception_status']
        : (hands.isEmpty ? 'processed_no_hand' : 'processed_hands');
    final hz = v2 ? value['capture_hz'] : null;
    if (!perceptionStates.contains(status) ||
        (hands.isNotEmpty && status != 'processed_hands') ||
        (hz != null && (hz is! num || !hz.isFinite || hz < 0 || hz > 1000)) ||
        (v2 &&
            value['timestamp_kind'] !=
                (camera
                    ? 'software_read_complete_not_exposure'
                    : 'synthetic_generation'))) {
      throw const FormatException('Invalid perception/clock metadata');
    }
    final hasImage = jpeg != null && jpeg.isNotEmpty;
    if (hasImage != (camera && state == 'preview') ||
        (jpeg?.length ?? 0) > 131072 ||
        (hasImage &&
            (jpeg.length < 4 ||
                jpeg[0] != 255 ||
                jpeg[1] != 216 ||
                jpeg[jpeg.length - 2] != 255 ||
                jpeg[jpeg.length - 1] != 217))) {
      throw const FormatException(
        'Invalid JPEG payload or image/state mismatch',
      );
    }
    final classification = version == 3 && value['classification'] != null
        ? ModelClassification.parse(value['classification'])
        : null;
    if (classification != null) {
      if (!camera ||
          state != 'preview' ||
          value['model_status'] != 'loaded_uncalibrated' ||
          classification.epoch != wireCounter(value['epoch']) ||
          classification.frameId != wireCounter(value['frame_id']) ||
          classification.sourceNs != frameTime) {
        throw const FormatException(
          'Classification does not belong to this frame/epoch',
        );
      }
      if (classification.state == 'predicted' &&
          (status != 'processed_hands' ||
              hands.length != 1 ||
              !hands.single.usable ||
              hands.single.validity.any((v) => !v))) {
        throw const FormatException(
          'Prediction requires one valid observed hand',
        );
      }
    } else if (value['model_status'] == 'loaded_uncalibrated') {
      throw const FormatException('Missing model status for current frame');
    }
    return CoreSnapshot._(
      version: version,
      sessionId: session,
      sequence: wireCounter(value['sequence']),
      epoch: wireCounter(value['epoch']),
      lastCommandId: wireCounter(value['last_command_id']),
      timestampNs: wireCounter(value['timestamp_ns']),
      state: state as String,
      reason: reason as String,
      frameId: wireCounter(value['frame_id']),
      frameTimestampNs: frameTime,
      source: value['source'] as String,
      hands: hands,
      width: width,
      height: height,
      perceptionStatus: status as String,
      captureHz: (hz as num?)?.toDouble(),
      modelStatus: value['model_status'] as String,
      classification: classification,
      jpeg: jpeg,
    );
  }

  static List<StagePoint> _points(Object? raw, {required bool v2}) {
    if (raw is! List || (raw.isNotEmpty && raw.length != 21)) {
      throw const FormatException('Expected zero or 21 landmarks');
    }
    return raw.map((point) {
      if (point is! Map ||
          point.length != 3 ||
          !point.keys.toSet().containsAll({'x', 'y', 'z'})) {
        throw const FormatException('Invalid landmark');
      }
      double coordinate(String key) {
        final number = point[key];
        if (number is! num ||
            !number.isFinite ||
            (v2 && (number < -16 || number > 16)) ||
            (!v2 && key != 'z' && (number < 0 || number > 1))) {
          throw const FormatException('Invalid coordinate');
        }
        return number.toDouble();
      }

      return StagePoint(coordinate('x'), coordinate('y'), coordinate('z'));
    }).toList();
  }
}
