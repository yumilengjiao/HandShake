import 'wire_values.dart';

/// Read-only raw network scores; never calibrated probability or action authority.
class ModelClassification {
  ModelClassification._(
    this.state,
    this.reason,
    this.modelVersion,
    this.modelType,
    this.vocabularyVersion,
    this.syntheticTraining,
    this.streamId,
    this.epoch,
    this.frameId,
    this.sourceNs,
    this.trackId,
    this.windowLength,
    this.inferenceNs,
    List<double> scores,
  ) : logits = List.unmodifiable(scores);

  final String state;
  final String reason;
  final String modelVersion;
  final String modelType;
  final String vocabularyVersion;
  final bool syntheticTraining;
  final String streamId;
  final BigInt epoch;
  final BigInt frameId;
  final BigInt sourceNs;
  final String? trackId;
  final int windowLength;
  final BigInt? inferenceNs;
  final List<double> logits;

  static const vocabularies = {
    'gesture-v1-draft': [
      'no_action',
      'wake_open',
      'swipe_left',
      'swipe_right',
      'exit_pose',
    ],
    'ipn-hands-v1': [
      'NOG',
      'P1F',
      'P2F',
      'C1F',
      'C2F',
      'TUP',
      'TDN',
      'TLT',
      'TRT',
      'OP2',
      '2C1',
      '2C2',
      'ZIN',
      'ZOT',
      'GRB',
    ],
  };
  static const fields = {
    'state',
    'reason',
    'model_version',
    'model_type',
    'vocabulary_version',
    'synthetic_training',
    'calibrated',
    'stream_id',
    'context_epoch',
    'source_frame_id',
    'source_timestamp_ns',
    'track_id',
    'window_length',
    'inference_ns',
    'logits',
  };
  static const reasons = {
    'ready',
    'warming_up',
    'current_observation_invalid',
    'history_uninitialized',
    'stale',
    'insufficient_valid_ratio',
    'window_span_exceeded',
    'no_hand',
    'multiple_hands',
    'not_processed',
    'perception_error',
    'out_of_order',
    'future_input',
    'inference_error',
    'cancelled',
  };

  String? get topToken {
    if (logits.isEmpty) return null;
    var best = 0;
    for (var i = 1; i < logits.length; i++) {
      if (logits[i] > logits[best]) best = i;
    }
    return vocabularies[vocabularyVersion]![best];
  }

  double? get topLogit =>
      logits.isEmpty ? null : logits.reduce((a, b) => a > b ? a : b);

  factory ModelClassification.parse(Object? value) {
    if (value is! Map ||
        value.length != fields.length ||
        !fields.containsAll(value.keys)) {
      throw const FormatException('Invalid classification fields');
    }
    String identifier(String key, int max) {
      final item = value[key];
      if (item is! String || item.isEmpty || item.length > max) {
        throw const FormatException('Invalid classification ID');
      }
      return item;
    }

    final state = value['state'];
    final reason = value['reason'];
    final version = identifier('vocabulary_version', 40);
    final length = value['window_length'];
    final raw = value['logits'];
    if (!{'warming_up', 'rejected', 'predicted', 'error'}.contains(state) ||
        !reasons.contains(reason) ||
        !vocabularies.containsKey(version) ||
        !{'mlp', 'tcn'}.contains(value['model_type']) ||
        value['synthetic_training'] is! bool ||
        value['calibrated'] != false ||
        length is! int ||
        length < 0 ||
        length > 32 ||
        raw is! List ||
        raw.length > 15 ||
        raw.any((number) => number is! num || !number.isFinite)) {
      throw const FormatException('Invalid classification semantics');
    }
    if ((state == 'predicted') != (reason == 'ready')) {
      throw const FormatException('Only predictions may claim ready');
    }
    final inferenceNs = value['inference_ns'] == null
        ? null
        : wireCounter(value['inference_ns']);
    final track = value['track_id'] == null
        ? null
        : identifier('track_id', 128);
    if ((state == 'predicted' &&
            (reason != 'ready' ||
                raw.length != vocabularies[version]!.length ||
                track == null ||
                length != 32 ||
                inferenceNs == null)) ||
        (state != 'predicted' && (raw.isNotEmpty || inferenceNs != null))) {
      throw const FormatException('Incomplete prediction or stale scores');
    }
    return ModelClassification._(
      state as String,
      reason as String,
      identifier('model_version', 80),
      value['model_type'] as String,
      version,
      value['synthetic_training'] as bool,
      identifier('stream_id', 128),
      wireCounter(value['context_epoch']),
      wireCounter(value['source_frame_id']),
      wireCounter(value['source_timestamp_ns']),
      track,
      length,
      inferenceNs,
      raw.map((v) => (v as num).toDouble()).toList(),
    );
  }
}
