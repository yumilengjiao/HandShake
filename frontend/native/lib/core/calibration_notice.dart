import 'dart:math' as math;

import 'model_classification.dart';

/// Validation-only probability telemetry. Accepted means classification, not action.
class CalibrationNotice {
  CalibrationNotice._(
    this.sha256,
    this.synthetic,
    this.temperature,
    this.threshold,
    this.rejectionStatus,
    List<double> values,
  ) : probabilities = List.unmodifiable(values);

  final String sha256;
  final bool synthetic;
  final double temperature;
  final double? threshold;
  final String rejectionStatus;
  final List<double> probabilities;
  double get confidence => probabilities.reduce(math.max);

  static const fields = {
    'calibrated',
    'status',
    'actions_enabled',
    'calibration_sha256',
    'model_version',
    'synthetic',
    'temperature',
    'threshold',
    'probabilities',
    'rejection_status',
  };

  factory CalibrationNotice.parse(
    Object? raw,
    ModelClassification? classification,
  ) {
    if (raw is! Map ||
        raw.length != fields.length ||
        !fields.containsAll(raw.keys) ||
        raw['calibrated'] != true ||
        raw['status'] != 'draft' ||
        raw['actions_enabled'] != false ||
        raw['synthetic'] is! bool ||
        raw['calibration_sha256'] is! String ||
        !RegExp(r'^[a-f0-9]{64}$')
            .hasMatch(raw['calibration_sha256'] as String) ||
        classification == null ||
        classification.state != 'predicted' ||
        raw['model_version'] != classification.modelVersion ||
        (classification.syntheticTraining && raw['synthetic'] != true)) {
      throw const FormatException('Invalid calibration identity/authority');
    }
    final temperature = raw['temperature'];
    final threshold = raw['threshold'];
    final values = raw['probabilities'];
    if (temperature is! num ||
        !temperature.isFinite ||
        temperature < .05 ||
        temperature > 100 ||
        (threshold != null &&
            (threshold is! num ||
                !threshold.isFinite ||
                threshold < 0 ||
                threshold > 1)) ||
        values is! List ||
        values.length != classification.logits.length ||
        values.any((p) => p is! num || !p.isFinite || p < 0 || p > 1)) {
      throw const FormatException('Invalid calibrated probabilities');
    }
    final probabilities = values.map((p) => (p as num).toDouble()).toList();
    if ((probabilities.fold(0.0, (a, b) => a + b) - 1).abs() > 1e-8) {
      throw const FormatException('Probabilities are not normalized');
    }
    final top = classification.topLogit!;
    final scaled = classification.logits
        .map((v) => math.exp((v - top) / temperature))
        .toList();
    final total = scaled.fold(0.0, (a, b) => a + b);
    for (var i = 0; i < scaled.length; i++) {
      if ((probabilities[i] - scaled[i] / total).abs() > 1e-8) {
        throw const FormatException(
          'Calibration does not match current logits',
        );
      }
    }
    final expected = threshold == null
        ? 'unavailable'
        : probabilities.reduce(math.max) >= threshold
        ? 'accepted'
        : 'rejected';
    if (raw['rejection_status'] != expected) {
      throw const FormatException('Inconsistent rejection status');
    }
    return CalibrationNotice._(
      raw['calibration_sha256'] as String,
      raw['synthetic'] as bool,
      temperature.toDouble(),
      (threshold as num?)?.toDouble(),
      expected,
      probabilities,
    );
  }
}
