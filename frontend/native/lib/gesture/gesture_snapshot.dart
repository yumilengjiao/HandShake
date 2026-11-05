import 'dart:convert';
import 'dart:typed_data';

import '../core/wire_values.dart';

class GestureSnapshot {
  GestureSnapshot._(this._data, this.jpeg, this.lines);
  final Map<String, dynamic> _data;
  final Uint8List jpeg;
  final List<String> lines;
  static const flags = {
    'model_available',
    'camera_released',
    'enabled',
    'owner_bound',
    'eligible_owner',
    'owner_pending',
    'warming',
    'busy',
    'actions_enabled',
    'control_released',
    'rendered',
  };
  static const counters = {
    'sequence',
    'camera_epoch',
    'last_command_id',
    'timestamp_ns',
    'revision',
  };
  static const texts = {'reason', 'model_state', 'policy_reason', 'title'};
  static const nullableTexts = {'top_class', 'model_version', 'model_epoch'};

  static GestureSnapshot parse(String line) {
    final raw = jsonDecode(line);
    final keys = {
      'protocol',
      'version',
      'session_id',
      'scope',
      'external_control_available',
      'experimental',
      'camera_state',
      'page',
      'page_count',
      'lines',
      'jpeg',
      ...flags,
      ...counters,
      ...texts,
      ...nullableTexts,
    };
    if (raw is! Map<String, dynamic> ||
        raw.length != keys.length + (raw.containsKey('diagnostics') ? 1 : 0) ||
        !raw.keys.toSet().containsAll(keys) ||
        raw['protocol'] != 'native-gesture-v1' ||
        raw['version'] is! int ||
        raw['version'] != 1 ||
        raw['scope'] != 'application-local' ||
        raw['external_control_available'] != false ||
        raw['experimental'] != true ||
        raw['session_id'] is! String ||
        !RegExp(r'^[0-9a-f]{32}$').hasMatch(raw['session_id'] as String)) {
      throw const FormatException('Invalid native gesture envelope');
    }
    for (final name in flags) {
      if (raw[name] is! bool) throw const FormatException('Invalid boolean');
    }
    for (final name in counters) {
      wireCounter(raw[name]);
    }
    if (raw['model_epoch'] != null) wireCounter(raw['model_epoch']);
    for (final name in {...texts, ...nullableTexts}) {
      if (nullableTexts.contains(name) && raw[name] == null) continue;
      if (raw[name] is! String ||
          (raw[name] as String).isEmpty ||
          (raw[name] as String).length > 300) {
        throw const FormatException('Invalid gesture text');
      }
    }
    if (!{
          'stopped',
          'starting',
          'running',
          'stopping',
        }.contains(raw['camera_state']) ||
        raw['page'] is! int ||
        raw['page_count'] is! int ||
        raw['page'] < 1 ||
        raw['page_count'] > 32 ||
        raw['page'] > raw['page_count'] ||
        raw['lines'] is! List ||
        (raw['lines'] as List).isEmpty ||
        (raw['lines'] as List).length > 8 ||
        (raw['lines'] as List).any(
          (v) => v is! String || v.isEmpty || v.length > 300,
        ) ||
        raw['jpeg'] is! String ||
        (raw['jpeg'] as String).length > 174764) {
      throw const FormatException('Invalid gesture page or image');
    }
    final diagnostics = raw['diagnostics'];
    if (diagnostics != null) {
      const fields = {
        'window_length',
        'warmup_valid',
        'source_gap_ms',
        'frame_age_ms',
        'max_gap_ms',
        'reset_reason',
      };
      const timingFields = {'capture_hz', 'perception_ms', 'model_ms'};
      if (diagnostics is! Map<String, dynamic> ||
          diagnostics.length !=
              fields.length +
                  timingFields.where(diagnostics.containsKey).length ||
          !diagnostics.keys.toSet().containsAll(fields)) {
        throw const FormatException('Invalid warmup diagnostics');
      }
      for (final key in ['window_length', 'warmup_valid']) {
        final value = diagnostics[key];
        if (value is! int || value < 0 || value > 32) {
          throw const FormatException('Invalid warmup count');
        }
      }
      for (final key in ['source_gap_ms', 'frame_age_ms', 'max_gap_ms']) {
        final value = diagnostics[key];
        if (value == null && key == 'source_gap_ms') continue;
        if (value is! num ||
            !value.isFinite ||
            value < 0 ||
            value > 60000 ||
            key == 'max_gap_ms' && value <= 0) {
          throw const FormatException('Invalid warmup timing');
        }
      }
      for (final key in timingFields) {
        final value = diagnostics[key];
        if (value == null) continue;
        if (value is! num ||
            !value.isFinite ||
            value < 0 ||
            value > (key == 'capture_hz' ? 1000 : 60000)) {
          throw const FormatException('Invalid processing timing');
        }
      }
      final reset = diagnostics['reset_reason'];
      if (reset != null &&
          (reset is! String || reset.isEmpty || reset.length > 128)) {
        throw const FormatException('Invalid warmup reset reason');
      }
      raw['diagnostics'] = Map<String, dynamic>.unmodifiable(diagnostics);
    }
    final jpeg = base64Decode(raw['jpeg'] as String);
    if (jpeg.length > 131072 ||
        (raw['camera_state'] != 'running' &&
            (jpeg.isNotEmpty ||
                raw['enabled'] == true ||
                raw['actions_enabled'] == true)) ||
        (raw['actions_enabled'] == true &&
            (raw['enabled'] != true ||
                raw['owner_bound'] != true ||
                raw['model_epoch'] == null)) ||
        (raw['enabled'] == true && raw['control_released'] == true) ||
        (raw['camera_state'] != 'stopped' && raw['camera_released'] == true)) {
      throw const FormatException('Contradictory camera/control state');
    }
    return GestureSnapshot._(
      Map.unmodifiable(raw),
      jpeg.asUnmodifiableView(),
      List<String>.unmodifiable(raw['lines'] as List),
    );
  }

  bool flag(String name) => _data[name] as bool;
  String text(String name) => _data[name] as String;
  BigInt counter(String name) => wireCounter(_data[name]);
  String get sessionId => text('session_id');
  BigInt get sequence => counter('sequence');
  BigInt get cameraEpoch => counter('camera_epoch');
  BigInt get lastCommandId => counter('last_command_id');
  BigInt get timestampNs => counter('timestamp_ns');
  BigInt? get modelEpoch =>
      _data['model_epoch'] == null ? null : wireCounter(_data['model_epoch']);
  BigInt get revision => counter('revision');
  int get page => _data['page'] as int;
  int get pageCount => _data['page_count'] as int;
  String get cameraState => text('camera_state');
  String? get topClass => _data['top_class'] as String?;
  Map<String, dynamic>? get diagnostics =>
      _data['diagnostics'] as Map<String, dynamic>?;
  int? get warmupValid => diagnostics?['warmup_valid'] as int?;
  double? get sourceGapMs =>
      (diagnostics?['source_gap_ms'] as num?)?.toDouble();
  double? get maxGapMs => (diagnostics?['max_gap_ms'] as num?)?.toDouble();
  bool get samplingOverBudget =>
      sourceGapMs != null && maxGapMs != null && sourceGapMs! > maxGapMs!;
  String get warmupResetText {
    final reason = diagnostics?['reset_reason'] as String?;
    if (reason?.startsWith('quality_invalid:') == true) {
      return '手部质量不足（${reason!.substring(16)}）';
    }
    return switch (reason) {
      null => '无',
      'sampling_gap' => '处理帧间隔超过上限',
      'stale' => '帧已过期',
      'no_hand' => '未检测到手',
      'multiple_hands' => '检测到多只手',
      'source_changed' => '轨迹或控制代际变化',
      'out_of_order' || 'future_input' => '源时间戳异常',
      'perception_error' => '感知处理失败',
      _ => reason,
    };
  }
}
