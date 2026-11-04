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
        raw.length != keys.length ||
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
}
