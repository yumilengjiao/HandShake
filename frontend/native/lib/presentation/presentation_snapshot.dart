import 'dart:convert';

import '../core/wire_values.dart';

class PageReceipt {
  const PageReceipt(
    this.requestId,
    this.action,
    this.outcome,
    this.reason,
    this.before,
    this.after,
    this.revision,
  );
  final BigInt requestId;
  final String action;
  final String outcome;
  final String? reason;
  final int before;
  final int after;
  final BigInt revision;
}

class PresentationSnapshot {
  PresentationSnapshot._(Map<String, dynamic> data)
    : sessionId = data['session_id'],
      sequence = wireCounter(data['sequence']),
      epoch = wireCounter(data['epoch']),
      lastCommandId = wireCounter(data['last_command_id']),
      timestampNs = wireCounter(data['timestamp_ns']),
      state = data['state'],
      reason = data['reason'],
      deckId = data['deck_id'],
      deckTitle = data['deck_title'],
      page = data['page'],
      pageCount = data['page_count'],
      revision = wireCounter(data['revision']),
      renderedRevision = data['rendered_revision'] == null
          ? null
          : wireCounter(data['rendered_revision']),
      busy = data['busy'],
      released = data['release_confirmed'],
      title = data['slide']['title'],
      lines = List.unmodifiable(
        (data['slide']['lines'] as List).cast<String>(),
      ),
      lastAction = data['last_action'] == null
          ? null
          : _receipt(data['last_action']);

  final String sessionId, state, reason, deckId, deckTitle, title;
  final BigInt sequence, epoch, lastCommandId, timestampNs, revision;
  final BigInt? renderedRevision;
  final int page, pageCount;
  final bool busy, released;
  final List<String> lines;
  final PageReceipt? lastAction;
  bool get rendered => renderedRevision == revision;

  static const fields = {
    'version',
    'session_id',
    'sequence',
    'epoch',
    'last_command_id',
    'timestamp_ns',
    'scope',
    'control_origin',
    'external_control_available',
    'gesture_control_available',
    'state',
    'reason',
    'deck_id',
    'deck_title',
    'page',
    'page_count',
    'revision',
    'rendered_revision',
    'busy',
    'release_confirmed',
    'slide',
    'last_action',
  };

  factory PresentationSnapshot.parse(String line) {
    final data = jsonDecode(line);
    if (data is! Map<String, dynamic> ||
        data.length != fields.length ||
        !fields.containsAll(data.keys) ||
        data['version'] is! int ||
        data['version'] != 1 ||
        data['scope'] != 'application-local' ||
        data['control_origin'] != 'manual' ||
        data['external_control_available'] != false ||
        data['gesture_control_available'] != false ||
        !{'stopped', 'playing'}.contains(data['state']) ||
        data['busy'] is! bool ||
        data['release_confirmed'] is! bool ||
        data['session_id'] is! String ||
        !RegExp(r'^[0-9a-f]{32}$').hasMatch(data['session_id']) ||
        data['deck_id'] is! String ||
        !RegExp(r'^[0-9a-f]{64}$').hasMatch(data['deck_id'])) {
      throw const FormatException('Unsupported presentation state/capability');
    }
    _text(data['reason'], 128);
    _text(data['deck_title'], 200);
    _page(data['page']);
    _page(data['page_count']);
    if (data['page'] > data['page_count']) {
      throw const FormatException('Page outside deck');
    }
    final slide = data['slide'];
    if (slide is! Map ||
        slide.length != 2 ||
        !slide.keys.toSet().containsAll({'title', 'lines'}) ||
        slide['lines'] is! List ||
        (slide['lines'] as List).isEmpty ||
        (slide['lines'] as List).length > 8) {
      throw const FormatException('Invalid slide');
    }
    _text(slide['title'], 200);
    for (final value in slide['lines']) {
      _text(value, 300);
    }
    final value = PresentationSnapshot._(data);
    if (value.renderedRevision != null &&
        value.renderedRevision! > value.revision) {
      throw const FormatException('Future render receipt');
    }
    final result = value.lastAction;
    if (result != null &&
        (result.before > value.pageCount ||
            result.after > value.pageCount ||
            result.revision > value.revision ||
            (result.outcome == 'application_updated' &&
                result.before == result.after) ||
            (result.outcome == 'boundary_no_change' &&
                result.before != result.after))) {
      throw const FormatException('Action receipt contradicts deck/page state');
    }
    return value;
  }

  static void _text(Object? value, int max) {
    if (value is! String || value.isEmpty || value.length > max) {
      throw const FormatException('Invalid presentation text');
    }
  }

  static void _page(Object? value) {
    if (value is! int || value < 1 || value > 32) {
      throw const FormatException('Invalid page');
    }
  }

  static PageReceipt _receipt(dynamic value) {
    const fields = {
      'request_id',
      'action',
      'outcome',
      'reason',
      'page_before',
      'page_after',
      'revision',
    };
    if (value is! Map ||
        value.length != fields.length ||
        !fields.containsAll(value.keys) ||
        !{'next_slide', 'previous_slide'}.contains(value['action']) ||
        !{
          'application_updated',
          'boundary_no_change',
          'rejected',
          'uncertain',
        }.contains(value['outcome'])) {
      throw const FormatException('Invalid page receipt');
    }
    _page(value['page_before']);
    _page(value['page_after']);
    if (value['reason'] != null) _text(value['reason'], 128);
    return PageReceipt(
      wireCounter(value['request_id']),
      value['action'],
      value['outcome'],
      value['reason'],
      value['page_before'],
      value['page_after'],
      wireCounter(value['revision']),
    );
  }
}
