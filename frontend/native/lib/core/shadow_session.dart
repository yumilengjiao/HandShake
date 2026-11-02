import 'wire_values.dart';

class ShadowCandidate {
  const ShadowCandidate(
    this.id,
    this.action,
    this.frameId,
    this.sourceNs,
    this.epoch,
  );
  final String id;
  final String action;
  final BigInt frameId;
  final BigInt sourceNs;
  final BigInt epoch;
}

/// Control-plane state only. It never grants application execution authority.
class ShadowSession {
  ShadowSession._(
    this.epoch,
    this.sequence,
    this.revision,
    this.streamId,
    this.modelVersion,
    this.state,
    this.reason,
    this.warming,
    this.busy,
    this.surfaceReady,
    this.eligibleOwner,
    this.owner,
    this.ownerPending,
    this.candidatesTotal,
    this.candidate,
    this.synthetic,
    this.controlStatus,
  );
  final BigInt epoch;
  final BigInt sequence;
  final BigInt revision;
  final String streamId;
  final String modelVersion;
  final String state;
  final String reason;
  final bool warming;
  final bool busy;
  final bool surfaceReady;
  final String? eligibleOwner;
  final String? owner;
  final bool ownerPending;
  final BigInt candidatesTotal;
  final ShadowCandidate? candidate;
  final bool synthetic;
  final String controlStatus;

  static const fields = {
    'execution',
    'actions_enabled',
    'actual_page',
    'model_epoch',
    'model_sequence',
    'model_revision',
    'model_stream_id',
    'model_version',
    'state',
    'reason',
    'model_warming',
    'model_busy',
    'surface_ready',
    'eligible_owner',
    'owner_track_id',
    'owner_request_pending',
    'candidates_total',
    'candidate',
    'synthetic',
    'control_status',
  };

  factory ShadowSession.parse(Object? raw) {
    if (raw is! Map ||
        raw.length != fields.length ||
        !fields.containsAll(raw.keys) ||
        raw['execution'] != 'shadow' ||
        raw['actions_enabled'] != false ||
        raw['actual_page'] is! int ||
        raw['actual_page'] != 1) {
      throw const FormatException(
        'Unknown shadow fields or forged execution authority',
      );
    }
    String text(String key, int max) {
      final v = raw[key];
      if (v is! String || v.isEmpty || v.length > max) {
        throw const FormatException('Invalid shadow text');
      }
      return v;
    }

    for (final key in [
      'model_warming',
      'model_busy',
      'surface_ready',
      'owner_request_pending',
      'synthetic',
    ]) {
      if (raw[key] is! bool) {
        throw const FormatException('Invalid shadow boolean');
      }
    }
    final state = text('state', 32);
    final control = text('control_status', 32);
    if (!{
          'DISABLED',
          'WAIT_OWNER',
          'WAIT_RELEASE',
          'ACTIVE',
          'CANDIDATE',
          'SUSPENDED',
        }.contains(state) ||
        !{'none', 'accepted', 'rejected'}.contains(control)) {
      throw const FormatException('Unknown shadow state');
    }
    final epoch = wireCounter(raw['model_epoch']);
    final revision = wireCounter(raw['model_revision']);
    final count = wireCounter(raw['candidates_total']);
    if (revision != BigInt.zero) {
      throw const FormatException('Shadow cannot change application revision');
    }
    final offer = raw['eligible_owner'] == null
        ? null
        : text('eligible_owner', 128);
    final owner = raw['owner_track_id'] == null
        ? null
        : text('owner_track_id', 128);
    final candidate = raw['candidate'];
    ShadowCandidate? event;
    if (candidate != null) {
      if (candidate is! Map ||
          candidate.length != 5 ||
          !candidate.keys.toSet().containsAll({
            'event_id',
            'action',
            'source_frame_id',
            'source_timestamp_ns',
            'model_epoch',
          }) ||
          candidate['event_id'] is! String ||
          (candidate['event_id'] as String).isEmpty ||
          (candidate['event_id'] as String).length > 256 ||
          !{'next_slide', 'previous_slide'}.contains(candidate['action'])) {
        throw const FormatException('Invalid shadow candidate');
      }
      event = ShadowCandidate(
        candidate['event_id'] as String,
        candidate['action'] as String,
        wireCounter(candidate['source_frame_id']),
        wireCounter(candidate['source_timestamp_ns']),
        wireCounter(candidate['model_epoch']),
      );
      if (event.epoch != epoch || count == BigInt.zero) {
        throw const FormatException('Old shadow candidate');
      }
    }
    if ((state == 'DISABLED' || state == 'SUSPENDED') &&
        (offer != null ||
            owner != null ||
            event != null ||
            raw['owner_request_pending'] == true ||
            raw['model_warming'] == true)) {
      throw const FormatException(
        'Inactive shadow retained authority/evidence',
      );
    }
    if ({'WAIT_RELEASE', 'ACTIVE', 'CANDIDATE'}.contains(state) &&
            owner == null ||
        state == 'WAIT_OWNER' && owner != null ||
        raw['owner_request_pending'] == true && state != 'WAIT_OWNER') {
      throw const FormatException('Shadow phase/owner mismatch');
    }
    if (offer != null &&
            (state != 'WAIT_OWNER' ||
                owner != null ||
                raw['owner_request_pending'] == true) ||
        raw['model_warming'] == true &&
            (owner == null || state != 'WAIT_RELEASE')) {
      throw const FormatException('Inconsistent shadow owner/handover');
    }
    return ShadowSession._(
      epoch,
      wireCounter(raw['model_sequence']),
      revision,
      text('model_stream_id', 128),
      text('model_version', 80),
      state,
      text('reason', 256),
      raw['model_warming'] as bool,
      raw['model_busy'] as bool,
      raw['surface_ready'] as bool,
      offer,
      owner,
      raw['owner_request_pending'] as bool,
      count,
      event,
      raw['synthetic'] as bool,
      control,
    );
  }
}
