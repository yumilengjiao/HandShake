import 'dart:async';

import 'package:flutter/foundation.dart';

import 'gesture_repository.dart';
import 'gesture_snapshot.dart';

enum GestureLink { disconnected, connecting, connected }

class GestureViewModel extends ChangeNotifier {
  GestureViewModel(this._factory);
  final GestureRepository Function() _factory;
  GestureRepository? _repository;
  StreamSubscription<GestureSnapshot>? _subscription;
  Future<void>? _closing;
  Timer? _heartbeat, _deadline, _intentDeadline;
  bool _disposed = false;
  bool active = false;
  int _generation = 0;
  BigInt _commandId = BigInt.zero;
  BigInt? _pending;
  String? _pendingKind;
  GestureSnapshot? _painted;
  GestureSnapshot? latest;
  GestureLink link = GestureLink.disconnected;
  String message = '实时手势翻页 · 摄像头与控制默认关闭';

  bool get pending => _pending != null;
  bool get canOpen =>
      active &&
      link == GestureLink.connected &&
      !pending &&
      latest?.flag('model_available') == true &&
      latest?.flag('camera_released') == true;
  bool get _freshPaint {
    final value = _painted, current = latest;
    return value != null &&
        current != null &&
        value.cameraEpoch == current.cameraEpoch &&
        value.modelEpoch == current.modelEpoch &&
        value.revision == current.revision &&
        current.timestampNs >= value.timestampNs &&
        current.timestampNs - value.timestampNs < BigInt.from(500000000);
  }

  bool get canEnable =>
      active &&
      link == GestureLink.connected &&
      !pending &&
      _freshPaint &&
      latest?.cameraState == 'running' &&
      latest?.flag('enabled') == false &&
      latest?.flag('busy') == false &&
      latest?.flag('control_released') == true;
  bool get canChoose =>
      active &&
      link == GestureLink.connected &&
      !pending &&
      _freshPaint &&
      latest?.flag('enabled') == true &&
      latest?.flag('owner_bound') == false &&
      latest?.flag('eligible_owner') == true &&
      _painted?.flag('eligible_owner') == true &&
      latest?.flag('owner_pending') == false;

  Future<void> connect() async {
    if (_disposed || link != GestureLink.disconnected) return;
    final generation = ++_generation;
    link = GestureLink.connecting;
    latest = _painted = null;
    _commandId = BigInt.zero;
    _pending = null;
    _pendingKind = null;
    message = '连接手势翻页核心…';
    notifyListeners();
    try {
      await _closing;
      if (_disposed || generation != _generation) return;
      final repo = _factory();
      _repository = repo;
      _subscription = repo.snapshots.listen(
        (value) {
          if (!_disposed && generation == _generation) _receive(value);
        },
        onError: (Object _) {
          if (!_disposed && generation == _generation) disconnect();
        },
        onDone: () {
          if (!_disposed && generation == _generation) disconnect();
        },
      );
      _armDeadline(const Duration(seconds: 10));
      await repo.connect();
      if (_disposed || generation != _generation) return;
      _heartbeat = Timer.periodic(
        const Duration(milliseconds: 400),
        (_) => _send('heartbeat'),
      );
    } on Object {
      if (!_disposed && generation == _generation) disconnect();
    }
  }

  void activate() {
    active = true;
    connect();
    notifyListeners();
  }

  void deactivate() {
    active = false;
    closeCamera();
  }

  void _receive(GestureSnapshot value) {
    final old = latest;
    if (value.lastCommandId > _commandId ||
        old != null && value.sessionId != old.sessionId) {
      disconnect();
      return;
    }
    if (old != null &&
        (value.sequence <= old.sequence ||
            value.cameraEpoch < old.cameraEpoch ||
            value.timestampNs < old.timestampNs ||
            value.lastCommandId < old.lastCommandId)) {
      return;
    }
    if (old != null &&
        value.cameraEpoch == old.cameraEpoch &&
        (value.revision < old.revision ||
            value.revision == old.revision && value.page != old.page ||
            old.modelEpoch != null &&
                value.modelEpoch != null &&
                value.modelEpoch! < old.modelEpoch!)) {
      disconnect();
      return;
    }
    if (old?.cameraEpoch != value.cameraEpoch ||
        old?.modelEpoch != value.modelEpoch) {
      _painted = null;
    }
    latest = value;
    link = GestureLink.connected;
    _armDeadline(const Duration(seconds: 2));
    if (_pending != null &&
        value.lastCommandId >= _pending! &&
        (_pendingKind != 'close_camera' || value.flag('camera_released')) &&
        (_pendingKind != 'stop' || value.flag('control_released'))) {
      _pending = null;
      _pendingKind = null;
      _intentDeadline?.cancel();
    }
    message = _status(value);
    if (_pendingKind == 'stop') message = '停止已请求，等待控制释放确认…';
    if (_pendingKind == 'close_camera') message = '正在关闭相机，等待设备释放确认…';
    notifyListeners();
  }

  String _status(GestureSnapshot v) {
    final reason = v.text('reason');
    if (!v.flag('model_available')) return '未找到完整模型与校准。请先训练、校准、导出，或指定可信模型包。';
    if (reason == 'release_unconfirmed') return '设备释放未确认，不允许再次打开相机。';
    if (v.cameraState == 'stopping') return '正在释放摄像头与感知资源…';
    if (v.cameraState == 'starting') return '加载模型并打开相机，请稍候…';
    if (v.cameraState == 'stopped') {
      return switch (reason) {
        'initial' || 'camera_closed' || 'shutdown' => '相机已关闭 · 点击“打开相机”开始预览',
        _ => '相机已停止：$reason。检查依赖、设备占用与权限后再试。',
      };
    }
    if (reason == 'enable_rejected') return '启用未成功：模型忙或画面已变化，请稍候再点击启用。';
    if (reason == 'owner_rejected') return '选手未成功：只留一只自然静止的手，等待按钮重新可用。';
    if (!v.flag('enabled')) return '相机预览中 · 点击“启用手势”才开始识别控制';
    final policyReason = v.text('policy_reason');
    if (policyReason.startsWith('perception_not_processed:') ||
        policyReason == 'not_processed') {
      return switch (policyReason) {
        'perception_not_processed:model_not_loaded' =>
          'O 未就绪：手部感知模型未加载，不是手型或等待时间的问题。请关闭相机后检查模型与依赖。',
        'perception_not_processed:same_mediapipe_millisecond' ||
        'perception_not_processed:duplicate_or_backwards_source_time' =>
          'O 未就绪：感知收到重复或间隔不足的帧时间戳，请关闭相机后重新打开。',
        _ => 'O 未就绪：相机帧未进入感知处理，不是手型问题；具体原因：$policyReason',
      };
    }
    if (v.flag('owner_pending')) return '已请求控制手，等待下一帧确认…';
    if (v.flag('warming')) return '正在预热32帧窗口，请保持手部稳定';
    if (!v.flag('owner_bound')) {
      return v.flag('eligible_owner')
          ? '中立手已就绪 · 点击“选择控制手”或按 O'
          : switch (v.text('policy_reason')) {
              'owner_requires_neutral_pose' =>
                'O 未就绪：掌心居中，食指、中指伸直，其余两指弯曲；不要完全张掌或握拳。',
              'owner_requires_nog' =>
                'O 未就绪：模型仍识别为 ${v.topClass ?? '动作'}，请保持中立手型静止，等待 NOG。',
              'owner_requires_confident_nog' =>
                'O 未就绪：中立类别 NOG 置信度不足；保持手部清晰、光照充足。',
              'no_hand' => 'O 未就绪：未检测到手，请将一只手完整放入画面。',
              'multiple_hands' ||
              'owner_requires_unique_visible_hand' => 'O 未就绪：只留一只手，避免遮挡和交叉。',
              'warming_up' ||
              'history_uninitialized' => 'O 未就绪：正在积累32帧有效历史，请保持一只手稳定。',
              'stale' ||
              'window_span_exceeded' => 'O 未就绪：识别帧过期或处理过慢，请检查相机帧率和系统负载。',
              'current_observation_invalid' ||
              'owner_requires_valid_hand' => 'O 未就绪：手部关键点质量不足，请完整入镜、避免遮挡。',
              _ => '请只留一只中立手静止，等待判定；当前原因：${v.text('policy_reason')}',
            };
    }
    return '控制手已绑定 · 右抛下一页，左抛上一页；动作后回到中立';
  }

  void _armDeadline(Duration duration) {
    _deadline?.cancel();
    _deadline = Timer(duration, disconnect);
  }

  BigInt? _send(String kind, {GestureSnapshot? basedOn}) {
    final value = basedOn ?? latest;
    if (_disposed || link != GestureLink.connected || value == null) {
      return null;
    }
    if (_commandId - latest!.lastCommandId >= BigInt.from(8)) {
      disconnect();
      return null;
    }
    final id = _commandId + BigInt.one;
    _commandId = id;
    try {
      _repository!.send(kind, value, id);
      return id;
    } on Object {
      disconnect();
      return null;
    }
  }

  void surfacePainted(GestureSnapshot value) {
    final current = latest;
    if (_disposed ||
        !active ||
        current == null ||
        _pendingKind == 'stop' ||
        _pendingKind == 'close_camera' ||
        value.sessionId != current.sessionId ||
        value.cameraEpoch != current.cameraEpoch ||
        value.modelEpoch != current.modelEpoch ||
        value.revision != current.revision ||
        current.timestampNs < value.timestampNs ||
        current.timestampNs - value.timestampNs >= BigInt.from(500000000) ||
        _painted != null && value.sequence <= _painted!.sequence) {
      return;
    }
    final couldEnable = canEnable, couldChoose = canChoose;
    _painted = value;
    if (value.modelEpoch != null) _send('surface_ack', basedOn: value);
    // A first eligible/epoch-changing packet was built before its paint receipt.
    // Refresh the buttons once if their readiness changed, not on every ACK.
    if (couldEnable != canEnable || couldChoose != canChoose) {
      notifyListeners();
    }
  }

  void _intent(String kind, {bool painted = false}) {
    _pending = _send(kind, basedOn: painted ? _painted : null);
    _pendingKind = _pending == null ? null : kind;
    if (_pending != null) {
      _intentDeadline?.cancel();
      _intentDeadline = Timer(const Duration(seconds: 3), disconnect);
    }
    notifyListeners();
  }

  void openCamera() {
    if (canOpen) _intent('open_camera');
  }

  void enable() {
    if (canEnable) _intent('enable', painted: true);
  }

  void chooseOwner() {
    if (canChoose) _intent('choose_owner', painted: true);
  }

  void stop() {
    _painted = null;
    if (_pendingKind != 'stop' && _pendingKind != 'close_camera') {
      _intent('stop');
    }
  }

  void closeCamera() {
    _painted = null;
    if (_pendingKind != 'close_camera') _intent('close_camera');
  }

  Future<bool> closeCameraAndWait() async {
    if (latest?.flag('camera_released') == true ||
        link == GestureLink.disconnected && _repository == null) {
      return true;
    }
    final completed = Completer<bool>();
    void check() {
      if (completed.isCompleted) return;
      if (latest?.flag('camera_released') == true && !pending) {
        completed.complete(true);
      } else if (link == GestureLink.disconnected) {
        completed.complete(false);
      }
    }

    addListener(check);
    closeCamera();
    check();
    try {
      return await completed.future.timeout(
        const Duration(seconds: 3),
        onTimeout: () => false,
      );
    } finally {
      removeListener(check);
    }
  }

  void disconnect() {
    if (_disposed || link == GestureLink.disconnected) return;
    ++_generation;
    _heartbeat?.cancel();
    _deadline?.cancel();
    _intentDeadline?.cancel();
    link = GestureLink.disconnected;
    latest = _painted = null;
    _pending = null;
    _pendingKind = null;
    message = '手势核心失联 · 停止未确认；不会自动重开相机或恢复控制。';
    unawaited(_subscription?.cancel());
    _closing = _repository?.close();
    // Retain the failed future so reconnect cannot bypass unconfirmed process exit.
    unawaited(_closing?.catchError((Object _) {}));
    notifyListeners();
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    ++_generation;
    _heartbeat?.cancel();
    _deadline?.cancel();
    _intentDeadline?.cancel();
    unawaited(_subscription?.cancel());
    unawaited(_repository?.close().catchError((Object _) {}));
    super.dispose();
  }
}
