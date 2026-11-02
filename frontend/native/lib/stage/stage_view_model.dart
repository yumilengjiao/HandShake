import 'dart:async';

import 'package:flutter/foundation.dart';

import '../core/core_diagnostics.dart';
import '../core/core_repository.dart';
import '../core/core_snapshot.dart';

enum LinkState { disconnected, connecting, connected }

/// View state only: no gesture inference, no action dispatch, no copied safety FSM.
class StageViewModel extends ChangeNotifier {
  StageViewModel(this._factory);
  final CoreRepository Function() _factory;
  CoreRepository? _repository;
  StreamSubscription<CoreSnapshot>? _subscription;
  Timer? _heartbeat;
  Timer? _deadline;
  Timer? _intentDeadline;
  int _generation = 0;
  bool _disposed = false;
  BigInt _commandId = BigInt.zero;
  BigInt? _pendingStop;
  BigInt? _pendingStart;
  BigInt? _presentedSequence;
  CoreSnapshot? _presented;

  LinkState link = LinkState.disconnected;
  CoreSnapshot? latest;
  CoreSnapshot? inspected;
  final List<CoreSnapshot> _history = [];
  List<CoreSnapshot> get history => List.unmodifiable(_history);
  CoreSnapshot? get displayed => inspected ?? latest;
  bool get reviewing => inspected != null;
  bool mirror = false;
  bool explain = false;
  String message = '连接本机核心以开始；不会打开相机或控制应用。';
  bool get stopPending => _pendingStop != null;
  bool get canStart =>
      link == LinkState.connected &&
      !reviewing &&
      !stopPending &&
      _pendingStart == null &&
      latest?.state == 'stopped';
  bool get canStartCamera =>
      canStart &&
      (latest?.version == 2 ||
          latest?.version == 3 ||
          latest?.version == 4 ||
          latest?.version == 5);
  bool get canEnableShadow =>
      link == LinkState.connected &&
      !reviewing &&
      !stopPending &&
      _pendingStart == null &&
      latest?.isPreview == true &&
      latest?.shadow?.state == 'DISABLED' &&
      latest?.shadow?.surfaceReady == true &&
      latest?.shadow?.busy == false;
  bool get _presentedOwnerIsFresh {
    final shown = _presented;
    final current = latest;
    if (shown == null ||
        current == null ||
        shown.shadow?.eligibleOwner == null ||
        shown.sessionId != current.sessionId ||
        shown.epoch != current.epoch ||
        shown.shadow?.epoch != current.shadow?.epoch ||
        shown.shadow?.eligibleOwner != current.shadow?.eligibleOwner ||
        shown.frameTimestampNs == null) {
      return false;
    }
    final age = current.timestampNs - shown.frameTimestampNs!;
    return age >= BigInt.zero && age <= BigInt.from(150000000);
  }

  bool get canRequestOwner =>
      link == LinkState.connected &&
      !reviewing &&
      !stopPending &&
      _pendingStart == null &&
      latest?.isPreview == true &&
      latest?.shadow?.eligibleOwner != null &&
      _presentedOwnerIsFresh &&
      latest?.shadow?.ownerPending == false;

  Future<void> connect() async {
    if (_disposed || link != LinkState.disconnected) return;
    final generation = ++_generation;
    link = LinkState.connecting;
    latest = inspected = null;
    _history.clear();
    _commandId = BigInt.zero;
    _pendingStart = _pendingStop = null;
    _presentedSequence = null;
    _presented = null;
    message = '正在连接本机核心…';
    notifyListeners();
    final repository = _factory();
    _repository = repository;
    _subscription = repository.snapshots.listen(
      (value) {
        if (_disposed || generation != _generation) return;
        _receive(value);
      },
      onError: (Object error) {
        if (!_disposed && generation == _generation) {
          _disconnect(
            detail: error is CoreProcessFailure ? error.toString() : null,
          );
        }
      },
      onDone: () {
        if (!_disposed && generation == _generation) _disconnect();
      },
    );
    // A cold Python process may need time to import/validate the protocol. No
    // source or authority exists yet; connected liveness/stop deadlines stay 2s.
    _armDeadline(timeout: const Duration(seconds: 10));
    try {
      await repository.connect();
      if (_disposed || generation != _generation) return;
      _heartbeat = Timer.periodic(const Duration(milliseconds: 400), (_) {
        if (link == LinkState.connected) _send('heartbeat');
      });
    } on Object {
      if (!_disposed && generation == _generation) _disconnect();
    }
  }

  void _receive(CoreSnapshot value) {
    final previous = latest;
    if (previous != null && value.sessionId != previous.sessionId) {
      _disconnect();
      return;
    }
    if (previous != null &&
        (value.sequence <= previous.sequence ||
            value.epoch < previous.epoch ||
            value.timestampNs < previous.timestampNs ||
            value.lastCommandId < previous.lastCommandId)) {
      return;
    }
    if (previous != null &&
        value.epoch == previous.epoch &&
        previous.shadow != null &&
        value.shadow != null) {
      if (value.shadow!.epoch < previous.shadow!.epoch ||
          value.shadow!.sequence < previous.shadow!.sequence) {
        return;
      }
      if (value.shadow!.streamId != previous.shadow!.streamId ||
          value.shadow!.modelVersion != previous.shadow!.modelVersion) {
        _disconnect();
        return;
      }
    }
    if (value.lastCommandId > _commandId) {
      _disconnect();
      return;
    }
    if (previous != null &&
        (value.epoch != previous.epoch ||
            value.shadow?.epoch != previous.shadow?.epoch)) {
      // Never carry a review frame or a trail across a source generation.
      _history.clear();
      inspected = null;
      _presented = null;
    }
    latest = value;
    link = LinkState.connected;
    _armDeadline();
    if (_pendingStop != null &&
        value.lastCommandId >= _pendingStop! &&
        value.state == 'stopped') {
      _pendingStop = null;
    }
    if (_pendingStart != null && value.lastCommandId >= _pendingStart!) {
      _pendingStart = null;
    }
    if (_pendingStart == null && _pendingStop == null) {
      _intentDeadline?.cancel();
    }
    if (value.isPreview &&
        (_history.isEmpty || _history.last.frameId != value.frameId)) {
      _history.add(value);
      if (_history.length > 32) _history.removeAt(0);
    }
    message = switch (value.reason) {
      'initial' => '核心已连接 · 尚未启动',
      'synthetic_preview' => '合成骨架预览 · 无相机、无模型、无应用命令',
      'camera_starting' => '正在打开相机 / 感知资产；不保存画面，不控制应用',
      'camera_preview' => '本机相机预览 · 仅内存处理，不保存画面，不发送应用命令',
      'camera_unavailable' => '相机不可用；检查设备编号、占用和权限后手动重试',
      'perception_unavailable' => '感知资产或SDK不可用；检查模型路径与perception依赖',
      'input_stalled' => '输入停滞或帧已过期；停止后可手动重试',
      'preview_error' => '预览处理/释放失败；停止未确认时请断开核心',
      'user_stop' => '核心已确认停止 · 无应用控制能力',
      'heartbeat_timeout' => '心跳超时，核心已停止；需要手动重新开始',
      'stale_epoch' => '已拒绝旧会话请求；请重新开始',
      _ => '核心停止或协议异常；请检查连接',
    };
    if (value.shadow != null) {
      final shadow = value.shadow!;
      message = shadow.ownerPending
          ? '已请求影子owner，等待下一条新模型结果；不执行命令'
          : shadow.warming
          ? 'owner已绑定，正在重新预热；无动作权限'
          : shadow.controlStatus == 'rejected'
          ? '影子请求已拒绝：状态已变化或证据不再有效，请查看当前状态'
          : '影子会话 ${shadow.state} · 只记录候选，不翻页、不控制外部应用';
    }
    if (value.modelStatus == 'error') {
      message = '手势模型包校验或加载失败；检查外部SHA、资产与依赖，未执行任何动作。';
    }
    if (value.state == 'stopping') message = '等待相机/感知线程释放，尚未确认停止…';
    if (stopPending) message = '已请求停止，等待核心确认…';
    notifyListeners();
  }

  void _armDeadline({Duration timeout = const Duration(seconds: 2)}) {
    _deadline?.cancel();
    _deadline = Timer(timeout, _disconnect);
  }

  BigInt? _send(String kind, {CoreSnapshot? basedOn}) {
    final snapshot = basedOn ?? latest;
    if (snapshot == null || link != LinkState.connected) return null;
    // Bound the sink even if the child stops reading while its output continues.
    if (_commandId - (latest?.lastCommandId ?? snapshot.lastCommandId) >=
        BigInt.from(8)) {
      _disconnect();
      return null;
    }
    final id = _commandId + BigInt.one;
    _commandId = id;
    try {
      _repository!.send(kind, snapshot, id);
      return id;
    } on Object {
      _disconnect();
      return null;
    }
  }

  void start() {
    if (!canStart || latest?.version == 5) return;
    _pendingStart = _send('start');
    if (_pendingStart != null) _armIntentDeadline();
    notifyListeners();
  }

  void startCamera() {
    if (!canStartCamera) return;
    _pendingStart = _send('start_camera');
    if (_pendingStart != null) _armIntentDeadline();
    notifyListeners();
  }

  void enableShadow() {
    if (!canEnableShadow) return;
    _pendingStart = _send('enable_shadow');
    if (_pendingStart != null) _armIntentDeadline();
    notifyListeners();
  }

  void requestOwner() {
    if (!canRequestOwner) return;
    _pendingStart = _send('request_owner', basedOn: _presented);
    if (_pendingStart != null) _armIntentDeadline();
    notifyListeners();
  }

  void surfacePresented(CoreSnapshot value) {
    final current = latest;
    if (_disposed ||
        reviewing ||
        stopPending ||
        value.shadow == null ||
        current == null ||
        current.sessionId != value.sessionId ||
        current.epoch != value.epoch ||
        current.shadow?.epoch != value.shadow?.epoch ||
        current.sequence < value.sequence ||
        current.timestampNs < value.timestampNs ||
        current.timestampNs - value.timestampNs >= BigInt.from(500000000) ||
        (_presentedSequence != null && _presentedSequence! >= value.sequence)) {
      return;
    }
    // Decode/paint may finish after another telemetry packet arrived. A captured,
    // still-current model token can be acknowledged; never relabel it as latest.
    _presentedSequence = value.sequence;
    _presented = value;
    _send('surface_ack', basedOn: value);
    notifyListeners();
  }

  void stop() {
    inspected = null;
    _presented = null;
    if (!stopPending) {
      _pendingStop = _send('stop');
      if (_pendingStop != null) _armIntentDeadline();
    }
    if (stopPending) message = '已请求停止，等待核心确认…';
    notifyListeners();
  }

  void _armIntentDeadline() {
    _intentDeadline?.cancel();
    _intentDeadline = Timer(const Duration(seconds: 2), _disconnect);
  }

  void inspect(int index) {
    if (index < 0 || index >= _history.length) return;
    inspected = _history[index];
    notifyListeners();
  }

  void returnToNow() {
    inspected = null;
    notifyListeners();
  }

  void toggleMirror() {
    mirror = !mirror;
    notifyListeners();
  }

  void toggleExplain() {
    explain = !explain;
    notifyListeners();
  }

  void _disconnect({String? detail}) {
    if (_disposed || link == LinkState.disconnected) return;
    ++_generation;
    _heartbeat?.cancel();
    _deadline?.cancel();
    _intentDeadline?.cancel();
    link = LinkState.disconnected;
    latest = inspected = null;
    _history.clear();
    _pendingStart = _pendingStop = null;
    message = '${detail ?? "核心失联"} · 停止未确认，无应用控制。检查Python路径/依赖后重新连接。';
    unawaited(_subscription?.cancel());
    unawaited(_repository?.close());
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
    unawaited(_repository?.close());
    super.dispose();
  }
}
