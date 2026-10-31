import 'dart:async';

import 'package:flutter/foundation.dart';

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
      latest?.isPreview == false;

  Future<void> connect() async {
    if (_disposed || link != LinkState.disconnected) return;
    final generation = ++_generation;
    link = LinkState.connecting;
    latest = inspected = null;
    _history.clear();
    _commandId = BigInt.zero;
    _pendingStart = _pendingStop = null;
    message = '正在连接本机核心…';
    notifyListeners();
    final repository = _factory();
    _repository = repository;
    _subscription = repository.snapshots.listen(
      (value) {
        if (_disposed || generation != _generation) return;
        _receive(value);
      },
      onError: (Object _) {
        if (!_disposed && generation == _generation) _disconnect();
      },
      onDone: () {
        if (!_disposed && generation == _generation) _disconnect();
      },
    );
    _armDeadline();
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
    if (value.lastCommandId > _commandId) {
      _disconnect();
      return;
    }
    if (previous != null && value.epoch != previous.epoch) {
      // Never carry a review frame or a trail across a source generation.
      _history.clear();
      inspected = null;
    }
    latest = value;
    link = LinkState.connected;
    _armDeadline();
    if (_pendingStop != null &&
        value.lastCommandId >= _pendingStop! &&
        !value.isPreview) {
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
      'user_stop' => '核心已确认停止 · 无应用控制能力',
      'heartbeat_timeout' => '心跳超时，核心已停止；需要手动重新开始',
      'stale_epoch' => '已拒绝旧会话请求；请重新开始',
      _ => '核心停止或协议异常；请检查连接',
    };
    if (stopPending) message = '已请求停止，等待核心确认…';
    notifyListeners();
  }

  void _armDeadline() {
    _deadline?.cancel();
    _deadline = Timer(const Duration(seconds: 2), _disconnect);
  }

  BigInt? _send(String kind) {
    final snapshot = latest;
    if (snapshot == null || link != LinkState.connected) return null;
    // Bound the sink even if the child stops reading while its output continues.
    if (_commandId - snapshot.lastCommandId >= BigInt.from(8)) {
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
    if (!canStart) return;
    _pendingStart = _send('start');
    if (_pendingStart != null) _armIntentDeadline();
    notifyListeners();
  }

  void stop() {
    inspected = null;
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

  void _disconnect() {
    if (_disposed || link == LinkState.disconnected) return;
    ++_generation;
    _heartbeat?.cancel();
    _deadline?.cancel();
    _intentDeadline?.cancel();
    link = LinkState.disconnected;
    latest = inspected = null;
    _history.clear();
    _pendingStart = _pendingStop = null;
    message = '核心失联 · 停止未确认；仅合成原型，无应用控制。请重新连接。';
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
