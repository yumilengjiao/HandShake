import 'dart:async';

import 'package:flutter/foundation.dart';

import 'presentation_repository.dart';
import 'presentation_snapshot.dart';

enum PresentationLink { disconnected, connecting, connected }

class PresentationViewModel extends ChangeNotifier {
  PresentationViewModel(this._factory);
  final PresentationRepository Function() _factory;
  PresentationRepository? _repository;
  StreamSubscription<PresentationSnapshot>? _subscription;
  Timer? _heartbeat, _deadline, _commandDeadline;
  int _generation = 0;
  bool _disposed = false;
  bool active = false;
  PresentationLink link = PresentationLink.disconnected;
  PresentationSnapshot? latest;
  BigInt _commandId = BigInt.zero;
  BigInt? _pending;
  String? _pendingKind, _renderKey;
  String message = '内置演示 · 人工操作，不控制其他应用。';

  bool get busy => _pending != null || latest?.busy == true;
  bool get stopPending => _pendingKind == 'stop';
  bool get canPlay =>
      active &&
      link == PresentationLink.connected &&
      !busy &&
      latest?.state == 'stopped' &&
      latest?.released == true &&
      latest?.rendered == true;
  bool get canNavigate =>
      active &&
      link == PresentationLink.connected &&
      !busy &&
      latest?.state == 'playing' &&
      latest?.rendered == true;

  Future<void> connect() async {
    if (_disposed || link != PresentationLink.disconnected) return;
    final generation = ++_generation;
    link = PresentationLink.connecting;
    latest = null;
    _renderKey = null;
    _commandId = BigInt.zero;
    _pending = null;
    _pendingKind = null;
    message = '连接内置演示核心…';
    notifyListeners();
    final repository = _factory();
    _repository = repository;
    _subscription = repository.snapshots.listen(
      (value) {
        if (_disposed || generation != _generation) return;
        _receive(value);
      },
      onError: (Object _) {
        if (!_disposed && generation == _generation) disconnect();
      },
      onDone: () {
        if (!_disposed && generation == _generation) disconnect();
      },
    );
    _armDeadline();
    try {
      await repository.connect();
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
    _renderKey = null;
    connect();
    notifyListeners();
  }

  void deactivate() {
    active = false;
    stop();
  }

  void _receive(PresentationSnapshot value) {
    final old = latest;
    if (old != null &&
        (value.sessionId != old.sessionId ||
            value.deckId != old.deckId ||
            value.pageCount != old.pageCount ||
            (value.revision == old.revision && value.page != old.page))) {
      disconnect();
      return;
    }
    if (old != null &&
        (value.sequence <= old.sequence ||
            value.epoch < old.epoch ||
            value.revision < old.revision ||
            value.timestampNs < old.timestampNs ||
            value.lastCommandId < old.lastCommandId)) {
      return;
    }
    if (value.lastCommandId > _commandId) {
      disconnect();
      return;
    }
    latest = value;
    link = PresentationLink.connected;
    _armDeadline();
    if (value.renderedRevision == null &&
        (old?.renderedRevision != null || value.reason == 'stale_snapshot')) {
      _renderKey = null;
    }
    if (_pending != null &&
        value.lastCommandId >= _pending! &&
        !value.busy &&
        (_pendingKind != 'stop' ||
            (value.state == 'stopped' && value.released))) {
      _pending = null;
      _pendingKind = null;
      _commandDeadline?.cancel();
    }
    message = switch (value.reason) {
      'manual_playing' => '人工授权已启用 · 只作用于本应用',
      'command_submitted' => value.busy ? '调度器处理中…' : '应用状态已更新；绘制确认单独显示',
      'user_stop' =>
        value.released && !value.busy ? '核心已停止 · 无持有按键' : '停止已请求，等待调度与释放确认…',
      'heartbeat_timeout' => '心跳超时，已停用；需重新人工播放',
      'render_timeout' => '界面未确认新页绘制，已停用；不自动重试翻页',
      'stale_snapshot' => '拒绝过期界面请求，已停用；请重新播放',
      'stale_context' => '拒绝旧代际/页码请求',
      'surface_not_ready' => '等待演示页面绘制确认',
      'awaiting_render_or_result' => '等待当前操作完成及新页绘制',
      'not_playing' => '请先人工播放',
      'release_pending' => '释放未确认，暂不能播放',
      'initial' => '已连接 · 点击播放才允许翻页',
      _ => value.state == 'stopped' ? '演示已停用' : '内置演示播放中',
    };
    if (stopPending) message = '停止已请求，等待核心确认…';
    notifyListeners();
  }

  void _armDeadline() {
    _deadline?.cancel();
    _deadline = Timer(const Duration(seconds: 2), disconnect);
  }

  BigInt? _send(String kind) {
    final value = latest;
    if (_disposed || value == null || link != PresentationLink.connected) {
      return null;
    }
    if (_commandId - value.lastCommandId >= BigInt.from(8)) {
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

  void surfacePainted(PresentationSnapshot value) {
    if (_disposed || !active || latest == null) return;
    final current = latest!;
    if (value.epoch != current.epoch ||
        value.revision != current.revision ||
        value.deckId != current.deckId) {
      return;
    }
    final key = '${value.epoch}/${value.revision}';
    if (current.rendered || _renderKey == key) return;
    if (_send('render_ack') != null) _renderKey = key;
  }

  void _intent(String kind) {
    _pending = _send(kind);
    _pendingKind = _pending == null ? null : kind;
    if (_pending != null) {
      _commandDeadline?.cancel();
      _commandDeadline = Timer(const Duration(seconds: 2), disconnect);
    }
    notifyListeners();
  }

  void play() {
    if (canPlay) _intent('play');
  }

  void next() {
    if (canNavigate) _intent('next');
  }

  void previous() {
    if (canNavigate) _intent('previous');
  }

  void stop() {
    if (_disposed) return;
    _renderKey = null;
    if (!stopPending) _intent('stop');
  }

  void disconnect() {
    if (_disposed || link == PresentationLink.disconnected) return;
    ++_generation;
    _heartbeat?.cancel();
    _deadline?.cancel();
    _commandDeadline?.cancel();
    link = PresentationLink.disconnected;
    latest = null;
    _pending = null;
    _pendingKind = null;
    _renderKey = null;
    message = '演示核心失联 · 停止未确认，不补发旧操作。';
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
    _commandDeadline?.cancel();
    unawaited(_subscription?.cancel());
    unawaited(_repository?.close());
    super.dispose();
  }
}
