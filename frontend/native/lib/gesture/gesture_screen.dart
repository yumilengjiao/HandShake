import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'gesture_snapshot.dart';
import 'gesture_view_model.dart';

class GestureScreen extends StatelessWidget {
  const GestureScreen({super.key, required this.model});
  final GestureViewModel model;

  @override
  Widget build(BuildContext context) => CallbackShortcuts(
    bindings: {
      for (final shift in [false, true]) ...{
        SingleActivator(
          LogicalKeyboardKey.keyC,
          shift: shift,
          includeRepeats: false,
        ): model.openCamera,
        SingleActivator(
          LogicalKeyboardKey.keyE,
          shift: shift,
          includeRepeats: false,
        ): model.enable,
        SingleActivator(
          LogicalKeyboardKey.keyO,
          shift: shift,
          includeRepeats: false,
        ): model.chooseOwner,
      },
      const SingleActivator(LogicalKeyboardKey.space, includeRepeats: false):
          model.stop,
      const SingleActivator(LogicalKeyboardKey.escape, includeRepeats: false):
          model.closeCamera,
    },
    child: Focus(
      autofocus: true,
      child: ListenableBuilder(
        listenable: model,
        builder: (context, _) {
          final value = model.latest;
          return Column(
            children: [
              Expanded(
                child: value == null
                    ? Center(child: Text(model.message))
                    : LayoutBuilder(
                        builder: (context, constraints) {
                          return SingleChildScrollView(
                            padding: const EdgeInsets.all(16),
                            child: constraints.maxWidth >= 900
                                ? Row(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Expanded(
                                        flex: 3,
                                        child: _Slide(
                                          value: value,
                                          model: model,
                                        ),
                                      ),
                                      const SizedBox(width: 16),
                                      Expanded(
                                        flex: 2,
                                        child: _Camera(value: value),
                                      ),
                                    ],
                                  )
                                : Column(
                                    children: [
                                      _Slide(value: value, model: model),
                                      const SizedBox(height: 16),
                                      _Camera(value: value),
                                    ],
                                  ),
                          );
                        },
                      ),
              ),
              const Divider(height: 1),
              Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text(
                      model.message,
                      key: const Key('gesture-status'),
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          child: FilledButton.icon(
                            key: const Key('gesture-stop'),
                            onPressed: model.stop,
                            icon: const Icon(Icons.stop),
                            label: const Text('停止控制 · 空格'),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: OutlinedButton(
                            key: const Key('gesture-close'),
                            onPressed: model.closeCamera,
                            child: const Text('关闭相机 · Esc'),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        spacing: 8,
                        children: [
                          if (model.link == GestureLink.disconnected)
                            OutlinedButton(
                              onPressed: model.connect,
                              child: const Text('重新连接'),
                            ),
                          OutlinedButton.icon(
                            key: const Key('gesture-camera'),
                            onPressed: model.canOpen ? model.openCamera : null,
                            icon: const Icon(Icons.videocam_outlined),
                            label: const Text('1 打开相机 · C'),
                          ),
                          OutlinedButton(
                            key: const Key('gesture-enable'),
                            onPressed: model.canEnable ? model.enable : null,
                            child: const Text('2 启用手势 · E'),
                          ),
                          OutlinedButton(
                            key: const Key('gesture-owner'),
                            onPressed: model.canChoose
                                ? model.chooseOwner
                                : null,
                            child: const Text('3 选择控制手 · O'),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      '只翻动本应用示例页，不控制其他软件。实验性识别可能漏触发；相机画面仅在内存中处理。',
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(fontSize: 11),
                    ),
                  ],
                ),
              ),
            ],
          );
        },
      ),
    ),
  );
}

class _Slide extends StatelessWidget {
  const _Slide({required this.value, required this.model});
  final GestureSnapshot value;
  final GestureViewModel model;
  @override
  Widget build(BuildContext context) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (context.mounted) model.surfacePainted(value);
    });
    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(minHeight: 340),
      padding: const EdgeInsets.all(24),
      color: const Color(0xfff3eee3),
      child: DefaultTextStyle(
        style: const TextStyle(
          color: Color(0xff173139),
          fontSize: 18,
          height: 1.5,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '${value.page} / ${value.pageCount}',
              key: const Key('gesture-page'),
            ),
            const SizedBox(height: 20),
            Text(
              value.text('title'),
              style: const TextStyle(fontSize: 28, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 20),
            for (final line in value.lines)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Text(line),
              ),
            const SizedBox(height: 16),
            Text(
              '实际页码 · 修订 ${value.revision} · ${value.flag('rendered') ? '页面已绘制' : '等待绘制确认'}',
              style: const TextStyle(fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }
}

class _Camera extends StatelessWidget {
  const _Camera({required this.value});
  final GestureSnapshot value;
  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      AspectRatio(
        aspectRatio: 4 / 3,
        child: Container(
          color: Colors.black,
          child: value.cameraState == 'running'
              ? _CameraImage(
                  key: ValueKey(value.cameraEpoch),
                  bytes: value.jpeg,
                )
              : const Center(child: Text('相机未打开')),
        ),
      ),
      const SizedBox(height: 12),
      Text(
        '识别类别：${value.topClass ?? '等待有效模型结果'}',
        key: const Key('gesture-class'),
      ),
      Text(
        '控制手：${value.flag('owner_bound') ? '已绑定' : '未选择'} · ${value.flag('warming') ? '预热中' : value.text('model_state')}',
      ),
      Text('翻页权限：${value.flag('actions_enabled') ? '可提交符合条件的手势' : '尚未就绪'}'),
      const SizedBox(height: 8),
      SelectableText(
        '判定原因：${value.text('policy_reason')}',
        style: const TextStyle(fontSize: 12),
      ),
      const SizedBox(height: 12),
      const Text(
        '选手时只留一只手，掌心放在画面中央：食指、中指伸直，无名指、小指弯曲，静止保持。完全张掌或握拳不符合当前中立条件。选手后继续静止预热，再做左右抛动；动作后回到中立。',
      ),
    ],
  );
}

/// One decoder + one latest frame; no global image cache or camera history.
class _CameraImage extends StatefulWidget {
  const _CameraImage({super.key, required this.bytes});
  final Uint8List bytes;
  @override
  State<_CameraImage> createState() => _CameraImageState();
}

class _CameraImageState extends State<_CameraImage> {
  ui.Image? _image;
  Uint8List? _pending;
  bool _decoding = false;
  Timer? _emptyDeadline;

  @override
  void initState() {
    super.initState();
    _queue();
  }

  @override
  void didUpdateWidget(_CameraImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Telemetry/ACK packets can repeat the same JPEG. Do not decode it again.
    if (!listEquals(oldWidget.bytes, widget.bytes)) _queue();
  }

  void _queue() {
    if (widget.bytes.isEmpty) {
      _pending = null;
      // A missing image packet must not destroy the decoder/render surface.
      // Bound the hold, however: no indefinite frozen image on a stalled input.
      _emptyDeadline ??= Timer(const Duration(milliseconds: 750), () {
        _emptyDeadline = null;
        if (!mounted || widget.bytes.isNotEmpty) return;
        final old = _image;
        setState(() => _image = null);
        WidgetsBinding.instance.addPostFrameCallback((_) => old?.dispose());
      });
      return;
    }
    _emptyDeadline?.cancel();
    _emptyDeadline = null;
    _pending = widget.bytes;
    if (!_decoding) unawaited(_decode());
  }

  Future<void> _decode() async {
    _decoding = true;
    try {
      while (mounted && _pending != null) {
        final bytes = _pending!;
        _pending = null;
        try {
          final codec = await ui.instantiateImageCodec(
            bytes,
            targetWidth: 640,
            targetHeight: 480,
          );
          ui.FrameInfo frame;
          try {
            frame = await codec.getNextFrame();
          } finally {
            codec.dispose();
          }
          if (!mounted || widget.bytes.isEmpty && _emptyDeadline == null) {
            frame.image.dispose();
            continue;
          }
          final old = _image;
          setState(() => _image = frame.image);
          // The old image may still be referenced by this frame's render object.
          WidgetsBinding.instance.addPostFrameCallback((_) => old?.dispose());
        } on Object {
          // A bad frame never acquires control or drops a queued good frame.
        }
      }
    } finally {
      _decoding = false;
    }
  }

  @override
  Widget build(BuildContext context) => _image == null
      ? const Center(child: Text('等待新鲜相机画面'))
      : RawImage(image: _image, fit: BoxFit.contain);

  @override
  void dispose() {
    _pending = null;
    _emptyDeadline?.cancel();
    _image?.dispose();
    super.dispose();
  }
}
