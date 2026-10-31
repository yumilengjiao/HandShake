import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../core/core_snapshot.dart';
import 'skeleton_painter.dart';

Future<ui.Image> decodePreviewImage(CoreSnapshot snapshot) async {
  final buffer = await ui.ImmutableBuffer.fromUint8List(snapshot.jpeg!);
  ui.ImageDescriptor? descriptor;
  ui.Codec? codec;
  try {
    descriptor = await ui.ImageDescriptor.encoded(buffer);
    // Check source dimensions before allocating decoded pixels, not after resizing.
    if (descriptor.width != snapshot.width ||
        descriptor.height != snapshot.height) {
      throw const FormatException('JPEG dimensions do not match source frame');
    }
    codec = await descriptor.instantiateCodec();
    return (await codec.getNextFrame()).image;
  } finally {
    codec?.dispose();
    descriptor?.dispose();
    buffer.dispose();
  }
}

/// One decode in flight, one latest desired frame. No global ImageCache of camera frames.
class PreviewSurface extends StatefulWidget {
  const PreviewSurface({
    super.key,
    required this.snapshot,
    required this.history,
    required this.mirror,
    required this.reviewing,
    this.decoder = decodePreviewImage,
  });
  final CoreSnapshot? snapshot;
  final List<CoreSnapshot> history;
  final bool mirror;
  final bool reviewing;
  final Future<ui.Image> Function(CoreSnapshot) decoder;

  @override
  State<PreviewSurface> createState() => _PreviewSurfaceState();
}

class _PreviewSurfaceState extends State<PreviewSurface> {
  ui.Image? _image;
  CoreSnapshot? _desired;
  CoreSnapshot? _decoded;
  bool _decoding = false;
  bool _failed = false;

  String? _key(CoreSnapshot? value) => value == null
      ? null
      : '${value.sessionId}/${value.epoch}/${value.frameId}/${value.state}/${value.source}';

  @override
  void initState() {
    super.initState();
    _update();
  }

  @override
  void didUpdateWidget(covariant PreviewSurface oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (_key(oldWidget.snapshot) != _key(widget.snapshot)) _update();
  }

  bool _sameStream(CoreSnapshot? a, CoreSnapshot? b) =>
      a != null &&
      b != null &&
      a.sessionId == b.sessionId &&
      a.epoch == b.epoch &&
      a.source == b.source &&
      a.isPreview &&
      b.isPreview;

  bool _canDisplay(CoreSnapshot target) =>
      _key(target) == _key(_desired) ||
      (!widget.reviewing &&
          _sameStream(target, _desired) &&
          target.frameId <= _desired!.frameId &&
          _desired!.frameTimestampNs! - target.frameTimestampNs! <=
              BigInt.from(750000000));

  void _update() {
    _desired = widget.snapshot;
    if (_decoded == null || !_canDisplay(_decoded!)) {
      _image?.dispose();
      _image = null;
      _decoded = null;
    }
    _failed = false;
    _decode();
  }

  Future<void> _decode() async {
    final target = _desired;
    if (_decoding || target == null || target.jpeg == null) return;
    _decoding = true;
    ui.Image? result;
    try {
      result = await widget.decoder(target);
      if (!mounted || !_canDisplay(target)) {
        result.dispose();
      } else {
        setState(() {
          _image?.dispose();
          _image = result;
          _decoded = target;
        });
      }
    } on Object {
      if (mounted && _key(target) == _key(_desired)) {
        setState(() {
          _failed = true;
        });
      }
    } finally {
      _decoding = false;
      if (mounted && _key(target) != _key(_desired)) _decode();
    }
  }

  @override
  void dispose() {
    _desired = null;
    _image?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final camera = widget.snapshot?.isCamera == true;
    return Stack(
      fit: StackFit.expand,
      children: [
        CustomPaint(
          painter: SkeletonPainter(
            snapshot: camera ? _decoded : widget.snapshot,
            history: widget.history,
            mirror: widget.mirror,
            reviewing: widget.reviewing,
            image: _image,
          ),
        ),
        if (camera && _decoded != null)
          Positioned(
            left: 8,
            top: 8,
            child: ColoredBox(
              color: const Color(0x99000000),
              child: Padding(
                padding: const EdgeInsets.all(4),
                child: Text(
                  '同帧图像 / 骨架 #${_decoded!.frameId}',
                  style: const TextStyle(fontSize: 12),
                ),
              ),
            ),
          ),
        if (camera && _image == null)
          Center(
            child: Text(
              _failed ? '预览图像解码失败 · 不显示未对齐骨架' : '等待当前相机帧…',
              textAlign: TextAlign.center,
            ),
          ),
      ],
    );
  }
}
