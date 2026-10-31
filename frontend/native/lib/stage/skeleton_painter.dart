import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../core/core_snapshot.dart';

const stageMint = Color(0xff91dcc7);
const stageGold = Color(0xffefbd80);

Rect previewRect(Size size, {double aspect = 4 / 3}) {
  final width = math.min(size.width, size.height * aspect);
  final height = width / aspect;
  return Rect.fromLTWH(
    (size.width - width) / 2,
    (size.height - height) / 2,
    width,
    height,
  );
}

/// Both image and overlays use this exact, display-only letterbox/mirror transform.
Offset projectPoint(
  StagePoint point,
  Size size, {
  required bool mirror,
  double aspect = 4 / 3,
}) {
  final rect = previewRect(size, aspect: aspect);
  return Offset(
    rect.left + (mirror ? 1 - point.x : point.x) * rect.width,
    rect.top + point.y * rect.height,
  );
}

class SkeletonPainter extends CustomPainter {
  SkeletonPainter({
    required this.snapshot,
    required this.history,
    required this.mirror,
    required this.reviewing,
    this.image,
  });
  final CoreSnapshot? snapshot;
  final List<CoreSnapshot> history;
  final bool mirror;
  final bool reviewing;
  final ui.Image? image;

  @override
  void paint(Canvas canvas, Size size) {
    final current = snapshot;
    final camera = current?.isCamera == true;
    final aspect = current != null && current.width > 0
        ? current.width / current.height
        : 4 / 3;
    final rect = previewRect(size, aspect: aspect);
    if (camera) {
      if (image == null) {
        return; // Never put newer bones on a stale/undecoded image.
      }
      canvas.save();
      if (mirror) {
        canvas.translate(size.width, 0);
        canvas.scale(-1, 1);
      }
      paintImage(canvas: canvas, rect: rect, image: image!, fit: BoxFit.fill);
      canvas.restore();
    } else {
      final grid = Paint()..color = const Color(0xff273b41);
      for (double x = 24; x < size.width; x += 40) {
        for (double y = 24; y < size.height; y += 40) {
          canvas.drawCircle(Offset(x, y), .8, grid);
        }
      }
      final orbit = Paint()
        ..color = stageMint.withValues(alpha: .1)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1;
      for (final ratio in [.25, .38]) {
        canvas.drawCircle(
          Offset(size.width / 2, size.height / 2),
          math.min(size.width, size.height) * ratio,
          orbit,
        );
      }
    }
    Offset project(StagePoint point) =>
        projectPoint(point, size, mirror: mirror, aspect: aspect);
    canvas.save();
    canvas.clipRect(rect);
    if (!reviewing && !camera) {
      final trail = history.reversed.take(8).toList().reversed.toList();
      for (var i = 1; i < trail.length; i++) {
        if (trail[i - 1].landmarks.isEmpty || trail[i].landmarks.isEmpty) {
          continue;
        }
        canvas.drawLine(
          project(trail[i - 1].landmarks.first),
          project(trail[i].landmarks.first),
          Paint()
            ..color = stageGold.withValues(alpha: i / trail.length * .5)
            ..strokeWidth = 2,
        );
      }
    }
    for (final hand in current?.hands ?? <StageHand>[]) {
      final bone = Paint()
        ..color = reviewing || !hand.usable ? stageGold : stageMint
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round;
      void edge(int a, int b) {
        if (hand.validity[a] && hand.validity[b]) {
          canvas.drawLine(
            project(hand.points[a]),
            project(hand.points[b]),
            bone,
          );
        }
      }

      for (var finger = 0; finger < 5; finger++) {
        final base = 1 + finger * 4;
        edge(0, base);
        for (var joint = base; joint < base + 3; joint++) {
          edge(joint, joint + 1);
        }
      }
      for (var i = 0; i < hand.points.length; i++) {
        if (!hand.validity[i]) continue;
        canvas.drawCircle(
          project(hand.points[i]),
          i == 0 ? 6 : 3,
          Paint()..color = i == 0 ? stageGold : const Color(0xffe6f7ef),
        );
      }
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant SkeletonPainter old) =>
      old.snapshot != snapshot ||
      old.mirror != mirror ||
      old.reviewing != reviewing ||
      old.history != history ||
      old.image != image;
}
