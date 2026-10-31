import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/core_snapshot.dart';

const stageMint = Color(0xff91dcc7);
const stageGold = Color(0xffefbd80);

/// One transform for both overlays and a future 4:3 source image. Mirror is display-only.
Offset projectPoint(StagePoint point, Size size, {required bool mirror}) {
  final width = math.min(size.width, size.height * 4 / 3);
  final height = width * 3 / 4;
  return Offset(
    (size.width - width) / 2 + (mirror ? 1 - point.x : point.x) * width,
    (size.height - height) / 2 + point.y * height,
  );
}

class SkeletonPainter extends CustomPainter {
  SkeletonPainter({
    required this.snapshot,
    required this.history,
    required this.mirror,
    required this.reviewing,
  });
  final CoreSnapshot? snapshot;
  final List<CoreSnapshot> history;
  final bool mirror;
  final bool reviewing;

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final grid = Paint()
      ..color = const Color(0xff273b41)
      ..strokeWidth = 1;
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
        center,
        math.min(size.width, size.height) * ratio,
        orbit,
      );
    }
    final points = snapshot?.landmarks ?? const <StagePoint>[];
    if (points.isEmpty) return;
    Offset project(StagePoint point) =>
        projectPoint(point, size, mirror: mirror);
    if (!reviewing) {
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
    final bone = Paint()
      ..color = reviewing ? stageGold : stageMint
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    for (var finger = 0; finger < 5; finger++) {
      final base = 1 + finger * 4;
      canvas.drawLine(project(points[0]), project(points[base]), bone);
      for (var joint = base; joint < base + 3; joint++) {
        canvas.drawLine(
          project(points[joint]),
          project(points[joint + 1]),
          bone,
        );
      }
    }
    for (var i = 0; i < points.length; i++) {
      canvas.drawCircle(
        project(points[i]),
        i == 0 ? 7 : 4,
        Paint()..color = i == 0 ? stageGold : const Color(0xffe6f7ef),
      );
    }
  }

  @override
  bool shouldRepaint(covariant SkeletonPainter oldDelegate) =>
      oldDelegate.snapshot != snapshot ||
      oldDelegate.mirror != mirror ||
      oldDelegate.reviewing != reviewing ||
      oldDelegate.history != history;
}
