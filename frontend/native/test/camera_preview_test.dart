import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/core/core_snapshot.dart';
import 'package:handshake_stage/core/preview_packets.dart';
import 'package:handshake_stage/stage/preview_surface.dart';
import 'package:handshake_stage/stage/skeleton_painter.dart';
import 'package:handshake_stage/stage/stage_screen.dart';
import 'package:handshake_stage/stage/stage_view_model.dart';

import 'test_support.dart';

Uint8List wireFixture() =>
    File('test/fixtures/preview_v2.bin').readAsBytesSync();

CoreSnapshot cameraFrame([
  Map<String, Object?> changes = const {},
  bool image = true,
]) {
  final bytes = wireFixture();
  final size = ByteData.sublistView(bytes).getUint32(4);
  final metadata = jsonDecode(
    utf8.decode(bytes.sublist(12, 12 + size)),
  ) as Map<String, dynamic>;
  metadata.addAll(changes);
  return CoreSnapshot.parse(
    jsonEncode(metadata),
    jpeg: image ? Uint8List.sublistView(bytes, 12 + size) : null,
  );
}

Map<String, Object?> inactive(String state) => {
  'state': state,
  'frame_width': 0,
  'frame_height': 0,
  'frame_timestamp_ns': null,
  'hands': [],
  'perception_status': 'unavailable',
};

void main() {
  test(
    'Python packet preserves image, hands, dimensions and >2^53 time',
    () async {
      final bytes = wireFixture();
      final frames = await decodePreviewPackets(
        Stream.fromIterable([
          for (var i = 0; i < bytes.length; i++) [bytes[i]],
        ]),
      ).toList();
      expect(frames, hasLength(1));
      final frame = frames.single;
      expect(frame.frameTimestampNs, BigInt.parse('9007199254740999'));
      expect(frame.hands.single.points.length, 21);
      expect(frame.width, 160);
      expect(() => frame.jpeg![0] = 0, throwsUnsupportedError);
      expect(
        () => frame.hands.single.validity[0] = false,
        throwsUnsupportedError,
      );
    },
  );

  test('multiple binary packets and incomplete/oversized data', () async {
    final bytes = wireFixture();
    expect(
      await decodePreviewPackets(Stream.value([...bytes, ...bytes])).length,
      2,
    );
    await expectLater(
      decodePreviewPackets(Stream.value(bytes.sublist(0, bytes.length - 1)))
          .toList(),
      throwsFormatException,
    );
    final prefix = Uint8List(12)..setRange(0, 4, ascii.encode('HSP2'));
    ByteData.sublistView(prefix).setUint32(4, 0x7fffffff);
    await expectLater(
      decodePreviewPackets(Stream.value(prefix)).toList(),
      throwsFormatException,
    );
    prefix[3] = 57;
    await expectLater(
      decodePreviewPackets(Stream.value(prefix)).toList(),
      throwsFormatException,
    );
  });

  for (final change in <Map<String, Object?>>[
    {'frame_width': 641},
    {'source': 'synthetic'},
    {'timestamp_kind': 'synthetic_generation'},
    {'capture_hz': -1},
    {'control_available': true},
    {'perception_status': 'unavailable'},
    {'state': 'stopping'},
    {'hands': [], 'frame_timestamp_ns': null},
  ]) {
    test('inconsistent v2 metadata rejected: $change', () {
      expect(() => cameraFrame(change), throwsFormatException);
    });
  }

  testWidgets('JPEG dimensions verified, RGB preserved; no image cache', (
    tester,
  ) async {
    final image = await tester.runAsync(
      () => decodePreviewImage(cameraFrame()),
    );
    expect(image!.width, 160);
    final pixels = await tester.runAsync(() => image.toByteData());
    final offsetLeft = (20 * 160 + 20) * 4;
    final offsetRight = (20 * 160 + 140) * 4;
    expect(pixels!.getUint8(offsetLeft), greaterThan(90));
    expect(pixels.getUint8(offsetLeft + 2), lessThan(5));
    expect(pixels.getUint8(offsetRight + 2), greaterThan(90));
    image.dispose();
    await tester.runAsync(() async {
      await expectLater(
        decodePreviewImage(cameraFrame({'frame_width': 161})),
        throwsFormatException,
      );
    });
  });

  testWidgets('one decode plus latest frame, stop discards in-flight image', (
    tester,
  ) async {
    final images = await tester.runAsync(
      () async => [
        await decodePreviewImage(cameraFrame()),
        await decodePreviewImage(cameraFrame()),
      ],
    );
    final pending = <Completer<ui.Image>>[];
    final ids = <BigInt>[];
    Future<ui.Image> decoder(CoreSnapshot value) {
      ids.add(value.frameId);
      final completer = Completer<ui.Image>();
      pending.add(completer);
      return completer.future;
    }

    Widget surface(CoreSnapshot value) => MaterialApp(
      home: PreviewSurface(
        snapshot: value,
        history: const [],
        mirror: false,
        reviewing: false,
        decoder: decoder,
      ),
    );
    await tester.pumpWidget(surface(cameraFrame()));
    await tester.pumpWidget(surface(cameraFrame({'frame_id': '18'})));
    await tester.pumpWidget(surface(cameraFrame({'frame_id': '19'})));
    expect(ids, [BigInt.from(17)]);
    pending[0].complete(images![0]);
    await tester.pump();
    expect(ids, [BigInt.from(17), BigInt.from(19)]);
    expect(
      images[0].debugDisposed,
      isFalse,
    ); // retained image and its own bones stay paired
    await tester.pumpWidget(surface(cameraFrame(inactive('stopped'), false)));
    pending[1].complete(images[1]);
    await tester.pump();
    expect(images[0].debugDisposed, isTrue);
    expect(images[1].debugDisposed, isTrue);
    final paint = tester
        .widgetList<CustomPaint>(find.byType(CustomPaint))
        .map((widget) => widget.painter)
        .whereType<SkeletonPainter>()
        .single;
    expect(paint.image, isNull);
    expect(paint.snapshot, isNull);
    await tester.pumpWidget(const SizedBox.shrink());
  });

  testWidgets('stopping is not stop confirmation; camera action explicit', (
    tester,
  ) async {
    final core = FakeCore();
    final model = StageViewModel(() => core);
    addTearDown(model.dispose);
    await model.connect();
    core.controller.add(
      cameraFrame({...inactive('stopped'), 'epoch': '0'}, false),
    );
    model.startCamera();
    expect(core.commands, ['start_camera']);
    core.controller.add(
      cameraFrame({
        ...inactive('starting'),
        'sequence': '2',
        'last_command_id': '1',
      }, false),
    );
    model.stop();
    core.controller.add(
      cameraFrame({
        ...inactive('stopping'),
        'reason': 'user_stop',
        'sequence': '3',
        'epoch': '2',
        'last_command_id': '2',
      }, false),
    );
    expect(model.stopPending, isTrue);
    expect(model.canStartCamera, isFalse);
    core.controller.add(
      cameraFrame({
        ...inactive('stopped'),
        'reason': 'user_stop',
        'sequence': '4',
        'epoch': '2',
        'last_command_id': '2',
      }, false),
    );
    expect(model.stopPending, isFalse);
    model.dispose();
  });

  testWidgets('camera controls fit narrow screen at doubled text size', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(360, 700);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final core = FakeCore();
    final model = StageViewModel(() => core);
    await model.connect();
    core.controller.add(cameraFrame(inactive('stopped'), false));
    await tester.pumpWidget(
      MaterialApp(
        home: MediaQuery(
          data: const MediaQueryData(textScaler: TextScaler.linear(2)),
          child: StageScreen(model: model),
        ),
      ),
    );
    expect(tester.takeException(), isNull);
    expect(find.byKey(const Key('stop')).hitTestable(), findsOneWidget);
    expect(find.byKey(const Key('start_camera')).hitTestable(), findsOneWidget);
    await tester.pumpWidget(const SizedBox.shrink());
    model.dispose();
  });

  test('non-4:3 image projection and mirror are aligned', () {
    const point = StagePoint(.25, .5, 0);
    final position = projectPoint(
      point,
      const Size(800, 300),
      mirror: false,
      aspect: 16 / 9,
    );
    expect(position.dx, closeTo(800 / 2 - (300 * 16 / 9) / 4, 1e-9));
    expect(position.dy, 150);
  });
}
