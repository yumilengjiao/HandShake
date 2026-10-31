import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:handshake_stage/core/core_repository.dart';
import 'package:handshake_stage/core/core_snapshot.dart';

Map<String, dynamic> fixture([Map<String, Object?> changes = const {}]) =>
    (jsonDecode(File('test/fixtures/snapshot.json').readAsStringSync())
          as Map<String, dynamic>)
      ..addAll(changes);

CoreSnapshot snapshot([Map<String, Object?> changes = const {}]) =>
    CoreSnapshot.parse(jsonEncode(fixture(changes)));

class FakeCore implements CoreRepository {
  final controller = StreamController<CoreSnapshot>.broadcast(sync: true);
  final commands = <String>[];
  bool closed = false;
  @override
  Stream<CoreSnapshot> get snapshots => controller.stream;
  @override
  Future<void> connect() async {}
  @override
  void send(String kind, CoreSnapshot context, BigInt commandId) {
    commands.add(kind);
  }

  @override
  Future<void> close() async {
    closed = true;
    await Future<void>.value();
    await controller.close();
  }

  void emit([Map<String, Object?> changes = const {}]) =>
      controller.add(snapshot(changes));
}
