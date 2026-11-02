import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:handshake_stage/core/core_diagnostics.dart';
import 'package:handshake_stage/stage/stage_view_model.dart';

import 'test_support.dart';

void main() {
  test(
    'chunked lifecycle events keep no arbitrary SDK text or private paths',
    () {
      final value = CoreDiagnostics();
      value.add(
        utf8.encode('SDK /private/model.task error\nHS_DIAG {"phase":"model_'),
      );
      value.add(utf8.encode('loading"}\n'));
      expect(value.summary, 'model_loading');
      value.add(
        utf8.encode(
          'HS_DIAG {"phase":"model_failed","error_type":"ValueError"}\n',
        ),
      );
      value.add(utf8.encode('HS_DIAG {"phase":"release_complete"}\n'));
      expect(value.summary, 'model_failed');
      expect(value.lastPhase, 'release_complete');
      value.add(utf8.encode('HS_DIAG {"phase":"/private/secret"}\n'));
      expect(value.summary, 'model_failed');
    },
  );

  test(
    'oversized malformed stderr is bounded and next valid event still works',
    () {
      final value = CoreDiagnostics();
      value.add(List.filled(100000, 65));
      value.add([10, 255, 10]);
      value.add(
        utf8.encode('HS_DIAG {bad json}\nHS_DIAG {"phase":"camera_ready"}\n'),
      );
      expect(value.summary, 'camera_ready');
    },
  );

  test(
    'disconnect shows safe phase and actual exit code, clears stale state',
    () async {
      final core = FakeCore();
      final model = StageViewModel(() => core);
      await model.connect();
      core.emit();
      core.controller.addError(
        CoreProcessFailure('model_loading', exitCode: -9),
      );
      expect(model.link, LinkState.disconnected);
      expect(model.message, contains('model_loading'));
      expect(model.message, contains('-9'));
      expect(model.latest, isNull);
      expect(model.canStartCamera, isFalse);
      model.dispose();
      await Future<void>.delayed(Duration.zero);
    },
  );
}
