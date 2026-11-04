import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'presentation_snapshot.dart';
import 'presentation_view_model.dart';

class PresentationScreen extends StatelessWidget {
  const PresentationScreen({super.key, required this.model});
  final PresentationViewModel model;

  @override
  Widget build(BuildContext context) => CallbackShortcuts(
    bindings: {
      const SingleActivator(LogicalKeyboardKey.escape, includeRepeats: false):
          model.stop,
      const SingleActivator(
        LogicalKeyboardKey.arrowRight,
        includeRepeats: false,
      ): model.next,
      const SingleActivator(
        LogicalKeyboardKey.arrowLeft,
        includeRepeats: false,
      ): model.previous,
    },
    child: Focus(
      autofocus: true,
      child: ListenableBuilder(
        listenable: model,
        builder: (context, _) {
          final value = model.latest;
          return Scaffold(
            backgroundColor: const Color(0xff102127),
            body: Column(
              children: [
                const Padding(
                  padding: EdgeInsets.fromLTRB(20, 14, 20, 10),
                  child: Text(
                    'APPLICATION-LOCAL / 人工演示 · 非外部软件控制',
                    style: TextStyle(color: Color(0xffefbd80), fontSize: 12),
                  ),
                ),
                Expanded(
                  child: value == null
                      ? Center(
                          child: Text(
                            model.link == PresentationLink.connecting
                                ? '连接演示核心…'
                                : '连接后可预览内置演示稿',
                          ),
                        )
                      : SingleChildScrollView(
                          child: Padding(
                            padding: const EdgeInsets.all(20),
                            child: _SlideSurface(value: value, model: model),
                          ),
                        ),
                ),
                const Divider(height: 1),
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Text(
                        model.message,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontSize: 12),
                      ),
                      const SizedBox(height: 6),
                      if (value != null)
                        Text(
                          '应用状态 ${value.state} / 修订 ${value.revision} · ${value.rendered ? 'Flutter已绘制该修订' : '等待绘制确认'}',
                          key: const Key('presentation-render-status'),
                          style: const TextStyle(
                            color: Color(0xff91dcc7),
                            fontSize: 12,
                          ),
                        ),
                      if (value?.lastAction != null)
                        Text(
                          _receipt(value!.lastAction!),
                          key: const Key('presentation-action-result'),
                          style: const TextStyle(fontSize: 12),
                        ),
                      const SizedBox(height: 8),
                      Align(
                        alignment: Alignment.centerLeft,
                        child: FilledButton.icon(
                          key: const Key('presentation-stop'),
                          onPressed: model.stop,
                          style: FilledButton.styleFrom(
                            backgroundColor: const Color(0xffefbd80),
                            foregroundColor: const Color(0xff25221c),
                          ),
                          icon: const Icon(Icons.stop),
                          label: Text(
                            model.stopPending ? '等待停止确认' : '停止 · Esc',
                          ),
                        ),
                      ),
                      const SizedBox(height: 8),
                      SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: Row(
                          spacing: 8,
                          children: [
                            if (model.link == PresentationLink.disconnected)
                              OutlinedButton(
                                onPressed: model.connect,
                                child: const Text('连接演示核心'),
                              )
                            else
                              OutlinedButton.icon(
                                key: const Key('presentation-play'),
                                onPressed: model.canPlay ? model.play : null,
                                icon: const Icon(Icons.play_arrow),
                                label: const Text('人工播放'),
                              ),
                            OutlinedButton.icon(
                              key: const Key('presentation-previous'),
                              onPressed: model.canNavigate
                                  ? model.previous
                                  : null,
                              icon: const Icon(Icons.arrow_back),
                              label: const Text('上一页'),
                            ),
                            OutlinedButton.icon(
                              key: const Key('presentation-next'),
                              onPressed: model.canNavigate ? model.next : null,
                              icon: const Icon(Icons.arrow_forward),
                              label: const Text('下一页'),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 6),
                      const Text(
                        '方向键不接受长按连发。Esc仅本窗口有效；手势控制请切换右上角“手势翻页”，不控制外部应用。',
                        style: TextStyle(
                          color: Color(0xffa6b9bc),
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          );
        },
      ),
    ),
  );

  String _receipt(PageReceipt value) => switch (value.outcome) {
    'application_updated' =>
      '应用状态更新：${value.before} → ${value.after}（请求 ${value.requestId}）',
    'boundary_no_change' => '已到页边界：页码保持 ${value.after}，不是重复注入',
    'rejected' => '请求被拒绝：${value.reason}',
    _ => '结果不确定：不自动重试，需停止并重新播放',
  };
}

class _SlideSurface extends StatelessWidget {
  const _SlideSurface({required this.value, required this.model});
  final PresentationSnapshot value;
  final PresentationViewModel model;

  @override
  Widget build(BuildContext context) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (context.mounted) model.surfacePainted(value);
    });
    return Semantics(
      label: '内置演示第${value.page}页，共${value.pageCount}页，${value.title}',
      liveRegion: true,
      child: Container(
        width: double.infinity,
        constraints: const BoxConstraints(minHeight: 320),
        decoration: const BoxDecoration(
          color: Color(0xfff3eee3),
          border: Border(left: BorderSide(color: Color(0xffc78947), width: 5)),
        ),
        padding: const EdgeInsets.all(28),
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
                '${value.page.toString().padLeft(2, '0')} / ${value.pageCount.toString().padLeft(2, '0')}',
                key: const Key('presentation-page'),
                style: const TextStyle(fontSize: 18, letterSpacing: 4),
              ),
              const SizedBox(height: 24),
              Text(
                value.title,
                key: const Key('presentation-title'),
                style: const TextStyle(
                  fontSize: 36,
                  fontWeight: FontWeight.w600,
                  height: 1.2,
                ),
              ),
              const SizedBox(height: 28),
              for (final line in value.lines)
                Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Text(line),
                ),
              const SizedBox(height: 24),
              Text(
                value.deckTitle,
                style: const TextStyle(color: Color(0xff627778), fontSize: 12),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
