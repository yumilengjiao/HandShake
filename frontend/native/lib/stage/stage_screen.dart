import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'preview_surface.dart';
import 'skeleton_painter.dart';
import 'stage_view_model.dart';

class StageScreen extends StatelessWidget {
  const StageScreen({super.key, required this.model});
  final StageViewModel model;

  @override
  Widget build(BuildContext context) => CallbackShortcuts(
    bindings: {const SingleActivator(LogicalKeyboardKey.escape): model.stop},
    child: Focus(
      autofocus: true,
      child: ListenableBuilder(
        listenable: model,
        builder: (context, _) => Scaffold(
          backgroundColor: const Color(0xff102127),
          body: SafeArea(
            child: Column(
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(24, 18, 24, 12),
                  child: Wrap(
                    alignment: WrapAlignment.spaceBetween,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    spacing: 32,
                    runSpacing: 8,
                    children: [
                      const Text(
                        'HandShake / 动作有迹',
                        style: TextStyle(
                          fontSize: 23,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      Text(
                        '${model.latest?.isCamera == true ? 'CAMERA' : 'SYNTHETIC'}  ·  DRY-RUN  ·  无应用控制',
                        style: TextStyle(color: stageGold, fontSize: 12),
                      ),
                    ],
                  ),
                ),
                const Divider(height: 1, color: Color(0xff30434a)),
                Expanded(
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      final wide =
                          constraints.maxWidth >= 960 &&
                          MediaQuery.textScalerOf(context).scale(16) < 25;
                      return SingleChildScrollView(
                        child: Padding(
                          padding: const EdgeInsets.all(24),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              if (wide)
                                Row(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Expanded(
                                      child: _stage(
                                        context,
                                        height: (constraints.maxHeight - 200)
                                            .clamp(240, 520),
                                      ),
                                    ),
                                    const SizedBox(width: 32),
                                    SizedBox(
                                      width: 270,
                                      child: _decisions(context),
                                    ),
                                  ],
                                )
                              else ...[
                                _stage(context, height: 300),
                              ],
                              const SizedBox(height: 20),
                              _timeline(context),
                              if (!wide)
                                ExpansionTile(
                                  title: Text(
                                    model.displayed?.classification == null
                                        ? '展开决策轨迹 · 模型未加载'
                                        : '展开决策轨迹 · 未校准分类',
                                  ),
                                  children: [_decisions(context)],
                                ),
                              if (model.explain)
                                Padding(
                                  padding: const EdgeInsets.only(top: 20),
                                  child: Text(
                                    '讲解模式 / 骨架 → 共享特征与缺失掩码 → 因果TCN → 校准拒识 → 安全策略。'
                                    '\n仅展示原始分类logits；未校准、未建立操作意图或动作授权。'
                                    '32步是UI历史容量，不代表当前网络的感受野；帧数也不等于固定时长。',
                                    style: TextStyle(
                                      color: stageMint,
                                      height: 1.7,
                                    ),
                                  ),
                                ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
                const Divider(height: 1, color: Color(0xff30434a)),
                _controls(context),
              ],
            ),
          ),
        ),
      ),
    ),
  );

  Widget _stage(BuildContext context, {required double height}) {
    final snapshot = model.displayed;
    final hasPoints = snapshot?.landmarks.isNotEmpty == true;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 16,
          runSpacing: 6,
          children: [
            Text(
              model.reviewing ? '01 / 时间透镜 · 只读回看' : '01 / 交互舞台',
              style: TextStyle(color: model.reviewing ? stageGold : stageMint),
            ),
            Text(
              model.mirror ? '显示镜像 ON · 不改变源坐标' : '源坐标 · 未镜像',
              style: const TextStyle(color: Color(0xffa6b9bc), fontSize: 12),
            ),
          ],
        ),
        const SizedBox(height: 10),
        SizedBox(
          height: height,
          width: double.infinity,
          child: Semantics(
            label: snapshot?.isCamera == true
                ? '本机相机画面与同帧手部骨架'
                : (hasPoints ? '合成手部骨架，21个关键点' : '无当前骨架'),
            child: Stack(
              fit: StackFit.expand,
              children: [
                PreviewSurface(
                  snapshot: snapshot,
                  history: model.history,
                  mirror: model.mirror,
                  reviewing: model.reviewing,
                ),
                if (!hasPoints && snapshot?.isCamera != true)
                  Center(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Text(
                        snapshot?.isPreview == true
                            ? '此合成帧没有手\n缺失就是证据，不补画骨架'
                            : '让动作成为界面的中心\n点击“开始合成预览”',
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          color: Color(0xffa6b9bc),
                          height: 1.8,
                        ),
                      ),
                    ),
                  ),
                Positioned(
                  left: 0,
                  bottom: 0,
                  right: 0,
                  child: Text(
                    snapshot?.isCamera == true
                        ? '相机只在内存预览 / ${snapshot!.perceptionStatus} / ${snapshot.captureHz?.toStringAsFixed(1) ?? '—'} Hz采集'
                        : '程序生成的测试骨架 / 不是相机画面或识别结果',
                    style: TextStyle(color: stageGold, fontSize: 12),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _decisions(BuildContext context) {
    final snapshot = model.displayed;
    final classification = snapshot?.classification;
    final modelEvidence = classification == null
        ? (snapshot?.modelStatus == 'error'
              ? '模型包校验/加载失败\n没有可用预测'
              : '未加载手势分类模型\nMediaPipe骨架不等于手势识别')
        : '${classification.modelType.toUpperCase()} / ${classification.vocabularyVersion}\n'
              '包 ${classification.modelVersion.length > 20 ? '${classification.modelVersion.substring(0, 20)}…' : classification.modelVersion}\n'
              '${classification.syntheticTraining ? '合成训练 · 仅管线诊断' : '训练数据非合成 · 效果仍待实测'}\n'
              '${classification.state == 'predicted' ? '来源类 ${classification.topToken} / logit ${classification.topLogit!.toStringAsFixed(3)}' : '${classification.reason} · 历史${classification.windowLength}/32步'}\n'
              '源帧 #${classification.frameId} · 未校准，不是操作意图';
    final steps = [
      (
        '观测 / OBSERVE',
        snapshot?.isPreview == true
            ? '帧 ${snapshot!.frameId} · ${snapshot.hands.length}只手\n'
                  '${snapshot.isCamera ? (snapshot.perceptionStatus == 'unavailable' ? '未配置感知资产，仅视频' : '感知: ${snapshot.perceptionStatus}') : '程序生成的合成骨架'}'
            : '当前没有输入帧',
      ),
      ('模型 / INFER', modelEvidence),
      ('策略 / DECIDE', '仅预览 · 不具备控制能力\n15个来源类别不直接等于操作意图'),
      ('结果 / ACT', '没有发送应用命令\n翻页、音乐、光标暂不可用'),
    ];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('02 / 决策轨迹', style: TextStyle(color: stageMint)),
        const SizedBox(height: 18),
        for (var i = 0; i < steps.length; i++)
          IntrinsicHeight(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Column(
                  children: [
                    Text(
                      '${i + 1}'.padLeft(2, '0'),
                      style: TextStyle(color: stageGold, fontSize: 12),
                    ),
                    Expanded(
                      child: VerticalDivider(
                        color: stageMint.withValues(alpha: .22),
                        width: 24,
                      ),
                    ),
                  ],
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.only(bottom: 24),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          steps[i].$1,
                          style: const TextStyle(fontWeight: FontWeight.w600),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          steps[i].$2,
                          style: const TextStyle(
                            color: Color(0xffa6b9bc),
                            height: 1.6,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        if (snapshot != null)
          Text(
            'epoch ${snapshot.epoch} / seq ${snapshot.sequence}\n'
            '源时间 ns: ${snapshot.frameTimestampNs ?? '—'}',
            style: const TextStyle(color: Color(0xffa6b9bc), fontSize: 11),
          ),
      ],
    );
  }

  Widget _timeline(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Wrap(
        crossAxisAlignment: WrapCrossAlignment.center,
        spacing: 12,
        children: [
          const Text('03 / 最近32步 · 点击只读回看', style: TextStyle(color: stageMint)),
          TextButton(
            onPressed: model.reviewing ? model.returnToNow : null,
            child: const Text('返回当前'),
          ),
        ],
      ),
      SizedBox(
        height: 64,
        child: model.history.isEmpty
            ? const Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  '等待收到的新帧，不预填时间轴。',
                  style: TextStyle(color: Color(0xffa6b9bc)),
                ),
              )
            : ListView.separated(
                scrollDirection: Axis.horizontal,
                itemCount: model.history.length,
                separatorBuilder: (_, _) => const SizedBox(width: 4),
                itemBuilder: (context, index) {
                  final frame = model.history[index];
                  return Semantics(
                    label:
                        '回看${frame.isCamera ? '相机' : '合成'}帧 ${frame.frameId}',
                    button: true,
                    selected: model.inspected == frame,
                    child: Tooltip(
                      message:
                          '帧 ${frame.frameId} · ${switch (frame.perceptionStatus) {
                            'unavailable' => '未感知',
                            'not_processed' => '未处理',
                            'error' => '感知错误',
                            'processed_no_hand' => '无手',
                            _ => '骨架',
                          }}',
                      child: SizedBox(
                        width: 40,
                        child: TextButton(
                          onPressed: () => model.inspect(index),
                          style: TextButton.styleFrom(
                            padding: EdgeInsets.zero,
                            backgroundColor: model.inspected == frame
                                ? const Color(0xff40524e)
                                : null,
                          ),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Container(
                                width: 3,
                                height: frame.landmarks.isEmpty ? 8 : 22,
                                color: frame.landmarks.isEmpty
                                    ? stageGold
                                    : stageMint,
                              ),
                              const SizedBox(height: 6),
                              Text(
                                '${frame.frameId}',
                                style: const TextStyle(fontSize: 10),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
      ),
    ],
  );

  Widget _controls(BuildContext context) => Padding(
    padding: const EdgeInsets.fromLTRB(20, 12, 20, 12),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          model.message,
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(color: Color(0xffbdccce), fontSize: 12),
        ),
        const SizedBox(height: 8),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            FilledButton.icon(
              key: const Key('stop'),
              onPressed: model.stop,
              style: FilledButton.styleFrom(
                backgroundColor: stageGold,
                foregroundColor: const Color(0xff25221c),
              ),
              icon: const Icon(Icons.stop_rounded),
              label: Text(model.stopPending ? '等待停止确认' : '停止 · Esc'),
            ),
            const SizedBox(height: 8),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                spacing: 10,
                children: [
                  if (model.latest?.version == 2 || model.latest?.version == 3)
                    OutlinedButton.icon(
                      key: const Key('start_camera'),
                      onPressed: model.canStartCamera
                          ? model.startCamera
                          : null,
                      icon: const Icon(Icons.videocam_outlined),
                      label: const Text('打开相机预览'),
                    ),
                  if (model.link == LinkState.disconnected)
                    OutlinedButton(
                      onPressed: model.connect,
                      child: const Text('连接核心'),
                    )
                  else
                    OutlinedButton(
                      key: const Key('start'),
                      onPressed: model.canStart ? model.start : null,
                      child: const Text('开始合成预览'),
                    ),
                  IconButton(
                    onPressed: model.toggleMirror,
                    tooltip: '切换显示镜像',
                    icon: Icon(model.mirror ? Icons.flip : Icons.flip_outlined),
                  ),
                  IconButton(
                    onPressed: model.toggleExplain,
                    tooltip: '切换讲解模式',
                    icon: const Icon(Icons.account_tree_outlined),
                  ),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        const Text(
          'Esc 仅窗口内有效，不是系统全局停止。',
          style: TextStyle(color: Color(0xff94a8ad), fontSize: 11),
        ),
      ],
    ),
  );
}
