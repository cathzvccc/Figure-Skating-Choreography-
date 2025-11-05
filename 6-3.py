# 音频BPM分段分析与花样滑冰步法匹配系统（最终优化版 - 带双图例可自定义版）
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import struct
from scipy.fftpack import fft
import time
import librosa
import librosa.display
import warnings
import json
from datetime import datetime
from collections import defaultdict
import textwrap

# ✅ 解决中文显示
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
    'Arial Unicode MS', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False


# 自动换行
def wrap_text(text, max_width=20):
    lines = []
    for line in text.split('\n'):
        wrapped_lines = textwrap.wrap(line, width=max_width)
        lines.extend(wrapped_lines)
    return '\n'.join(lines)


# 实时音频窗口（可选，可注释掉）
class AudioStream(object):
    def __init__(self):
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS,
                                  rate=self.RATE, input=True, output=True,
                                  frames_per_buffer=self.CHUNK)
        self.init_plots()
        self.start_plot()

    def init_plots(self):
        x = np.arange(0, 2 * self.CHUNK, 2)
        xf = np.linspace(0, self.RATE, self.CHUNK)
        self.fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.line, = ax1.plot(x, np.random.rand(self.CHUNK), '-', lw=2)
        self.line_fft, = ax2.semilogx(xf, np.random.rand(self.CHUNK), '-', lw=2)
        ax1.set_title('REAL-TIME AUDIO WAVEFORM')
        ax1.set_xlabel('Samples');
        ax1.set_ylabel('Volume')
        ax1.set_ylim(0, 255);
        ax1.set_xlim(0, 2 * self.CHUNK)
        ax2.set_title('REAL-TIME AUDIO SPECTRUM')
        ax2.set_xlabel('Frequency (Hz)');
        ax2.set_ylabel('Intensity')
        ax2.set_xlim(20, self.RATE / 2)
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(5, 120, 1910, 1070)
        plt.show(block=False)

    def start_plot(self):
        print('Real-time stream started (click window to pause)')
        frame_count = 0
        start_time = time.time()
        while not self.pause:
            data = self.stream.read(self.CHUNK)
            data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
            data_np = np.array(data_int, dtype=np.int16)[::2] + 128
            self.line.set_ydata(data_np)
            yf = fft(data_int)
            self.line_fft.set_ydata(np.abs(yf[0:self.CHUNK]) / (128 * self.CHUNK))
            self.fig.canvas.draw();
            self.fig.canvas.flush_events()
            frame_count += 1
        else:
            avg_fps = frame_count / (time.time() - start_time)
            print(f'Average frame rate: {avg_fps:.0f} FPS')
            self.exit_app()

    def exit_app(self):
        print('Real-time stream closed')
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def onClick(self, event):
        self.pause = True


# 文件选择
def select_audio_file():
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(
        caption="选择音频文件（支持WAV和MP3格式）",
        directory=".", filter="Audio Files (*.wav *.mp3);;Wave Files (*.wav);;MP3 Files (*.mp3);;All Files (*)"
    )
    app.quit()
    if file_path:
        return file_path
    else:
        print("未选择任何音频文件，程序将退出")
        sys.exit()


# 花样滑冰步法定义
SKATING_STEPS = {
    "60-90": {"bpm_range": (60, 90), "steps": ["Forward Outside Bracket", "Choctaw", "Spiral Sequence"]},
    "90-110": {"bpm_range": (90, 110), "steps": ["Backward Inside Bracket", "Rocker with Arm Sweep Glide"]},
    "110-130": {"bpm_range": (110, 130),
                "steps": ["Forward Outside Bracket", "Counters", "Hydroblading", "Running Edge"]},
    "130-150": {"bpm_range": (130, 150),
                "steps": ["Twizzle Sequence", "Loop Step", "Power Glide", "Choreographic Lunge"]},
    "150-170": {"bpm_range": (150, 170),
                "steps": ["Rocker–Choctaw Sequence", "Counter", "Running Edge", "Hydroblading with Body Wave"]},
    "170-190": {"bpm_range": (170, 190),
                "steps": ["Quick Twizzle Burst", "Backward Outside Bracket", "Power Slide", "Body Snap Glide"]},
    "190+": {"bpm_range": (190, 300),
             "steps": ["Twizzle + Counter Chain", "Sync Beat Hydroblading", "Body Pulse Motion"]}
}


# BPM -> 步法类型
def get_step_type(bpm_value):
    for k, v in SKATING_STEPS.items():
        if k == "渐强": continue
        min_bpm, max_bpm = v["bpm_range"]
        if min_bpm <= bpm_value <= max_bpm:
            return k
    if bpm_value > 160: return "高节奏适配"
    if bpm_value < 60: return "低节奏适配"
    return "未知"


# 获取建议步法
def get_segment_steps(segment):
    step_type = segment['step_type']
    if step_type in SKATING_STEPS:
        return SKATING_STEPS[step_type]['steps']
    elif step_type == "高节奏适配":
        return ["建议加快动作频率"]
    elif step_type == "低节奏适配":
        return ["建议延长动作幅度"]
    elif step_type == "停顿 间奏":
        return ["过渡或维持姿态"]
    return ["无推荐步法"]


# 合并区间
def merge_similar_segments(segments, min_duration=5):
    if not segments: return []
    merged = [segments[0]]
    for s in segments[1:]:
        last = merged[-1]
        if s["step_type"] == last["step_type"] or (s["end_time"] - s["start_time"]) < min_duration:
            merged[-1]["end_time"] = s["end_time"]
            d1 = last["end_time"] - last["start_time"]
            d2 = s["end_time"] - s["start_time"]
            merged[-1]["avg_bpm"] = (last["avg_bpm"] * d1 + s["avg_bpm"] * d2) / (d1 + d2)
        else:
            merged.append(s)
    return merged


# 保存 TXT
def save_analysis_result(audio_info, segments, save_dir="."):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"花样滑冰步法分段分析_{audio_info['文件名']}_{timestamp}.txt"
    path = f"{save_dir}/{name}".replace("//", "/")
    content = f"===== 花样滑冰音频BPM分段分析报告 =====\n分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n一、音频文件信息\n  文件名：{audio_info['文件名']}\n  采样率：{audio_info['采样率']} Hz\n  时长：{audio_info['时长']:.2f} 秒\n\n二、BPM分段与步法\n  共 {len(segments)} 段\n\n"
    for i, seg in enumerate(segments, 1):
        content += f"  {i}. 时间：{seg['start_time']:.2f}-{seg['end_time']:.2f}s | BPM：{seg['avg_bpm']:.2f} | 类型：{seg['step_type']}\n"
        steps = get_segment_steps(seg)
        content += "     建议步法：\n"
        for j, step in enumerate(steps, 1):
            content += f"       {j}. {step}\n"
        content += "\n"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


# 标注每个区间的步法类型
def add_step_annotations(ax, segments, times, bpms, colors):
    y_offset = 10
    for i, seg in enumerate(segments):
        step_type = seg["step_type"]
        avg_bpm = seg["avg_bpm"]
        mid_time = (seg["start_time"] + seg["end_time"]) / 2
        mid_bpm = avg_bpm
        ann_text = f"{step_type}\n({mid_bpm:.0f}BPM)"
        ax.annotate(wrap_text(ann_text, 12), xy=(mid_time, mid_bpm),
                    xytext=(0, y_offset), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7),
                    fontsize=8, ha='center', color='black', weight='bold')


# 主程序
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # stream = AudioStream()
    print("正在打开音频文件选择窗口...")
    audio_path = select_audio_file()
    print(f"已选中：{audio_path}\n")

    try:
        if audio_path.lower().endswith('.mp3'):
            y, sr = librosa.load(audio_path, sr=None, mono=True, dtype=np.float32)
        else:
            y, sr = librosa.load(audio_path, sr=None, mono=True, dtype=np.float32)
    except Exception as e:
        print(f"❌ 加载失败：{e}")
        sys.exit()

    audio_duration = len(y) / sr
    audio_file_name = audio_path.split("/")[-1] if "/" in audio_path else audio_path.split("\\")[-1]
    audio_info = {
        "文件名": audio_file_name,
        "文件路径": audio_path,
        "采样率": sr,
        "时长": audio_duration,
        "数据类型": str(y.dtype)
    }
    print("✅ 加载成功")
    print(f"📄 文件名：{audio_info['文件名']} | 时长：{audio_info['时长']:.2f}s | 采样率：{audio_info['采样率']}Hz")

    print("🔍 正在分段检测BPM...")
    segment_length = 5
    hop_length = 2048
    frame_length = 4096
    samples_per_segment = int(segment_length * sr)
    num_segments = int(np.ceil(len(y) / samples_per_segment))
    segments = []

    for i in range(num_segments):
        start = i * samples_per_segment
        end = min((i + 1) * samples_per_segment, len(y))
        seg_audio = y[start:end]
        if np.max(np.abs(seg_audio)) < 0.01:
            t1 = start / sr
            t2 = end / sr
            segments.append({"start_time": t1, "end_time": t2, "avg_bpm": 0, "step_type": "停顿 间奏"})
            continue
        onset = librosa.onset.onset_strength(y=seg_audio, sr=sr, hop_length=hop_length, aggregate=np.mean)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop_length, tightness=100)
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item()
        t1 = start / sr
        t2 = end / sr
        step_type = get_step_type(tempo)
        segments.append({"start_time": t1, "end_time": t2, "avg_bpm": tempo, "step_type": step_type})
        prog = (i + 1) / num_segments * 100
        if i % max(1, num_segments // 10) == 0:
            print(f"   进度：{prog:.1f}% | 时间：{t1:.1f}-{t2:.1f}s | BPM：{tempo:.1f}")

    print("\n🔗 合并相似区间...")
    merged = merge_similar_segments(segments)
    print(f"   合并前：{len(segments)} → 合并后：{len(merged)}")

    print("\n💾 保存分析结果...")
    save_analysis_result(audio_info, merged)
    json_name = f"音频BPM数据_{audio_file_name.replace('.wav', '').replace('.mp3', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_data = {
        "分析时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "音频信息": audio_info,
        "分段结果": merged,
        "步法规则": SKATING_STEPS
    }
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"📄 TXT：分析报告已保存 | 💾 JSON：{json_name}")

    print("\n📊 生成可视化图表...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 第一个图形：音频波形
    fig1, ax1 = plt.subplots(figsize=(18, 12))
    librosa.display.waveshow(y=y, sr=sr, ax=ax1, alpha=0.6, color="#1f77b4")
    ax1.set_title(f"音频波形 - {audio_file_name}", fontsize=14, pad=20)
    ax1.set_ylabel("振幅", fontsize=12)
    ax1.set_xlabel("时间（秒）", fontsize=12)
    wave_img_name = f"音频波形_{audio_file_name.replace('.wav', '').replace('.mp3', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout(pad=4.0)
    plt.savefig(wave_img_name, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 音频波形图表已保存：{wave_img_name}")

    # 第二个图形：BPM变化曲线与区间划分
    fig2, ax2 = plt.subplots(figsize=(18, 12))
    times = [seg["start_time"] for seg in merged]
    bpms = [seg["avg_bpm"] for seg in merged]
    colors = plt.cm.Set3(np.linspace(0, 1, len(merged)))

    ax2.plot(times, bpms, 'o-', color="#2ca02c", linewidth=2, markersize=6, markerfacecolor="#ff7f0e")
    ax2.set_title("BPM变化曲线与区间划分", fontsize=14, pad=20)
    ax2.set_xlabel("时间（秒）", fontsize=12)
    ax2.set_ylabel("BPM值", fontsize=12)
    ax2.set_ylim(0, (max(bpms) * 1.1) if bpms else 160)
    ax2.set_xlim(0, audio_duration)

    for i, seg in enumerate(merged):
        ax2.axvspan(seg["start_time"], seg["end_time"], alpha=0.15, facecolor=colors[i], linewidth=0)

    add_step_annotations(ax2, merged, times, bpms, colors)

    # ======================
    # 可修改的图例起始坐标参数（自定义）
    # ======================
    main_legend_x = 1.05  # 主要区间图例的水平位置（向右，如1.05）
    main_legend_y = 0.95  # 主要区间图例的垂直位置（0~1，越大越靠上，如0.95）

    step_legend_x = 1.05  # BPM步法图例的水平位置（与主要区间一致或不同）
    step_legend_y = 0.6  # BPM步法图例的垂直位置（0~1，比如0.6更靠下）
    # ======================

    # ======================
    # 图例1：主要区间（前5个区间）
    # ======================
    handles1 = []
    labels1 = []
    if len(merged) > 0:
        for i in range(min(5, len(merged))):
            color = colors[i]
            label = f"{i + 1}. {merged[i]['step_type']} ({merged[i]['avg_bpm']:.0f}BPM)"
            handles1.append(plt.Line2D([0], [0], color=color, lw=4))
            labels1.append(label)

    # 创建并添加第一个图例（主要区间）
    legend1 = ax2.legend(handles1, labels1,
                         bbox_to_anchor=(main_legend_x, main_legend_y),
                         loc='upper left',
                         fontsize=9,
                         title="主要区间",
                         title_fontsize=10)
    # 必须添加这个，不然会被下一个覆盖
    ax2.add_artist(legend1)

    # ======================
    # 图例2：BPM区间对应的花滑步法
    # ======================
    handles2 = []
    labels2 = []
    for idx, (range_key, step_info) in enumerate(SKATING_STEPS.items()):
        steps_text = "\n".join([
            textwrap.fill(step, width=40)
            for step in step_info["steps"]
        ])
        full_text = f"{range_key} BPM:\n{steps_text}\n{'-' * 25}"
        color = plt.cm.Set3(idx / len(SKATING_STEPS))
        handles2.append(plt.Line2D([0], [0], color=color, lw=4))
        labels2.append(full_text)

    # 创建并添加第二个图例（BPM步法）
    legend2 = ax2.legend(handles2, labels2,
                         bbox_to_anchor=(step_legend_x, step_legend_y),
                         loc='upper left',
                         fontsize=8,
                         title="BPM区间对应步法",
                         title_fontsize=10)

    
    bpm_img_name = f"BPM可视化_优化版_{audio_file_name.replace('.wav', '').replace('.mp3', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout(pad=4.0)
    plt.savefig(bpm_img_name, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ BPM可视化图表已保存：{bpm_img_name}")

    print(f"\n🎉 所有流程完成！")