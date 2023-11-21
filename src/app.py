import os
import threading
from typing import Optional

import noisereduce as nr  # type: ignore
import numpy as np
import PySimpleGUIQt as sg  # type: ignore
import pyworld as pw  # type: ignore
import torch
from resemblyzer import VoiceEncoder  # type: ignore

from model import Generator
from realtime_vc import RealtimeVC
from util import logsp_norm, logsp_unnorm, world_join, world_split

# # # 設定する変数 # # # # # # # # # # # # # # # #

# サンプルレート
SAMPLE_RATE = 16_000

# モデル構造
DIM_NECK = 32
DIM_EMB = 256
DIM_PRE = 512
FREQ = 32

# バッチサイズ
BATCH = 3

# 使用するモデル
MODEL_PATH = os.path.join(os.getcwd(), "models/world_autovc_jp_step001800.pth")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_all_mic_and_speaker(
    rvc: RealtimeVC
) -> tuple[list[str], Optional[str], Optional[str]]:
    """UI表示のためにマイクスピーカーを列挙

    Args:
        rvc (RealtimeVC): ボイスチェンジャー関連

    Returns:
        list[str]: マイク，スピーカの一覧
        str:
    """
    devices = []
    default_input = rvc.audio.get_default_input_device_info()
    default_output = rvc.audio.get_default_output_device_info()
    default_in = None
    default_out = None
    for i in range(rvc.audio.get_device_count()):
        data = rvc.audio.get_device_info_by_index(i)
        str_dev = f'{data["index"]}: {data["name"]}'
        if default_input["index"] == data["index"]:
            default_in = str_dev
        if default_output["index"] == data["index"]:
            default_out = str_dev
        devices.append(str_dev)
    return devices, default_in, default_out


def device_str_to_index(device_str: str) -> int:
    idx = device_str.split(":")[0]
    return int(idx)


def create_ui(rvc: RealtimeVC) -> sg.Window:
    devices, default_in, default_out = get_all_mic_and_speaker(rvc)
    layout_tab1 = [
        [
            sg.Text("変換元話者情報(npy):", font=("Helvetica", 12)),
            sg.InputText("", size=(30, 1), font=("Helvetica", 15), key="T1_EMB_FROM"),
            sg.Text("→", font=("Helvetica", 20)),
            sg.Text("変換先話者情報(npy):", font=("Helvetica", 12)),
            sg.InputText("", size=(30, 1), font=("Helvetica", 15), key="T1_EMB_TO"),
        ],
        [
            sg.Button(
                "実行",
                size=(30, 2),
                font=("Helvetica", 15),
                pad=((10, 10), (10, 10)),
                key="T1_VC_RUN",
            ),
            sg.Button(
                "停止",
                size=(30, 2),
                font=("Helvetica", 15),
                pad=((10, 10), (10, 10)),
                key="T1_VC_STOP",
            ),
        ],
    ]
    layout_tab2 = [
        [
            sg.Text("登録名:", font=("Helvetica", 12)),
            sg.InputText("", size=(30, 1), font=("Helvetica", 15), key="T2_EMB_NAME"),
            sg.Text("出力ディレクトリ:", font=("Helvetica", 12)),
            sg.InputText("", size=(30, 1), font=("Helvetica", 15), key="T2_EMB_DIR"),
        ],
        [
            sg.Button(
                "録音",
                size=(30, 2),
                font=("Helvetica", 15),
                pad=((10, 10), (10, 10)),
                key="T2_REC_RUN",
            ),
            sg.Button(
                "録音完了",
                size=(30, 2),
                font=("Helvetica", 15),
                pad=((10, 10), (10, 10)),
                key="T2_REC_STOP",
            ),
        ],
    ]
    layout = [
        [
            sg.Text(
                "WorldAutoVC",
                size=(20, 2),
                justification="left",
                font=("Helvetica", 20),
            ),
            sg.Text(
                "",
                size=(20, 2),
                justification="left",
                font=("Helvetica", 20),
                text_color="red",
                key="REC_TEXT",
            ),
        ],
        [
            sg.TabGroup(
                [
                    [
                        sg.Text("マイク: ", font=("Helvetica", 12)),
                        sg.InputCombo(
                            devices,
                            default_value=default_in,
                            size=(30, 1),
                            font=("Helvetica", 15),
                            key="MIC_CONFIG",
                        ),
                        sg.Text("スピーカー: ", font=("Helvetica", 12)),
                        sg.InputCombo(
                            devices,
                            default_value=default_out,
                            size=(30, 1),
                            font=("Helvetica", 15),
                            key="SPEAKER_CONFIG",
                        ),
                    ],
                    [
                        sg.Checkbox(
                            "ノイキャン", key="USE_DENOISE", font=("Helvetica", 15)
                        ),
                    ],
                    [
                        sg.Tab("推論", layout_tab1, key="TAB1"),
                        sg.Tab("登録", layout_tab2, key="TAB2"),
                    ],
                ],
                change_submits=True,
                key="TAB_GROUP",
            )
        ],
    ]
    return sg.Window("タイトル", layout)


def get_synthe(
    model: Generator,
    emb_src: np.ndarray,
    src_f0_logmean: float,
    src_f0_logstd: float,
    emb_tgt: np.ndarray,
    tgt_f0_logmean: float,
    tgt_f0_logstd: float,
    device: torch.device,
    use_noisereduce: bool,
):
    """
    音声変換の処理
    """

    _sembs = np.array([emb_src for _ in range(BATCH)])
    _tembs = np.array([emb_tgt for _ in range(BATCH)])

    def analysis_resynthesis(signal):
        nonlocal _sembs, _tembs

        signal /= 32767.0

        if use_noisereduce:
            signal = nr.reduce_noise(y=signal, sr=SAMPLE_RATE)

        wav_len = signal.shape[0]
        pad_size = (256 * 20 - 1) * (
            (signal.shape[0] // (256 * 20 - 1)) + 1
        ) - signal.shape[0]
        signal = np.pad(signal, [0, pad_size], "constant")

        f0, _, sp_in, ap = world_split(signal)
        sp = sp_in.reshape((-1, 64, 513))
        sp = logsp_norm(np.log(sp))

        mels = torch.from_numpy(sp.astype(np.float32)).clone()
        sembs = torch.from_numpy(_sembs.astype(np.float32)).clone()
        tembs = torch.from_numpy(_tembs.astype(np.float32)).clone()

        with torch.inference_mode():
            m = mels.to(device)
            se = sembs.to(device)
            te = tembs.to(device)
            _, mel_outputs_postnet, _ = model(m, se, te)

        sp_out = np.exp(
            logsp_unnorm(mel_outputs_postnet.to("cpu").detach().numpy().copy())
        )
        sp_out = sp_out.reshape((-1, 513)).astype(np.double)

        f0_out = np.exp(
            (np.ma.log(f0).data - src_f0_logmean) / src_f0_logstd * tgt_f0_logstd
            + tgt_f0_logmean
        )

        wav_out = world_join(f0_out, sp_out, ap)
        wav_out = wav_out[:wav_len]

        wav_out *= 32767.0

        return wav_out

    return analysis_resynthesis


if __name__ == "__main__":
    print("# # # # 変換モデルロード中 # # # #")
    print("load:", MODEL_PATH)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Generator(DIM_NECK, DIM_EMB, DIM_PRE, FREQ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    encoder = VoiceEncoder()
    print("ロード完了")

    rvc = RealtimeVC(
        sample_rate=SAMPLE_RATE,
        input_buffer_size=(256 * 20 - 1) * BATCH - 1,
        output_buffer_size=(256 * 20 - 1) * BATCH - 1,
    )

    window = create_ui(rvc)

    thread1: Optional[threading.Thread] = None
    finish_flag: list[bool] = [False]

    all_signals: list[np.ndarray] = []

    print("アプリ起動")
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == "T1_VC_RUN":
            r_emb_from: str = values["T1_EMB_FROM"].replace("file://", "")
            r_emb_to: str = values["T1_EMB_TO"].replace("file://", "")
            if (
                os.path.isfile(r_emb_from)
                and os.path.isfile(r_emb_to)
                and r_emb_from.split(".")[-1] == "npz"
                and r_emb_to.split(".")[-1] == "npz"
                and r_emb_from.split(".")[-2] == "wavc"
                and r_emb_to.split(".")[-2] == "wavc"
            ):
                try:
                    src = np.load(r_emb_from)
                    emb_src = src["embed"]
                    src_f0_logmean = src["f0_logmean"]
                    src_f0_logstd = src["f0_logstd"]
                    tgt = np.load(r_emb_to)
                    emb_tgt = tgt["embed"]
                    tgt_f0_logmean = tgt["f0_logmean"]
                    tgt_f0_logstd = tgt["f0_logstd"]
                except Exception as e:
                    print(e)
                    sg.popup("話者の特徴量が読み込めません")

                print("npyファイル読み込み完了")

                input_device_index = device_str_to_index(values["MIC_CONFIG"])
                output_device_index = device_str_to_index(values["SPEAKER_CONFIG"])
                use_denoise: bool = values["USE_DENOISE"]

                def run():
                    finish_flag[0] = False
                    rvc.run(
                        get_synthe(
                            model,
                            emb_src,
                            src_f0_logmean,
                            src_f0_logstd,
                            emb_tgt,
                            tgt_f0_logmean,
                            tgt_f0_logstd,
                            device,
                            use_denoise,
                        ),
                        input_device_index=input_device_index,
                        output_device_index=output_device_index,
                        finish_flag=finish_flag,
                    )

                if thread1 is None:
                    thread1 = threading.Thread(target=run)
                    thread1.start()
                    window["REC_TEXT"].update("REC●")
                else:
                    sg.popup("一度停止してください")
            else:
                sg.popup("npyファイルが発見できませんでした")

        if event == "T1_VC_STOP":
            print("vc stop")
            finish_flag[0] = True
            if thread1 is not None:
                thread1.join()
            thread1 = None
            window["REC_TEXT"].update("")

        if event == "T2_REC_RUN":
            r_emb_name: str = values["T2_EMB_NAME"]
            r_emb_dir: str = values["T2_EMB_DIR"].replace("file://", "")
            if (
                len(r_emb_dir) > 0
                and os.path.isdir(r_emb_dir)
                and len(r_emb_name) > 0
                and "/" not in r_emb_name
                and "." not in r_emb_name
            ):
                save_path = os.path.join(r_emb_dir, f"{r_emb_name}.wavc.npz")
                input_device_index = device_str_to_index(values["MIC_CONFIG"])
                use_denoise = values["USE_DENOISE"]

                all_signals = [np.array([])]

                def recode(signal):
                    global all_signals
                    signal /= 32767.0
                    all_signals[0] = np.append(all_signals[0], signal)

                def run():
                    finish_flag[0] = False
                    rvc.recording(
                        recode,
                        input_device_index=input_device_index,
                        finish_flag=finish_flag,
                    )

                if thread1 is None:
                    thread1 = threading.Thread(target=run)
                    thread1.start()
                    window["REC_TEXT"].update("REC●")
                else:
                    sg.popup("一度停止してください")

            else:
                sg.popup("ディレクトリが存在しないか，名前が不適切です")

        if event == "T2_REC_STOP":
            r_emb_name = values["T2_EMB_NAME"]
            r_emb_dir = values["T2_EMB_DIR"].replace("file://", "")
            finish_flag[0] = True
            if thread1 is not None:
                thread1.join()
                all_signal = all_signals[0]
                all_signals = [np.array([])]

                print("# # # # 音声解析中 # # # #")
                save_path = os.path.join(r_emb_dir, f"{r_emb_name}.wavc.npz")
                embed = encoder.embed_utterance(all_signal)
                wav = all_signal.astype(np.float64)
                f0, _ = pw.harvest(wav, SAMPLE_RATE)
                logf0 = np.log(f0[np.nonzero(f0)])
                f0_logmean = logf0.mean()
                f0_logstd = logf0.std()
                np.savez_compressed(
                    save_path, embed=embed, f0_logmean=f0_logmean, f0_logstd=f0_logstd
                )
                print("保存完了:", save_path)
            thread1 = None
            window["REC_TEXT"].update("")

        if event == "TAB_GROUP":
            finish_flag[0] = True
            if thread1 is not None:
                thread1.join()
            thread1 = None

    window.close()
    finish_flag[0] = True
    if thread1 is not None:
        thread1.join()
    thread1 = None
    rvc.stop(is_all=True)
